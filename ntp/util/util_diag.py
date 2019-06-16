"""utility functions to investigate what the NTP model is learning"""

import numpy as np
import tensorflow as tf 

from ntp.modules.nunify import l2_sim_np

def closest_unification_score(relationships, emb, kb_ids, vocab):
    """Calculate best score, averaged over all relationships, of directly unifying head fact with one of the body facts""" 
    emb_np = emb.numpy()
    best_scores = []
    for head, body in relationships.items():
        head_id = vocab.sym2id[head]
        body_ids = [vocab.sym2id[body_pred] for body_pred in body]
        best_score = 0
        for body_id in body_ids:
            id_score = l2_sim_np(np.expand_dims(emb_np[head_id], axis=0), np.expand_dims(emb_np[body_id], axis=0))[0][0]
            if id_score > best_score:
                best_score = id_score
        best_scores.append(best_score)    
    average_score_closest = np.mean(best_scores)
    return average_score_closest


### Functions below need to be updated for new formulation of goals / rhs ###
#############################################################

def identify_relevant_goals(goal, relationship_ids, fact_dict):

    predicates = goal[0].numpy()
    constants = goal[1].numpy()

    active_indices = []
    
    for i, predicate in enumerate(predicates):
        if predicate in relationship_ids:
            if relationship_ids[predicate] in fact_dict[constants[i]]:
                active_indices.append(i)
    
    return np.array(active_indices, dtype=np.int32)

def decode_proof(index, kb, n_facts):
    
    decoding = []
    if index < n_facts:
        decoding.append(kb[index][0].predicate + "(" + kb[index][0].arguments[0] + ")")
    else:
        
        proof_index = index - n_facts

        fact_id = proof_index % n_facts
        rule_id = int((proof_index - fact_id) / n_facts)

        decoding.append(kb[fact_id][0].predicate + "(" + kb[fact_id][0].arguments[0] + ")")
        decoding.append("Rule " + str(rule_id))
    
    return decoding


def decode_proofs(indices, kb, n_facts):
    decoded_proofs = [decode_proof(index, kb, n_facts) for index in indices]
    return decoded_proofs

# needs to be changed if multiple relationships

def find_correct_proof_indices(goal, n_rules, n_facts, relationship_ids, inverse_kb):
    predicates = goal[0].numpy()
    constants = goal[1].numpy()

    n_goals = tf.size(predicates)
    base_predicates = [relationship_ids[predicates[i]][0] for i in range(n_goals)]
    fact_indices = [inverse_kb[(base_predicates[i], constants[i])] for i in range(n_goals)]

    correct_indices = np.zeros((n_rules, n_goals), dtype=np.int32)

    for i in range(n_goals):
        for j in range(n_rules):
            correct_indices[j, i] = int((j + 1) * n_facts + fact_indices[i])
    return correct_indices


def calc_proof_gradient(emb1, emb2, score, target):
        
        if target == 1.0:
            proof_gradient = - (emb1 - emb2) / np.log(score)
        elif target == 0.0:
            proof_gradient = score / (1 - score) * (emb1 - emb2) / np.log(score)
           
        return proof_gradient


def calculate_gradient(predicates, constants, indices, emb, targets, shape, n_facts, kb_ids, goal_struct):

    rule_struct = (('p0', 'X0'), ('p1', 'X0'))

    gradient = np.zeros(shape)

    for i, index in enumerate(indices):
        fact_id = index % n_facts

        fact_predicate_id = kb_ids[goal_struct][0][0][fact_id]
        fact_constant_id = kb_ids[goal_struct][0][1][fact_id]

        fact_predicate_embedding = emb[fact_predicate_id]
        fact_constant_embedding = emb[fact_constant_id]

        goal_predicate_embedding = emb[predicates[i]]
        goal_constant_embedding = emb[constants[i]]

        constant_score = np.exp(-np.linalg.norm(fact_constant_embedding - goal_constant_embedding))

        if index < n_facts:

            pred_score = np.exp(-np.linalg.norm(fact_predicate_embedding - goal_predicate_embedding))

            if pred_score > constant_score:
                proof_grad = calc_proof_gradient(fact_constant_embedding, goal_constant_embedding, constant_score, targets[i])
                gradient[fact_constant_id, :] += proof_grad
                gradient[constants[i], :] -= proof_grad

            else:
                proof_grad = calc_proof_gradient(fact_predicate_embedding, goal_predicate_embedding, pred_score, targets[i])
                gradient[fact_predicate_id, :] += proof_grad
                gradient[predicates[i], :] -= proof_grad

        else:

            rule_id = int((index - fact_id) / n_facts) - 1
            rule_goal_predicate_id = kb_ids[rule_struct][0][0][rule_id]
            rule_body_predicate_id = kb_ids[rule_struct][1][0][rule_id]

            rule_goal_predicate_embedding = emb[rule_goal_predicate_id]
            rule_body_predicate_embedding = emb[rule_body_predicate_id]

            pred_score_1 = np.exp(-np.linalg.norm(fact_predicate_embedding - rule_body_predicate_embedding))
            pred_score_2 = np.exp(-np.linalg.norm(goal_predicate_embedding - rule_goal_predicate_embedding))

            worst_unification = np.argmin([constant_score, pred_score_1, pred_score_2])
            if worst_unification == 0:
                proof_grad = calc_proof_gradient(fact_constant_embedding, goal_constant_embedding, constant_score, targets[i])
                gradient[fact_constant_id, :] += proof_grad
                gradient[constants[i]] -= proof_grad
            elif worst_unification == 1:
                proof_grad = calc_proof_gradient(fact_predicate_embedding, rule_body_predicate_embedding, pred_score_1, targets[i])
                gradient[fact_predicate_id, :] += proof_grad
                gradient[rule_body_predicate_id, :] -= proof_grad
            elif worst_unification == 2:
                proof_grad = calc_proof_gradient(goal_predicate_embedding, rule_goal_predicate_embedding, pred_score_2, targets[i])
                gradient[predicates[i], :] += proof_grad
                gradient[rule_goal_predicate_id, :] -= proof_grad

    return gradient

def score_proof(index, predicate, constant, emb, kb_ids, n_facts, goal_struct):

    rule_struct = (('p0', 'X0'), ('p1', 'X0'))

    fact_id = index % n_facts

    fact_predicate_id = kb_ids[goal_struct][0][0][fact_id]
    fact_constant_id = kb_ids[goal_struct][0][1][fact_id]

    fact_predicate_embedding = emb[fact_predicate_id]
    fact_constant_embedding = emb[fact_constant_id]

    goal_predicate_embedding = emb[predicate]
    goal_constant_embedding = emb[constant]

    constant_score = np.exp(-np.linalg.norm(fact_constant_embedding - goal_constant_embedding))

    if index < n_facts:

        pred_score = np.exp(-np.linalg.norm(fact_predicate_embedding - goal_predicate_embedding))
        proof_score = np.min([constant_score, pred_score])

    else:

        rule_id = int((index - fact_id) / n_facts) - 1
        rule_goal_predicate_id = kb_ids[rule_struct][0][0][rule_id]
        rule_body_predicate_id = kb_ids[rule_struct][1][0][rule_id]

        rule_goal_predicate_embedding = emb[rule_goal_predicate_id]
        rule_body_predicate_embedding = emb[rule_body_predicate_id]

        pred_score_1 = np.exp(-np.linalg.norm(fact_predicate_embedding - rule_body_predicate_embedding))
        pred_score_2 = np.exp(-np.linalg.norm(goal_predicate_embedding - rule_goal_predicate_embedding))


        proof_score = np.min([constant_score, pred_score_1, pred_score_2])

    return proof_score