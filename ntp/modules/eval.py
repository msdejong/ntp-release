"""Contains functions to evaluate performance of NTP model"""

import copy
import itertools

import numpy as np
from sklearn import metrics
import tensorflow as tf

from ntp.modules.prover import prove
from ntp.modules.gradient import retrieve_top_k
from ntp.modules.kb import kb2ids, partition, kb2nkb
from ntp.modules.nunify import l2_sim_np
from ntp.util.util_eval import decode_rules, harmonic


def auc_helper(relationships, run_rules, run_confidences):
    """
    Calculates auc-roc measuring the recall and precision of learned rules relative to a set of existing relationships
    """    
    targets = []
    scores = []

    for head, body in relationships.items():
        targets.append(1.0)
        if [head, body] in run_rules:
            index = run_rules.index([head, body])
            scores.append(run_confidences[index])
        else:
            scores.append(0.0)
    

    for j, rule in enumerate(run_rules):
        if rule[0] in rule[1]:
            continue

        # Append incorrect rules with score of 0
        if rule[0] in relationships:
            if relationships[rule[0]] == rule[1]:
                continue
        
        targets.append(0.0)
        scores.append(run_confidences[j])

    return targets, scores

def prop_rules(relationships, run_rules, run_confidences, threshold=0.0, allow_reverse=False):
    """
    From a list of rules, calculates the proportion of relationships injected into the data that are present in the list of learned rules
    """

    relationships_found = 0

    for head, body in relationships.items():
        if [head, body] in run_rules:
            # This finds the first such rule, rules should be sorted by confidence to make sure it's the highest confidence of those rules. 
            index = run_rules.index([head, body])
            if run_confidences[index] > threshold:
                relationships_found += 1
        elif allow_reverse == True and len(body) == 1 and [head, body] in run_rules:
            index = run_rules.index([head, body])
            if run_confidences[index] > threshold:
                relationships_found += 1

    return relationships_found / len(relationships)

def weighted_prop_rules(relationships, run_rules, run_confidences, threshold=0.0, allow_reverse=False):
    """
    From a list of rules and confidences, calculates the proportion of relationships injected into the data that are present in the list of learned rules, weighted by rule confidence
    Args:
        relationships: relationships injected into the data
        run_rules: learned rules 
        run_confidences: confidences corresponding to those rules. Rules should be sorted by confidence, from high to low. 
        threshold: minimum confidence under which a rule is not considered
        allow_reverse: whether or not a rule 1>0 is accepted if the true rule is 0>1
    Returns: 
        Proportion of relationships injected into the data that are present in the list of learned rules, weighted by confidence
    """
    relationships_found = 0

    for head, body in relationships.items():
        if [head, body] in run_rules:
            # This finds the first such rule, rules should be sorted by confidence to make sure it's the highest confidence of those rules. 
            index = run_rules.index([head, body])
            if run_confidences[index] > threshold:
                relationships_found += run_confidences[index]
        elif allow_reverse == True and len(body) == 1 and [head, body] in run_rules:
            index = run_rules.index([head, body])
            if run_confidences[index] > threshold:
                relationships_found += run_confidences[index]

    return relationships_found / len(relationships)


def weighted_precision(relationships, run_rules, run_confidences, threshold=0.0, allow_reverse=False):
    """
    From a list of rules and confidences, calculates the proportion of those rules that match relationships injected into the data, weighted by confidence.
    """
    wrong_relationship_weight = 0
    total_relationship_weight = 0

    for j, rule in enumerate(run_rules):
        
        # Skip rules with confidence below threshold

        if run_confidences[j] < threshold or rule[0] in rule[1]:
            continue
        
        total_relationship_weight += run_confidences[j]

        # Check if rule is correct
        if rule[0] in relationships:
            if relationships[rule[0]] == rule[1]:
                continue

        if len(rule[1]) == 1:
            body_pred = list(rule[1])[0]

            # If learning reverse rule is acceptable, check for reverse rule for rules with only one body predicate
            if allow_reverse and body_pred in relationships and relationships[body_pred] == {rule[0]}:
                continue
                
            # Learning x-->x is not wrong, technically
            elif len(rule) == 2 and rule[0] == body_pred:
                continue
            
        wrong_relationship_weight += run_confidences[j]

    if total_relationship_weight != 0:
        return (total_relationship_weight - wrong_relationship_weight) / total_relationship_weight
    else:
        return 0


def confidence_accuracy(relationships, run_rules, run_confidences, threshold=0.0, allow_reverse=False):
    """
    From a list of rules and confidences, calculates 'confidence accuracy', giving positive points for being confident and right and negative points for confident and wrong
    """

    score = 0
    for j, rule in enumerate(run_rules):
        # Skip rules with confidence below threshold
        if run_confidences[j] < threshold:
            continue
        if rule[0] in relationships:
            if relationships[rule[0]] == rule[1]:
                score += run_confidences[j]
                continue

        if len(rule) == 2:
            body_pred = list(rule[1])[0]

            if allow_reverse and relationships[body_pred] == rule[0]:
                score += run_confidences[j]
                continue

            # skip identity
            if rule[0] == body_pred:
                continue

        # if rule was not correct, add negative confidence
        score -= run_confidences[j]

    return score

def eval_batch(goal, target, emb, kb_ids, vocab, goal_struct, batch_mask, k_max, max_depth):
    """Retrieve accuracy of batch of facts relative to target, given kb of training facts"""
    kb_goal = copy.deepcopy(kb_ids)
    kb_goal['goal'] = [[row.numpy() for row in goal]]
    nkb = kb2nkb(kb_goal, emb)

    goal = [{'struct': 'goal', 'atom': 0, 'symbol': i} for i in range(len(goal_struct[0]))]

    proofs = prove(nkb, goal, goal_struct, batch_mask,
                k_max=k_max, max_depth=max_depth, vocab=vocab)

    score = np.squeeze(retrieve_top_k(proofs).numpy())
    target = target.numpy()
    result = score > 0.5
    accuracy = np.mean(result == target)
    weighted_accuracy = np.mean((target == 1) * (1 * result) + (target == 0) * (-1 * result))

    return accuracy, weighted_accuracy

def eval_fact_accuracy(batch_list, emb, kb_ids, vocab, k_max, max_depth):
    """Retrieve average accuracy of list of _train_ fact batches, given kb of training facts. """

    accuracy_list= []
    weighted_accuracy_list = []

    for j, batch in enumerate(batch_list):

        goal = tf.constant(batch["goal"])
        mask_indices = tf.constant(batch["mask_indices"])

        with tf.device("/device:GPU:0"):
            target = tf.constant(batch["target"], dtype=tf.float32)
            base_mask = tf.ones([batch["n_facts_struct"], batch["batch_size"]], dtype=tf.float32)
            updates = -1.0 * tf.ones(len(batch["mask_indices"]), dtype=tf.float32)
            batch_mask = tf.transpose(tf.constant(
                    base_mask + tf.scatter_nd(mask_indices, updates, base_mask.shape)))   
            batch_accuracy, weighted_accuracy = eval_batch(goal, target, emb, kb_ids, vocab, batch["struct"], batch_mask, k_max, max_depth)
        accuracy_list.append(batch_accuracy)
        weighted_accuracy_list.append(weighted_accuracy)

    return np.mean(accuracy_list), np.mean(weighted_accuracy_list)

def eval_rule_accuracy(kb_ids, true_predicate_ids, constant_ids, emb, vocab, relationships=None):
    """Calculate proportion of relationships that are part of the decoded rules, weighted by confidence of those rules""" 
    rules, confidences = decode_rules(kb_ids, true_predicate_ids, constant_ids, emb, vocab)
    if relationships is not None:
        return weighted_prop_rules(relationships, rules, confidences)

def closest_rule_score(relationships, emb, kb_ids, vocab, fact_structs):
    """Calculate score of best rule for each relationship and average. 
    Ignores whether there is an even better decoding for a particular rule.""" 
    emb_np = emb.numpy()
    best_scores = []
    for head, body in relationships.items():
        head_id = vocab.sym2id[head]
        body_ids = [vocab.sym2id[body_pred] for body_pred in body]
        best_score = 0
        for struct in kb_ids:
            if struct in fact_structs:
                continue
            struct_ids = kb_ids[struct]
            n_struct = len(struct_ids[0][0])
            for rule_nr in range(n_struct):
                head_score = l2_sim_np(np.expand_dims(emb_np[head_id], axis=0), np.expand_dims(emb_np[struct_ids[0][0][rule_nr]], axis=0))[0][0]
                body_score = 0
                for permutation in itertools.permutations(body_ids):
                    perm_score = 1
                    for body_atom, elem in enumerate(permutation):
                        elem_score = l2_sim_np(np.expand_dims(emb_np[elem], axis=0), np.expand_dims(emb_np[struct_ids[1 + body_atom][0][rule_nr]], axis=0))[0][0]
                        if elem_score < perm_score:
                            perm_score = elem_score
                    if perm_score > body_score:
                        body_score = perm_score
                rule_score = min(body_score, head_score)
                if rule_score > best_score:
                    best_score = rule_score
        best_scores.append(best_score)
    
    average_score_closest = np.mean(best_scores)
    return average_score_closest


def eval_rank(emb, kb_ids, test_kb, known_facts, vocab, constant_ids, batch_size, k_max, max_depth, eval_dict):
    """For a set of true test facts, corrupt those test facts in every possible way that is not in the training set. 
    Calculate the scores of all such facts, and calculate ranking measures of the true test fact in that list of corrupted facts""" 
    
    id_dict = kb2ids(partition(test_kb))[0]


    counter, MRR, randomMRR = 0, 0, 0
    targets, auc_scores = [], []

    for struct in id_dict:
        struct_ids = id_dict[struct][0]    
        n_test_struct = len(struct_ids[0])

        for i in range(n_test_struct):
            symbols_in_fact = len(struct_ids)
            fact = np.array([struct_ids[k][i] for k in range(symbols_in_fact)])
            fact_corruptions = [[] for q in range(symbols_in_fact)]
            for k in range(1, symbols_in_fact):
                corrupt_fact = copy.deepcopy(fact)
                for constant in constant_ids: 
                    corrupt_fact[k] = constant
                    if tuple(corrupt_fact) not in known_facts:
                        for j in range(symbols_in_fact):
                            fact_corruptions[j].append(corrupt_fact[j])
                    else:
                        continue
        

                combined = [[fact[q]] + fact_corruptions[q] for q in range(symbols_in_fact)]
                n_combined = len(combined[0])
                batches = []

                batch_pos = 0
                while batch_pos + batch_size + 1 <= n_combined:
                    new_batch = [combined[q][batch_pos:batch_pos + batch_size] for q in range(symbols_in_fact)]
                    batches.append(new_batch)
                    batch_pos += batch_size

                new_batch = [combined[q][batch_pos:] for q in range(symbols_in_fact)]
                batches.append(new_batch)
                scores = []

                for batch in batches:
                    
                    kb_goal = copy.deepcopy(kb_ids)
                    kb_goal['goal'] = [[np.array(row) for row in batch]]
                    nkb = kb2nkb(kb_goal, emb)

                    goal = [{'struct': 'goal', 'atom': 0, 'symbol': i} for i in range(symbols_in_fact)]

                    cur_batch_size = len(batch[0])
                    n_struct = len(kb_ids[struct][0][0])

                    test_mask = tf.ones([cur_batch_size, n_struct], dtype=tf.float32)

                    proofs = prove(nkb, goal, struct, test_mask,
                                k_max=k_max, max_depth=max_depth, vocab=vocab)
                    
                    batch_scores = retrieve_top_k(proofs).numpy().squeeze()
                    if batch_scores.size == 1:
                        batch_scores = [batch_scores]
                    else:
                        batch_scores = list(batch_scores)
                    scores.extend(batch_scores)


                fact_score = scores[0]

                # Duplicate real fact for each corruption
                targets.extend([1.0] * (len(scores) - 1) + [0.0] * (len(scores) - 1))
                auc_scores.extend([fact_score] *  (len(scores) - 1) + scores[1:])

                scores = sorted(scores)
                reversed_scores = scores[::-1]

                # This model frequently has ties. We resolve those by taking the average of optimistic and
                # pessimistic tiebreaking.
                rank = reversed_scores.index(fact_score) + 1
                rank_pess = len(scores) - scores.index(fact_score)
                rank = 0.5 * (rank + rank_pess)

                counter += 1.0
                MRR += 1.0 / rank
                randomMRR += harmonic(len(scores))/len(scores)

    MRR /= counter
    randomMRR /= counter

    roc_auc = metrics.roc_auc_score(targets, auc_scores)

    print("roc-auc: " + str(roc_auc))

    return MRR, randomMRR, roc_auc

