"""Module generates training batches and trains NTP model"""

from collections import defaultdict, OrderedDict
import random
import copy
import sys
import argparse
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from ntp.modules.kb import initialize_nkb, augment_with_templates, kb_ids2known_facts
from ntp.modules.prover import prove, representation_match, is_tensor
from ntp.modules.gradient import auto_gradient, loss_dict
from ntp.modules.eval import eval_fact_accuracy, eval_rule_accuracy, prop_rules, weighted_prop_rules, weighted_precision, auc_helper, eval_rank, closest_rule_score
from ntp.util.util_kb import is_parameter, rule2string, Atom, normalize, relationship_id_to_symbol
from ntp.util.util_training import TrainingState
from ntp.util.util_diag import identify_relevant_goals, decode_proof, decode_proofs, find_correct_proof_indices, calculate_gradient, score_proof, closest_unification_score
from ntp.util.util_eval import decode_rules
from ntp.util.util_data import load_conf

def corrupt_goal(goal, constant_ids, known_facts, args=[1], tries=100):
    """Given a fact from the training set, change an input constant at a given position 
    to another randomly chosen constant in such a way that the resulting fact is not in the training set"""

    if tries == 0:
        print("WARNING: Could not corrupt", goal)
        return None
    else:
        goal_corrupted = copy.deepcopy(goal)
        for arg in args:
            corrupt = constant_ids[random.randint(
                0, len(constant_ids) - 1)]
            goal_corrupted[arg] = corrupt

        if tuple(goal_corrupted) in known_facts:
            return corrupt_goal(goal, constant_ids, known_facts, args, tries-1)
        else:
            return goal_corrupted


def gen_train_batches(fact_structs, kb_ids, batch_size, num_corruptions, constant_ids, known_facts):
    """From list of training facts, generate training batches. 
    In terms of symbol ids, not embeddings, so very cheap to precompute"""
    train_batches = []

    for struct in fact_structs:
        struct_batches = []
        struct_order = len(struct[0]) - 1
        facts = kb_ids[struct][0]
        n_facts = len(facts[0])
        fact_ids = list(range(0, n_facts))

        random.shuffle(fact_ids)
        
        facts = [[facts[i][j] for i in range(struct_order + 1)] for j in range(n_facts)]

        all_facts = []
        targets = []
        mask_indices = []

        for fact_id in fact_ids:
            fact = facts[fact_id]
            all_facts.append(fact)
            targets.append(1.0)
            # Mark location of goal facts present in kb so they can be masked by prover
            mask_indices.append([fact_id, len(mask_indices) % batch_size])


            for constant_pos in range(struct_order):
                for i in range(num_corruptions):
                    corrupted_fact = corrupt_goal(fact, constant_ids, known_facts, args=[constant_pos + 1])
                    if corrupted_fact is not None:
                        all_facts.append(corrupted_fact)
                        targets.append(0.0)
                        mask_indices.append(None)

        n_facts_total = len(all_facts)

        def gen_batch(list_dict, slice):
            batch = {key: value[slice] for key, value in list_dict.items()}
            return batch

        list_dict = {"goal": all_facts, "target": targets, "mask_indices": mask_indices}

        facts_processed = 0
        while facts_processed + batch_size < n_facts_total:
            batch = gen_batch(list_dict, slice(facts_processed, facts_processed + batch_size))
            struct_batches.append(batch)
            facts_processed += batch_size
        
        if facts_processed < n_facts_total:
            batch = gen_batch(list_dict, slice(facts_processed, n_facts_total))
            struct_batches.append(batch)

        for batch in struct_batches:
            batch["goal"] = [[fact[k] for fact in batch["goal"]] for k in range(struct_order + 1)]
            batch["mask_indices"] = [value for value in batch["mask_indices"] if value is not None]
            batch["batch_size"] = len(batch["goal"][0])
            batch["struct"] = struct
            batch["n_facts_struct"] = n_facts

        train_batches += struct_batches
    return train_batches

def train_model(kb, rule_templates, conf, relationships=None, test_kb=None):
    """Main function in the repository. Given a kb of facts and templates, train NTP model. 

    Args:
        kb: list of training facts
        rule_templates: list of rule templates
        conf: dict that contains configuration information
        relationships: gold relationship structure
        test_kb: list of test facts
    Returns: 
        Rules, confidences, and a dict of evaluation metrics (recall, etc)
    """

    if "seed" in conf["training"]:
        tf.set_random_seed(conf["training"]["seed"])

    input_size = conf["model"]["input_size"]
    l2 = conf["model"]["l2"]
    k_max = conf["model"]["k_max"]
    max_depth = conf["model"]["max_depth"]
    loss_type = conf["model"]["loss_type"]
    loss_parameters = conf["model"]["loss_parameters"]

    report_interval = conf["logging"]["report_interval"]
    log_dir = conf["logging"]["log_dir"]
    verbose = conf["logging"]["verbose"]

    num_epochs = conf["training"]["num_epochs"]
    clip = conf["training"]["clip"]
    learning_rate = conf["training"]["learning_rate"]
    lr_decay_type = conf["training"]["lr_decay_type"]
    lr_decay_rate = conf["training"]["lr_decay_rate"]
    epsilon = conf["training"]["epsilon"]

    num_corruptions = conf["training"]["num_corruptions"]
    batch_size = conf["training"]["batch_size"]

    kb = augment_with_templates(kb, rule_templates)
    nkb, kb_ids, vocab, emb, predicate_ids, constant_ids = \
        initialize_nkb(kb, input_size)

    true_predicate_ids = {order: [pred for pred in pred_order_ids if not is_parameter(vocab.id2sym[pred])] for (order, pred_order_ids) in predicate_ids.items()}

    known_facts = kb_ids2known_facts(kb_ids)

    fact_structs = [rule for rule in kb_ids if len(rule) == 1]

    training_state = TrainingState(learning_rate, lr_decay_type, lr_decay_rate)

    optim = tf.train.AdamOptimizer(learning_rate=training_state.lr_variable,
                                       epsilon=epsilon)

    loss_function = loss_dict[loss_type]

    summary_writer = tf.contrib.summary.create_file_writer(log_dir)

    start_time = timer()
    for epoch in range(num_epochs):
        train_batches = gen_train_batches(fact_structs, kb_ids, batch_size, num_corruptions, constant_ids, known_facts)


        if epoch % report_interval == 0 or epoch == num_epochs - 1:
            print("epoch" + str(epoch))

            log_dict = OrderedDict()
            log_dict["fact_accuracy"] = eval_fact_accuracy(train_batches, emb, kb_ids,
                                                                 vocab, k_max, max_depth)
            if relationships is not None:
                log_dict["rule_recall"] = eval_rule_accuracy(kb_ids, true_predicate_ids, constant_ids, emb, vocab, relationships)
                log_dict["closest_rule_score"] = closest_rule_score(relationships, emb, kb_ids, vocab, fact_structs)
                log_dict["closest_unification_score"] = closest_unification_score(relationships, emb, kb_ids, vocab)

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for key in log_dict:
                    tf.contrib.summary.scalar(key, log_dict[key], step=epoch)

            if verbose:
                end_time = timer()

                for key in log_dict:
                    print(key + ": " + str(log_dict[key]))
                print("time per epoch: " + str((end_time - start_time) / report_interval))
                start_time = timer()

                                               
        for j, batch in enumerate(train_batches):

            goal = tf.constant(batch["goal"])

            mask_indices = tf.constant(batch["mask_indices"])
            loss_parameters["epoch"] = epoch

            with tf.device("/device:GPU:0"):
                # 1 for facts from training set, 0 for corruptions
                target = tf.constant(batch["target"], dtype=tf.float32)
                # Mask is necessary to hide the goal fact (which is part of the training set) from the prover
                base_mask = tf.ones([batch["n_facts_struct"], batch["batch_size"]], dtype=tf.float32)
                updates = -1.0 * tf.ones(len(batch["mask_indices"]), dtype=tf.float32)
                batch_mask = tf.transpose(tf.constant(
                    base_mask + tf.scatter_nd(mask_indices, updates, base_mask.shape)))   

                grad = auto_gradient(loss_function, goal, target, emb, kb_ids,
                                         vocab, batch["struct"], batch_mask, l2, k_max, max_depth, epsilon, loss_parameters)

            if clip is not None:
                capped_gradients = tf.clip_by_value(grad, clip[0], clip[1])

            grad_and_var = [(capped_gradients, emb)]

            optim.apply_gradients(grad_and_var)

            training_state.update_iteration()

    eval_dict = OrderedDict()

    rules, confidences = decode_rules(
        kb_ids, true_predicate_ids, constant_ids, emb, vocab, verbose=verbose, print_k=20)

    # if relationships are known, can calculate relationship learning accuracy
    if relationships is not None:

        print("closest rule score: " + str(closest_rule_score(relationships, emb, kb_ids, vocab, fact_structs)))
        print("closest unification score: " + str(closest_unification_score(relationships, emb, kb_ids, vocab)))

        eval_dict["prop_rules"] = prop_rules(relationships, rules, confidences)
        eval_dict["weighted_prop_rules"] = weighted_prop_rules(
            relationships, rules, confidences)
        eval_dict["weighted_precision"] = weighted_precision(
            relationships, rules, confidences)
        eval_dict["auc_helper"] = auc_helper(relationships, rules, confidences)

    if test_kb is not None:

        eval_dict["MRR"], eval_dict["randomMRR"], eval_dict["fact-roc-auc"] = \
        eval_rank(emb, kb_ids, test_kb, known_facts, vocab, constant_ids, batch_size, k_max, max_depth, eval_dict)

    tf.reset_default_graph()

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for key in eval_dict:
            tf.contrib.summary.scalar(key, eval_dict[key], step=num_epochs)

    return rules, confidences, eval_dict
