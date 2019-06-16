#!/usr/bin/env python

from collections import defaultdict, OrderedDict
import random
import copy
import sys
import argparse
import os
import re
from timeit import default_timer as timer
import datetime

import numpy as np
from sklearn import metrics
import tensorflow as tf

from ntp.modules.train import gen_train_batches
from ntp.modules.kb import initialize_nkb, augment_with_templates, rule2struct, kb_ids2known_facts
from ntp.modules.prover import prove, representation_match, is_tensor
from ntp.modules.gradient import manual_gradient, auto_gradient, loss_dict
from ntp.modules.eval import eval_fact_accuracy, eval_rule_accuracy, prop_rules, weighted_prop_rules, weighted_precision, auc_helper, eval_rank, closest_rule_score
from ntp.modules.nunify import l2_sim_np
from ntp.util.util_kb import is_parameter, rule2string, Atom, normalize, load_from_list, load_from_file, relationship_id_to_symbol
from ntp.util.util_training import TrainingState
from ntp.util.util_diag import identify_relevant_goals, decode_proof, decode_proofs, find_correct_proof_indices, calculate_gradient, score_proof, closest_unification_score
from ntp.util.util_eval import decode_rules
from ntp.util.util_data import load_conf, GeneratorWithRestart
from ntp.modules.generate import gen_simple, gen_relationships, write_data, write_relationships, write_simple_templates, gen_test_kb, gen_constant_dict, count_active




# @profile
def train_model(kb, rule_templates, conf, relationships=None, test_kb=None, r=1.0):

    if "seed" in conf["training"]:
        tf.set_random_seed(conf["training"]["seed"])

    input_size = conf["model"]["input_size"]
    l2 = conf["model"]["l2"]
    k_max = conf["model"]["k_max"]
    max_depth = conf["model"]["max_depth"]
    loss_type = conf["model"]["loss_type"]
    loss_parameters = conf["model"]["loss_parameters"]


    num_epochs = conf["training"]["num_epochs"]
    clip = conf["training"]["clip"]
    learning_rate = conf["training"]["learning_rate"]
    lr_decay_type = conf["training"]["lr_decay_type"]
    lr_decay_rate = conf["training"]["lr_decay_rate"]
    epsilon = conf["training"]["epsilon"]

    report_interval = conf["logging"]["report_interval"]
    log_dir = conf["logging"]["log_dir"]
    verbose = conf["logging"]["verbose"]

    num_corruptions = conf["training"]["num_corruptions"]
    batch_size = conf["training"]["batch_size"]

    kb = augment_with_templates(kb, rule_templates)
    nkb, kb_ids, vocab, emb, predicate_ids, constant_ids = \
        initialize_nkb(kb, input_size)
    true_predicate_ids = {order: [pred for pred in pred_order_ids if not is_parameter(vocab.id2sym[pred])] for (order, pred_order_ids) in predicate_ids.items()}

    known_facts = kb_ids2known_facts(kb_ids)

    fact_structs = [rule for rule in kb_ids if len(rule) == 1]

    emb_np = emb.numpy()

    # We're only doing this experiment with one rule template and relationship
    rule_struct = [rule for rule in kb_ids if len(rule) > 1][0]
    relationship = [(head, body) for head, body in relationships.items()][0]
    head_id = [vocab.sym2id[relationship[0]]]
    body_ids = [vocab.sym2id[body_pred] for body_pred in relationship[1]]

    relationship_ids = head_id + body_ids
    rule_ids = kb_ids[rule_struct]
    rule_size = len(relationship_ids)
    n_rules = len(rule_ids[0][0])

    rule_pred_ids = [[rule_ids[atom][0][rule_nr] for atom in range(rule_size)] for rule_nr in range(n_rules)]
    
    relationship_embs = [np.expand_dims(emb_np[relationship_id], axis=0) for relationship_id in relationship_ids]
    rule_embs = [[np.expand_dims(emb_np[pred_id], axis=0) for pred_id in rule] for rule in rule_pred_ids]

    best_rule = -1
    max_rule_score = 0

    for rule_nr in range(n_rules):
        rule_score = np.min([l2_sim_np(relationship_embs[i], rule_embs[rule_nr][i]) for i in range(rule_size)])
        if rule_score > max_rule_score:
            best_rule = rule_nr
            max_rule_score = rule_score
        
    for atom in range(rule_size):
        emb_dif = rule_embs[best_rule][atom] - relationship_embs[atom]
        emb_np[rule_pred_ids[best_rule][atom]] += (r - 1.0) * tf.squeeze(emb_dif)

    emb.assign(emb_np)

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
            loss_parameters["epoch"] = i

            with tf.device("/device:GPU:0"):
                target = tf.constant(batch["target"], dtype=tf.float32)
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
        kb_ids, true_predicate_ids, constant_ids, emb, vocab, verbose=True, print_k=20)

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
    return rules, confidences, eval_dict



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-conf_path', default="conf_synth/rexp.conf")
    args = parser.parse_args()

    path = args.conf_path

    tf.enable_eager_execution()


    conf = load_conf(path)


    conf["experiment"]["test"] = True
    conf["experiment"]["n_test"] = 5
    conf["experiment"]["test_active_only"] = True


    conf["experiment"]["n_pred"] = 5
    conf["experiment"]["n_constants"] = 200
    conf["experiment"]["n_rel"] = 1
    conf["experiment"]["n_body"] = 2
    conf["experiment"]["n_rules"] = 3
    conf["experiment"]["p_normal"] = 0.5
    conf["experiment"]["p_relationship"] = 1.0
    conf["experiment"]["order"] = 1

    r = 0.0

    conf["training"]["num_epochs"] = 50
    conf["training"]["learning_rate"] = 0.001
    conf["training"]["lr_decay_type"] = "exp"
    conf["training"]["lr_decay_rate"] = 0.0003
    
    conf["experiment"]["gradient_experiment"] = False
    conf["model"]["k_max"] = 10
    # conf["model"]["loss_type"] = "top_k_all_anneal"
    # conf["model"]["loss_type"] = "top_k_all_type"
    conf["model"]["loss_type"] = "top_k"

    conf["model"]["loss_parameters"] = {"k": 1}
    # conf["model"]["loss_parameters"] = {"k_values": [2, 1, 1], "regime_thresholds": [0, 50, 75], "regime_types": ["top_k_all_type", "top_k_all_type", "top_k"]}
    # conf["model"]["loss_parameters"] = {"k_values": [2, 1], "regime_thresholds": [0, 50], "regime_types": ["top_k_all_type", "top_k_all_type"]}

    curr_path = os.path.dirname(os.path.abspath(__file__))
    base_data_path = curr_path + "/../../data/synthetic/synth1"

    # test settings
    # conf["training"]["pos_per_batch"] = 1
    # conf["model"]["input_size"] = 1
    # conf["training"]["neg_per_pos"] = 1
    # # conf["experiment"]["n_constants"] = 20
    conf["training"]["num_epochs"] = 1

    n_pred = conf["experiment"]["n_pred"]
    n_constants = conf["experiment"]["n_constants"]
    n_rel = conf["experiment"]["n_rel"]
    body_predicates = conf["experiment"]["n_body"]
    order = conf["experiment"]["order"]
    n_rules = conf["experiment"]["n_rules"]
    p_normal = conf["experiment"]["p_normal"]
    p_relationship = conf["experiment"]["p_relationship"]
    base_seed = conf["experiment"]["base_seed"]

    random.seed(base_seed)
    np.random.seed(base_seed)

    n_runs = conf["experiment"]["n_runs"]
    n_runs = 2

    base_dir = conf["logging"]["log_dir"] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    summary_writer = tf.contrib.summary.create_file_writer(base_dir + conf["experiment"]["name"])

    eval_history = OrderedDict()
    eval_keys = ["prop_rules", "active_facts"]

    if conf["experiment"]["test"] == True:
        eval_keys.extend(["MRR", "randomMRR", "fact-roc-auc"])

    for key in eval_keys:
        eval_history[key] = list()
    auc_helper_list = list()
    
    for i in range(n_runs):
        print("Run: " + str(i))

        conf["logging"]["log_dir"] = base_dir + "run" + str(i) + "/"
        conf["training"]["seed"] = np.random.randint(100)

        relationships = gen_relationships(n_pred, n_rel, body_predicates=body_predicates)
        symbol_relationships = relationship_id_to_symbol(relationships)

        train_data = gen_simple(n_pred, relationships, p_normal, p_relationship, n_constants, order=order)

        train_list = write_data(train_data)
        rules_list = write_simple_templates(n_rules, body_predicates=body_predicates, order=order)

        if conf["experiment"]["test"] == True:
            test_kb, train_list = gen_test_kb(train_list, conf["experiment"]["n_test"], conf["experiment"]["test_active_only"], relationships)
        else:
            test_kb = None            

        kb = load_from_list(train_list)
        templates = load_from_list(rules_list, rule_template=True)

        rules, confidences, eval_dict = train_model(kb, templates, conf, relationships=symbol_relationships, test_kb=test_kb, r=r)
        
        print(relationships)

        constant_dict = gen_constant_dict(train_list)
        eval_dict["active_facts"] = count_active(constant_dict, relationships)
        eval_dict["active_ratio"] = eval_dict["active_facts"] / len(train_list)        

        for key, value in eval_dict.items():
            if key in eval_history:
                print(key, value)
                eval_history[key].append(value)
        
        auc_helper_list.append(eval_dict["auc_helper"])
        print(eval_dict["auc_helper"])
        
    print(conf["model"], conf["experiment"])
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for key, value in eval_history.items():
            tf.contrib.summary.scalar(key, np.mean(eval_history[key]), step=0)        
            print("average " + key + ":", str(np.mean(value)) + " (" + str(np.std(value)/np.sqrt(n_runs)) + ")")

    targets, scores = [], []
    for run_tuple in auc_helper_list:
        targets.extend(run_tuple[0])
        scores.extend(run_tuple[1])
    
    if all(elem == targets[0] for elem in targets):
        pr_auc = 1.0
    
    else:
        pr_auc = metrics.average_precision_score(targets, scores)

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():        
        tf.contrib.summary.scalar("rule-pr-auc", pr_auc, step=0)        
    print("rule-pr-auc: " + str(pr_auc))
