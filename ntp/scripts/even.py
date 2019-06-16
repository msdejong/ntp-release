from ntp.modules.learn import train_model
from ntp.modules.eval import prop_rules, weighted_prop_rules, weighted_precision, confidence_accuracy
from ntp.modules.generate import gen_simple, gen_relationships, write_data, write_relationships, write_simple_templates, gen_test_kb, gen_constant_dict, count_active

from ntp.util.util_data import load_conf
from ntp.util.util_kb import load_from_list, load_from_file

import os
import argparse
import numpy as np
import random
import re
from sklearn import metrics
from collections import OrderedDict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-conf_path', default="conf_synth/even.conf")
    args = parser.parse_args()

    path = args.conf_path

    conf = load_conf(path)


    conf["experiment"]["test"] = False
    conf["experiment"]["n_test"] = 5
    conf["experiment"]["test_active_only"] = True


    conf["experiment"]["n_constants"] = 200
    conf["experiment"]["n_rules"] = 3

    conf["training"]["num_epochs"] = 50
    conf["training"]["learning_rate"] = 0.001
    conf["training"]["lr_decay_type"] = "exp"
    conf["training"]["lr_decay_rate"] = 0.0005
    
    conf["experiment"]["gradient_experiment"] = False
    conf["model"]["k_max"] = 10
    # conf["model"]["loss_type"] = "top_k_all_anneal"
    conf["model"]["loss_type"] = "top_k_all_type"
    #conf["model"]["loss_type"] = "top_k"

    conf["model"]["loss_parameters"] = {"k": 2}
    # conf["model"]["loss_parameters"] = {"k_values": [2, 1, 1], "regime_thresholds": [0, 50, 75], "regime_types": ["top_k_all_type", "top_k_all_type", "top_k"]}
    # conf["model"]["loss_parameters"] = {"k_values": [2, 1], "regime_thresholds": [0, 50], "regime_types": ["top_k_all_type", "top_k_all_type"]}
    # conf["model"]["loss_parameters"] = {"k_values": [-1, -1], "regime_thresholds": [0, 50], "regime_types": ["all_type", "standard"]}

    curr_path = os.path.dirname(os.path.abspath(__file__))
    base_data_path = curr_path + "/../../data/synthetic/synth1"

    # test settings
    # conf["training"]["pos_per_batch"] = 1
    # conf["model"]["input_size"] = 1
    # conf["training"]["neg_per_pos"] = 1
    # # conf["experiment"]["n_constants"] = 20
    # conf["training"]["num_epochs"] = 1

    n_pred = conf["experiment"]["n_pred"]
    n_constants = conf["experiment"]["n_constants"]
    n_rel = conf["experiment"]["n_rel"]
    body_predicates = conf["experiment"]["n_body"]
    order = conf["experiment"]["order"]
    n_rules = conf["experiment"]["n_rules"]
    p_normal = conf["experiment"]["p_normal"]
    p_relationship = conf["experiment"]["p_relationship"]

    rseed = 0
    random.seed(rseed)
    np.random.seed(rseed)
    conf["training"]["seed"] = rseed

    n_runs = 50

    eval_history = OrderedDict()
    eval_keys = ["prop_rules", "active_facts"]

    if conf["experiment"]["test"] == True:
        eval_keys.extend(["MRR", "randomMRR", "fact-pr-auc", "fact-roc-auc"])

    for key in eval_keys:
        eval_history[key] = list()
    auc_helper_list = list()
    
    for i in range(n_runs):
        print("Run: " + str(i))

        relationships = gen_relationships(n_pred, n_rel, body_predicates=body_predicates)
        train_data = gen_simple(n_pred, relationships, p_normal, p_relationship, n_constants, order=order)

        train_list = write_data(train_data)
        rules_list = write_simple_templates(n_rules, body_predicates=body_predicates, order=order)

        if conf["experiment"]["test"] == True:
            test_kb, train_list = gen_test_kb(train_list, conf["experiment"]["n_test"], conf["experiment"]["test_active_only"], relationships)
        else:
            test_kb = None            

        kb = load_from_list(train_list)
        templates = load_from_list(rules_list, rule_template=True)

        rules, confidences, eval_dict = train_model(kb, templates, conf, diagnostic=True, relationships=relationships, test_kb=test_kb)
        
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

    for key, value in eval_history.items():
        print("average " + key + ":", str(np.mean(value)) + " (" + str(np.std(value)/np.sqrt(n_runs)) + ")")

    targets, scores = [], []
    for run_tuple in auc_helper_list:
        targets.extend(run_tuple[0])
        scores.extend(run_tuple[1])
    
    if all(elem == targets[0] for elem in targets):
        roc_auc = 1.0
        pr_auc = 1.0
        
    else:
        roc_auc = metrics.roc_auc_score(targets, scores)
        pr_auc = metrics.average_precision_score(targets, scores)
    print("rule-pr-auc: " + str(pr_auc))

    

    

    
