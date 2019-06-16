import argparse
from ntp.scripts.learn import train_model
from ntp.modules.eval_functions import prop_rules, weighted_prop_rules, weighted_precision
from ntp.modules.generate import gen_unary_relationships, write_data, write_relationships, write_unary_templates, gen_unary
from ntp.util.util_data import load_conf
import os
import csv
import numpy as np
from collections import defaultdict
import copy


def run_experiment(conf):

    n_draws = conf["experiment"]["n_draws"]
    runs_per_draw = conf["experiment"]["runs_per_draw"]
    n_pred = conf["experiment"]["n_pred"]
    n_constants = conf["experiment"]["n_constants"]
    test_constants = conf["experiment"]["test_constants"]
    n_rel = conf["experiment"]["n_rel"]
    n_rules = conf["experiment"]["n_rules"]
    p_normal = conf["experiment"]["p_normal"]
    p_relationship = conf["experiment"]["p_relationship"]
    allow_reverse = conf["experiment"]["allow_reverse"]

    prop_rules_list = []
    weighted_prop_list = []
    weighted_precision_list = []

    for draw in range(n_draws):

        draw_prop_rules = []
        draw_weighted_prop = []
        draw_weighted_precision = []

        relationships = gen_unary_relationships(n_pred, n_rel)
        train_data = gen_unary(n_pred, relationships, p_normal, p_relationship, n_constants)
        test_data = gen_unary(n_pred, relationships, p_normal, p_relationship, test_constants)

        write_data(train_data, base_data_path + "_train.nl")
        write_data(test_data, base_data_path + "_test.nl")
        write_relationships(relationships, base_data_path + "_relationships.json")
        write_unary_templates(n_rules, base_data_path + "_templates.nlt")


        for run in range(runs_per_draw):
            print("Starting run {0} in draw {1}".format(run + 1, draw + 1))
            print("\n")
            rules, confidences = train_model(conf, rules_experiment=True)

            draw_prop_rules.append(prop_rules(relationships, rules, allow_reverse=allow_reverse))
            draw_weighted_prop.append(weighted_prop_rules(relationships, rules, confidences, allow_reverse=allow_reverse))
            draw_weighted_precision.append(weighted_precision(relationships, rules, confidences, allow_reverse=allow_reverse))

        prop_rules_list.append(draw_prop_rules)
        weighted_prop_list.append(draw_weighted_prop)
        weighted_precision_list.append(draw_weighted_precision)

        print("\n")
        print("Proportion rules learned: {}".format(draw_prop_rules))
        print("Weighted by confidence proportion rules learned: {}".format(draw_weighted_prop))
        print("Weighted precision: {}".format(draw_weighted_precision))
    

    avg_prop_rules = np.mean(prop_rules_list)
    avg_weighted_prop = np.mean(weighted_prop_list)
    avg_weighted_precision = np.mean(weighted_precision_list)

    weighted_prop_std = np.std(weighted_prop_list)
    weighted_prop_std_within = np.sqrt(np.mean([np.var(weighted_prop_list[i]) for i in range(len(weighted_prop_list))]))

    columns = ["experiment_category", "ruleprop", "wruleprop", "wprecision", "wruleprop_std", "wruleprop_std_within", 
    "predicates", "constants", "relationships", "p_default", "p_rel", 
    "rules", "epochs", "units", "batch_size", "corruption_ratio", "lr", "draws", "runs", "allow_reverse", "custom1", "custom2", "custom3"]


    value_dict = defaultdict(str, {"experiment_category": experiment_category, 
    "ruleprop": avg_prop_rules, 
    "wruleprop": avg_weighted_prop,
    "wprecision": avg_weighted_precision, 
    "wruleprop_std": weighted_prop_std, 
    "wruleprop_std_within": weighted_prop_std_within,
    "predicates": n_pred,
    "constants": n_constants, 
    "relationships": n_rel,
    "p_default": p_normal,
    "p_rel": p_relationship,
    "rules": n_rules,
    "epochs": conf["training"]["num_epochs"],
    "units": conf["model"]["input_size"], 
    "batch_size": conf["training"]["pos_per_batch"]* conf["training"]["num_corruptions"], 
    "corruption_ratio": conf["training"]["num_corruptions"], 
    "lr": conf["training"]["learning_rate"], 
    "draws": n_draws, 
    "runs": runs_per_draw,
    "allow_reverse": allow_reverse
    }
    )

    with open(output_path, "a+") as output_file:
        
        result_writer = csv.writer(output_file, delimiter=',')

        if os.stat(output_path).st_size == 0:
            result_writer.writerow(columns)
        
        result_writer.writerow([value_dict[column] for column in columns])
        

    print("\n")
    print("Final proportion rules learned: {}".format(avg_prop_rules))
    print("Final weighted by confidence proportion rules learned: {}".format(avg_weighted_prop))
    print("Final weighted precision: {}".format(avg_weighted_precision))
    print("Std weighted proportion rules learned: {}".format(weighted_prop_std))
    print("Std within data draw weighted proportion rules learned: {}".format(weighted_prop_std_within))

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-conf_path', default="conf_synth/synth1.conf")
    args = parser.parse_args()


    curr_path = os.path.dirname(os.path.abspath(__file__))

    base_config_path =  curr_path + "/../../"
    base_data_path = curr_path + "/../../data/synthetic/synth1"
    base_output_path = curr_path + "/../../out/rule_exp/"

    experiment_category = "synth1"
    output_path = base_output_path + experiment_category + ".csv"
    
    config_path = base_config_path + args.conf_path 
    conf = load_conf(config_path)

    conf["experiment"] = {}
    conf["experiment"]["n_draws"] = 10
    conf["experiment"]["runs_per_draw"] = 10
    conf["experiment"]["n_pred"] = 3
    conf["experiment"]["n_constants"] = 100
    conf["experiment"]["test_constants"] = 20
    conf["experiment"]["n_rel"] = 1
    conf["experiment"]["n_rules"] = 3
    conf["experiment"]["p_normal"] = 0.3
    conf["experiment"]["p_relationship"] = 1
    conf["experiment"]["allow_reverse"] = True

    test = False


    if test:
        conf["training"]["num_epochs"] = 1
        conf["experiment"]["n_draws"] = 1
        conf["experiment"]["runs_per_draw"] = 1


    # conf["ranges"] = {"pos_per_batch": [50]}
    # conf["ranges"] = {"n_rules": [10, 20, 30, 40, 50]}
    # conf["ranges"] = {"n_pred": [20, 30, 40, 50]}
    # conf["ranges"] = {"n_constants": [25, 50, 75, 100, 200, 300, 400, 500]}
    conf["ranges"] = {"runs_per_draw": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
    




    rseed = 12
    # rd.seed(rseed)    
    
    # exp vars
    experiment_ranges = {}
    for var_name in conf["ranges"]:
        experiment_ranges[var_name] = conf["ranges"][var_name]
    
    for var_name in experiment_ranges:
        for var_value in experiment_ranges[var_name]:
            conf_curr = copy.deepcopy(conf)
            if var_name in conf["training"]:
                conf_curr["training"][var_name] = var_value
            elif var_name in conf["model"]:
                conf_curr["model"][var_name] = var_value
            elif var_name in conf["experiment"]:
                conf_curr["experiment"][var_name] = var_value

            run_experiment(conf_curr)
            pass


