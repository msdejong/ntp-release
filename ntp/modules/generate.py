"""Functions for generating random data with injected relationships"""

from itertools import product
import os
import json
import re
import random

import numpy as np
from numpy import random as rd
from scipy.special import comb

from ntp.util.util_kb import load_from_list



def gen_relationships(n_pred, n_rel, body_predicates=1):
    """
    Generates random relationships between predicates of the form goal predicate <-- {set of body predicates}. 
    Goal predicates have a higher number than body predicates. 
    
    Args:
        n_pred: number of total predicates
        n_rel: number of relationships
        body_predicates: number of body predicates for each relationship
    Returns:
        Dict, entries where keys are goal predicates and values are list of body predicates
    """

    relationship_dict = {}
    n_rel_possible = comb(n_pred, body_predicates + 1)
    pred_probs = [comb(i, body_predicates)/n_rel_possible for i in range(n_pred)]
    relationship_head_array = list(rd.choice(n_pred, size=n_rel, replace=False, p=pred_probs))
    relationship_body_array = [set(rd.choice(range(relationship_head_array[i]), size=body_predicates, replace=False)) for i in range(len(relationship_head_array))]

    for i in range(n_rel):
        relationship_dict[relationship_head_array[i]] = relationship_body_array[i]

    return relationship_dict


def gen_simple(n_pred, relationship_dict, p_normal, p_relationship, n_constants, order=1):
    """
    Generates random truth values for predicates for a set number of constants, and given some relationships
    Args:
        n_pred: number of total predicates
        relationship_dict: Dict of relationships
        p_normal: probability of predicate truth given no relationship/relationship body not true
        p_relationship: probability of goal predicate truth given body predicate truth
        n_constants: number of constants
        order: order of predicate (unary, binary)
    Returns:
        Numpy array where value j, i corresponds to the truth value of predicate i for constant j
    """

    # Checks whether body predicates for a particular relationship hold for a particular constant
    def body_holds(data, body_predicates, constant):
        holds = True
        for predicate in body_predicates:
            if data[index + (predicate,)] != 1:
                holds = False
                break
        return holds

    data = np.zeros([n_constants] * order + [n_pred])

    
    for predicate in range(n_pred):
        for index in product(*[range(n_constants) for i in range(order)]):
            if predicate in relationship_dict:
                if body_holds(data, relationship_dict[predicate], index): 
                    data[index + (predicate,)] = rd.binomial(1, p_relationship)
                    continue

            # Set variable normally if predicate from relationship doesn't hold
            data[index + (predicate,)] = rd.binomial(1, p_normal)            
    return data   

def write_data(data):
    """Convert numpy array of data into list of strings that the ntp algorithm can read"""
    shape = np.shape(data)
    text_list = []
    for pred in range(shape[-1]):
        for index in product(*[range(dim_size) for dim_size in shape[:-1]]):
            if data[index + (pred,)] == 1:
                write_string = "Predicate" + str(pred) + "("
                for const in index:
                    write_string += "Constant" + str(const) + ","
                write_string = write_string[:-1] + ").\n"
                text_list.append(write_string)
    return text_list
    

def write_relationships(relationships, path):
    """write relationship dict to file"""

    with open(path, "w") as f:
        json.dump(relationships, f)
    return


def write_simple_templates(n_rules, body_predicates=1, order=1):
    """Generate rule template of form C < A ^ B of varying size and order"""
    text_list = []

    const_term = "("
    for i in range(order):
        const_term += chr(ord('X') + i) + ","
    const_term = const_term[:-1] + ")"

    write_string = "{0} #1{1} :- #2{1}".format(n_rules, const_term) 
    if body_predicates > 1:
        for i in range(body_predicates - 1):
            write_string += ", #" + str(i + 3) + const_term
    
    text_list.append(write_string)
    return text_list



def gen_transitivity(n_preds, n_rules, n_constants, p_base, max_iterations=1):
    """Generate data with transitivity relationships, and also rule templates"""
    # active predicate is predicate 0 WLOG
    active_values = np.random.binomial(1, p_base, size=[n_constants, n_constants])

    edges = [(i, j) for i in range(n_constants) for j in range(n_constants) if active_values[i, j] == 1]

    closure = set(edges)
    
    while True:
        new_edges = set((x,w) for x,y in closure for q,w in closure if q == y)

        closure_until_now = closure | new_edges

        if closure_until_now == closure:
            break

        closure = closure_until_now

    edges = list(closure)

    active_values[tuple(np.transpose(edges))] = 1

    values = np.random.binomial(1, p_base, size=[n_constants, n_constants, n_preds])
    values[:, :, 0] = active_values

    fact_list = write_data(values)

    template = "{0} #1(X, Z) :- #1(X, Y), #1(Y, Z).".format(n_rules) 

    return fact_list, template



def text_to_id(fact):
    """Given a fact in text form, convert to predicate and constant numbers"""
    reduced = re.sub("[^0-9\(,]", '', fact)
    reduced_split = tuple(re.split("[\(,]", reduced))
    predicate = int(reduced_split[0])
    constants = tuple([int(constant_text) for constant_text in reduced_split[1:]])
    return predicate, constants

def gen_constant_dict(train_list):
    """Convert list of facts in text form to a dictionary of predicate truth values by constant"""
    constant_dict = {}
    for fact in train_list:
        predicate, constants = text_to_id(fact)
        if not constants in constant_dict:
            constant_dict[constants] = set([predicate])
        else: 
            constant_dict[constants].add(predicate)
    return constant_dict

def test_fact_active(constant_dict, constants, predicate, relationships):
    """Given relationships, determine whether the truth value of a fact could be predicted by a relationship"""
    if predicate in relationships:
        if all(body_pred in constant_dict[constants] for body_pred in relationships[predicate]):
            return True
    return False

def count_active(constant_dict, relationships):
    """Given relationships and a dataset of constants, determine for how many facts the truth value could be predicted by a relationship"""
    active_facts = 0
    for constants, predicates in constant_dict.items():
        for predicate in relationships:
            if predicate in predicates and all(body_pred in predicates for body_pred in relationships[predicate]):
                active_facts += 1
    return active_facts

def gen_test_kb(train_list, n_test, test_active_only=False, relationships=None):
    """Given a list of facts, choose some facts to be split off to a test dataset in such a way that there is at least one training fact left for each constant"""
    constant_dict = gen_constant_dict(train_list)
    random.shuffle(train_list)
    constant_set = set()
    new_train_list = []
    test_list = []
    for fact in train_list:
        predicate, constants = text_to_id(fact)

        if test_active_only:
            if test_fact_active(constant_dict, constants, predicate, relationships) and len(test_list) < n_test:
                test_list.append(fact)
                continue
        else:
            if all(constant in constant_set for constant in constants) and len(test_list) < n_test:
                test_list.append(fact)
                continue
            else:
                for constant in constants:
                    constant_set.add(constant)
        new_train_list.append(fact)

    train_list = new_train_list
    test_kb = load_from_list(test_list)
    return test_kb, train_list

