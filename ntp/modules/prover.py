"""This module contains the prover, which, given a kb, goal, and embedding tensor, calculates the score for each proof

The prover works like this. 

1) The goal, kb and embedding tensor are given to or_, along with an empty substitution dict and depth = 0
2) or_ selects a structs and attempts to unify the goal with the head of that struct
3) unify performs any necessary variable substitutions and calls batch_unify to calculate unification scores of
   head and goal
4) With this unification complete, or_  checks if the goal is proved (has been unified with a fact in
   the kb) or we have reached the maximal depth of the proof tree. If so, it adds the proof to a list and moves to
   step 2) for the next struct. If not, or_ calls and_, which generates new goals from the body of the previous
   struct. and_ then calls or_ recursively. or_ is given a dictionary containing substitutions and unification
   scores up to that point, as well as a depth of 1 to reflect that we are operating at a deeper level of a proof tree
5) Repeat steps 2 through 4 until every allowed proof path has been scored
"""


import copy
import collections
from pprint import pprint
import os

import numpy as np
import tensorflow as tf

from ntp.util.util_kb import is_variable
from ntp.modules.nunify import representation_match


def is_variables_list(xs):
    """check whether all elements in a list are variables"""
    if isinstance(xs, list):
        return all([is_variable(x) for x in xs])
    else:
        return False

def consists_variables(value):
    """check whether all elements in input are variables"""
    if isinstance(value, list):
        return all([is_variable(x) for x in value])
    else:
        return is_variable(value)

def is_tensor(arg):
    return isinstance(arg, tf.Tensor)

def detect_cycle(variable, substitutions):
    """check whether variable has been substituted before"""
    # cycle detection
    if not isinstance(variable, list) and variable in substitutions:
        return True
    elif tuple(variable) in substitutions:
        return True
    else:
        has_cycle = False
        for key in substitutions:
            if isinstance(key, list) and variable in key:
                has_cycle = True
        return has_cycle

def rule_struct_form(rule, struct):
    """convert from 'kb form' to 'rule form'
    kb form is [[symbol tensor for position in atom] for atom in rule]
    rule form is [[dict of struct/atom_nr/term_nr for position in atom] for atom in rule] 
    The motivation for doing this is that you can keep track of what groups of logical terms are being unified, not
    just the tensors
    Args: 
        rule: list of atoms
        struct: name of rule
    Returns: 
        rule in rule form
    """

    rule_value = []
    for atom_nr, atom in enumerate(rule):
        atom_value = []
        for term_nr, _ in enumerate(atom):
            term_value = {'struct': struct, 'atom': atom_nr, 'symbol': term_nr}
            atom_value.append(term_value)
        rule_value.append(atom_value)
    return rule_value

def kb_form(struct_form, kb):
    """Convert from rule form back to kb form when you need the tensor"""

    def kb_form_single(term, kb):
        return kb[term['struct']][term['atom']][term['symbol']] 

    if isinstance(struct_form, list):
        return [kb_form_single(term, kb) for term in struct_form]
    else:
        return kb_form_single(struct_form, kb)
        
def unify_variables(variables, values, substitutions, depth=0, goals_var=False):
    """Called when variables are unified with values. Stores variable value in substitution dict
    Args:
        variables: variables being unified
        values: values variables are unified with
        substitutions: dict of substitutions
        depth: current depth of proof
        goals_var: true if the goal is the variable, as opposed to the body
    """

    # store at what point in the proof the variable was substituted in order to calculate unification score
    # correctly if the same variable comes up at other stages of the proof
    success_subposition = len(substitutions['HISTORY']) - 1 
    if goals_var:
        success_subposition += 1

    if detect_cycle(variables, substitutions):
        return 'FAILURE'
    else:
        if isinstance(variables, list):
            for i, value in enumerate(values):
                value = copy.deepcopy(value)
                value['sub_position'] = success_subposition
                substitutions['VARSUBS'][variables[i]] = value
        else:
            values = copy.deepcopy(values)
            values['sub_position'] = success_subposition
            substitutions['VARSUBS'][variables] = values
        return substitutions

def find_goal_position(history, depth):
    """When batch unifying the embeddings of a goal symbol and a body symbol, the goal symbol is part of some logical atom.  
    This function finds the position of that atom in the current proof. 
    Works by finding the first position in the proof history where the depth is equal to current depth."""
    if depth == 0:
        return 0
    else:
        for i in range(len(history)):
            if history[i][1] == depth:
                return i
    
def batch_unify(rhs, goals, substitutions, kb, depth=0, mask=None, transpose=False, inner_tiling=True):
    """Given goal and rhs unification, calculate unification score and update the proof state with that score"""

    # If the goal contains a value that results from a variable substitution, the goal will contain information on
    # at what stage of the proof the substitution occurred. Want to update the unification at that position. 
    sub_position = -1
    if 'sub_position' in goals:
        sub_position = goals['sub_position']
    
    rhs_tensor = kb_form(rhs, kb) # Transform lists of symbol ids to tensors
    goal_tensor = kb_form(goals, kb)

    current_success = representation_match(goal_tensor, rhs_tensor)
    current_unifications = [goals, rhs]
    
    if 'SUCCESS' in substitutions:

        old_success = substitutions['SUCCESS']
        old_shape = old_success.get_shape()

        # If unification is moving to the next atom, as opposed to another symbol in the same atom, add a dimension
        # to unification score tensor
        if len(old_shape) - 1 < len(substitutions['HISTORY']):
            old_success = tf.expand_dims(old_success, axis=-1)
            old_shape = old_success.get_shape()
                
        current_position = len(old_success.get_shape()) - 1

        if sub_position != -1:
            # If current goal was result of substitution, expand dimensions to match position of goal in current success with appropriate position in old_success
            # Necessary because we want to update the unification scores at the correct proof tree paths             
            pos_difference = len(old_shape) - sub_position - 2
            for i in range(pos_difference):
                current_success = tf.expand_dims(current_success, axis=1)
            
            current_unifications.append([sub_position, len(old_success.get_shape()) - 1])
        else:
            # Find position of goal, and expand dimensions to match position of goal in current_success with appropriate position in old_success
            # Necessary in case there are multiple body atoms at the same depth, and we have to 'skip' a previous body atom
            goal_position = find_goal_position(substitutions['HISTORY'], depth)
            for i in range(current_position - goal_position - 1):
                current_success = tf.expand_dims(current_success, axis=1)
            current_unifications.append([goal_position, current_position])

        # If we are only using k_max proofs, we retrieve that subset from all current succes scores
        if 'subset' in goals:
            subset = goals['subset']
            current_success = tf.gather(current_success, subset)

        current_success = tf.minimum(old_success, current_success)

        # Expand dimensions so that the mask broadcasts with the new success tensor
        if mask is not None:
            mask_shape = mask.get_shape()
            dim_dif = len(current_success.get_shape()) - len(mask_shape) 
            for i in range(dim_dif):
                # Expand in the middle because goal dimension is rank 0 and the last dimension is the fact dimension
                mask = tf.expand_dims(mask, axis=1)

    if mask is not None:
        current_success = current_success * mask

    substitutions['SUCCESS'] = current_success

    if 'UNIFICATIONS' in substitutions:
        substitutions['UNIFICATIONS'].append(current_unifications)
    else:
        current_unifications.append([0, 1])
        substitutions['UNIFICATIONS'] = [current_unifications]
    return substitutions


def unify(rhs, goals, substitutions, kb, depth=0, mask_id=None, transpose=False, inner_tiling=True):
    """Unifies goal with the head of a struct, updating the substitutions dict with any variable substitutions 
    and updating the unification score tensor with the current unification"""


    if len(goals) == 1:
        goals = goals[0]
        rhs = rhs[0]

    if goals == rhs:
        return substitutions

    goals_kb = kb_form(goals, kb)
    rhs_kb = kb_form(rhs, kb)

    substitutions_copy = copy.copy(substitutions)
    if substitutions_copy == 'FAILURE':
        return substitutions_copy
    elif consists_variables(rhs_kb):
        return unify_variables(rhs_kb, goals, substitutions_copy, depth)
    elif consists_variables(goals_kb):
        return unify_variables(goals_kb, rhs, substitutions_copy, depth, goals_var=True)
    elif isinstance(rhs_kb, list) and isinstance(goals_kb, list) \
            and len(rhs_kb) == len(goals_kb):
        return unify(rhs[0], goals[0],
                     unify(rhs[1:], goals[1:], substitutions_copy, kb, depth,
                           mask_id, transpose, inner_tiling), kb,
                     depth, mask_id, transpose, inner_tiling)
    elif is_tensor(goals_kb):
        return batch_unify(rhs, goals, substitutions, kb, depth, mask_id, transpose, inner_tiling)


def substitute(goals, substitutions, kb):
    """Check if goal contains any variables for which we have already have substitutions, and if so, replace the variables with corresponding values."""

    goals_kb = kb_form(goals, kb)

    new_goal = []
    for arg_nr, arg in enumerate(goals_kb):
        new_arg = goals[arg_nr]
        if consists_variables(arg):
            if isinstance(arg, list):
                var = arg[0]
            else:
                var = arg
            if var in substitutions['VARSUBS']:
                new_arg = substitutions['VARSUBS'][var] 

        new_goal.append(new_arg)

    return new_goal

def flatten_proofs(proofs):
    def flatten(xs):
        for x in xs:
            if isinstance(x, collections.Iterable) \
                    and not isinstance(x, str) \
                    and not isinstance(x, dict):
                for sub in flatten(x):
                    yield sub
            else:
                yield x

    return list(flatten(proofs))


def applied_before(rule, substitutions, kb):
    """Check if a rule has already been applied earlier in the proof tree"""
    head = kb_form(rule[0], kb)
    head_vars = [x for x in head if is_variable(x)]

    return any([x for x in head_vars if x in substitutions['VARSUBS']])


def or_(kb, goals, substitutions=dict(), depth=0, mask=None,
        k_max=None, max_depth=1):
    """Base function of prover, called recursively. 
    Calls and_, which in turn calls or_, in order to recursively calculate scores for every possible proof in proof
    tree. 
    Args:
        kb: dict of facts / rules
        goals: goal to be proved
        substitutions: dict which contains current variable substitutions and scores of current proof path
        depth: current proof depth
        mask: mask to apply so that goal facts (which are drawn from kb) cannot be proved by unifying with themselves
        k_max: number of fact unifications to retain from unifications with all facts in kb
        max_depth: maximum allowed proof depth before termination
    Returns:
        List of proof paths of goal with corresponding scores
    """
    proofs = []

    # initialize history and substitutions as empty
    if substitutions == {}:
        substitutions['VARSUBS'] = {}
        substitutions['HISTORY'] = []
        

    for struct in kb:
        # avoid fake added struct
        if struct == 'goal':
            continue

        # Check if struct order matches
        if len(struct[0]) != len(goals):
            continue
            
        rule = rule_struct_form(kb[struct], struct)
        head = substitute(rule[0], substitutions, kb) 
        body = rule[1:]
        mask_id = None
        if mask is not None:
            mask_key, mask_id = mask
            mask_id = mask_id if mask_key == struct else None

        is_fact = len(struct) == 1 and all([not is_variable(x)
                                            for x in struct[0]])

        if not is_fact and depth == max_depth:
            # maximum depth reached
            continue
        
        # rule has been applied before
        elif applied_before(rule, substitutions, kb):
            continue

        substitutions_copy = copy.deepcopy(substitutions)          
        substitutions_copy['HISTORY'].append([struct, depth])    
        substitutions_ = unify(head, goals, substitutions_copy, kb, depth, mask_id,
                                transpose=is_fact)

        
        if is_fact and k_max is not None:
            new_success, success_indices = tf.nn.top_k(substitutions_["SUCCESS"], k_max)
            substitutions_["SUCCESS"] = new_success
            for value in substitutions_['VARSUBS'].values():
                if value['struct'] != 'goal' and not 'subset' in value:
                    value['subset'] = success_indices

        if substitutions_ != 'FAILURE':
            proof = and_(kb, body, substitutions_, depth, mask, k_max=k_max, max_depth=max_depth)

            if not isinstance(proof, list):
                proof = [proof]
            else:
                proof = flatten_proofs(proof)

            for proof_substitutions in proof:
                if proof_substitutions != 'FAILURE':
                    proofs.append(proof_substitutions)
    return flatten_proofs(proofs)


def and_(kb, subgoals, substitutions, depth=0, mask=None, in_body=False, k_max=None,
         max_depth=1):
    """Generate new goals consisting of the body of the struct that previous goal was unified with and calls or_ to prove new goals"""
    if len(subgoals) == 0:
        return substitutions
    elif depth == max_depth:  # maximum depth
        return 'FAILURE'
    else:
        head = subgoals[0]
        body = subgoals[1:]

        proofs = []

        new_goal = substitute(head, substitutions, kb)

        new_body = body

        for substitutions_ in or_(kb, new_goal, substitutions, depth+1, mask,
                                  k_max=k_max, max_depth=max_depth):
            proofs.append(and_(kb, new_body, substitutions_, depth, mask, in_body=True, k_max=k_max,
                               max_depth=max_depth))
        return proofs


def prove(kb, goals, mask_structure, mask_var,
          k_max=None, max_depth=1, vocab=None):

    proofs = or_(kb, goals,
                 mask=(mask_structure, mask_var),
                 k_max=k_max, max_depth=max_depth)

    return proofs
