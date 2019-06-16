"""Helper functions for evaluating NTP model performance"""

import tensorflow as tf
import numpy as np

from ntp.modules.kb import Atom
from ntp.util.util_kb import rule2string, is_variable
from ntp.modules.prover import representation_match, is_tensor


def decode_rules(kb, true_predicate_ids, constant_ids, emb, vocab, verbose=False, print_k=3):
    """
    Takes in kb and trained embedding matrix, and returns decodings of the rules in the kb
    Args:
        kb: knowledge base of facts and rules
        true_predicate_ids: list of ids that correspond to 'real' (i.e. non-rule) predicates
        constant_ids: list of ids that correspond to constants
        emb: embedding matrix
        vocab: vocab object that contains mapping between symbol ids and symbols
    Returns: 
        List of rules and corresponding confidences
    """
    rules_preds = []
    confidences = []

    for struct in kb:
        # it's a rule
        if len(struct) > 1:
            rule = kb[struct]
            rule_sym = []
            for atom in rule:
                atom_sym = []
                for i, sym in enumerate(atom):
                    if not is_variable(sym[0]):
                        valid_ids = true_predicate_ids[len(atom) - 1] if i == 0 else constant_ids
                        atom_sym.append(decode(sym, emb, vocab, valid_ids))
                    else:
                        atom_sym.append(sym[0])
                rule_sym.append(Atom(atom_sym[0], atom_sym[1:]))

            rules = unstack_rules(rule_sym)

            rules.sort(key=lambda x: -x[1])
    
            for j in range(len(rules)):
                rule, confidence = rules[j]
                preds = [rule[i][0] for i in range(len(rule))]
                # pred_text = [rule[i][0] for i in range(len(rule))]
                # preds = [int(text[-1]) for text in pred_text]
                preds_formatted = [preds[0], set(preds[1:])]
                rules_preds.append(preds_formatted)

                confidences.append(confidence)
                if verbose and j < print_k:
                    print(confidence, rule2string(rule))
    return rules_preds, confidences

def decode(x, emb, vocab, valid_ids):
    """
    Takes in embedding and returns closest matching symbols in embedding matrix
    Args:
        x: array of symbol ids
        emb: embedding matrix
        vocab: vocab object that contains mapping between symbol ids and symbols
        valid_ids: the set of ids in the embedding matrix that the input may be matched against (i.e. predicates with predicates)
    Returns: 
        List of (symbol, success) tuples. Each symbol is selected from the embedding matrix as the closest symbol to the corresponding input symbol, 
        and the success is a decreasing function of that distance. 
    """
    valid_ids = set(valid_ids)

    num_rules = len(x)
    num_symbols = int(emb.get_shape()[0])

    mask = np.ones([num_symbols], dtype=np.float32)
    for i in range(len(vocab)):
        if i not in valid_ids: # or i == vocab.sym2id[vocab.unk]:
            mask[i] = 0  # np.zeros([input_size], dtype=np.float32)

    # -- num_rules x num_symbols
    mask = tf.tile(tf.expand_dims(mask, 0), [num_rules, 1])

    # Retrieve embedding
    x_tensor = tf.nn.embedding_lookup(emb, x)

    # -- num_rules x num_symbols
    match = representation_match(x_tensor, emb)
    success_masked = match * mask

    success_val, ix_val = tf.nn.top_k(success_masked, 1)
    # success_val, ix_val = sess.run([success, ix], {})

    syms = []
    for i, row in enumerate(ix_val):
        sym_id = row[0].numpy()
        sym_success = success_val[i][0].numpy()
        sym = vocab.id2sym[sym_id]

        syms.append((sym, sym_success))

    return syms

def unstack_rules(rule):
    """
    Takes in rule array, unstacks the rule array, 
    and calculates the confidence for each rule from the success of the underlying atom unifications with symbols in the embedding matrix
    Args:
        rule: stacked rule array
    Returns: 
        List of rules with corresponding confidence
    """
    rules = []
    num_rules = len(rule[0].predicate)
    for i in range(num_rules):
        current_rule = []
        confidence = 1.0
        for atom in rule:
            predicate = atom.predicate
            if isinstance(predicate, list):
                predicate, success = predicate[i]
                confidence = min(confidence, success)
            arguments = []
            for argument in atom.arguments:
                if isinstance(argument, list):
                    argument, success = argument[i]
                    arguments.append(argument)
                    confidence = min(confidence, success)
                else:
                    arguments.append(argument)
            current_rule.append(Atom(predicate, arguments))
        rules.append((current_rule, confidence))
    return rules


def harmonic(n_numbers):
    """Calculate harmonic series, used for calculating MRR"""
    return sum([1.0/(i + 1) for i in range(n_numbers)])
