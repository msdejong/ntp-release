"""Transform data for use in NTP model"""

from pprint import pprint
import copy
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from ntp.util.util_kb import Atom, is_variable, is_parameter
from ntp.util.util_data import Vocab


def rule2struct(rule):
    """
    Returns the structure of a rule used to partition a knowledge base
    Args:
        rule: rule 
    Returns: a tuple representing the structure of the rule
    """
    predicates = {}
    constants = {}
    variables = {}
    struct = []
    for predicate, args in rule:
        atom_struct = []
        if predicate not in predicates:
            predicates[predicate] = "p" + str(len(predicates))
        atom_struct.append(predicates[predicate])
        for arg in args:
            if is_variable(arg):
                if arg not in variables:
                    variables[arg] = "X" + str(len(variables))
                atom_struct.append(variables[arg])
            else:
                if arg not in constants:
                    constants[arg] = "c"
                atom_struct.append(constants[arg])
        struct.append(tuple(atom_struct))
    return tuple(struct)


def augment_with_templates(kb, rule_templates):
    """
    Adds templates to kb
    Args:
        kb: a knowledge base with symbolic representations
    Returns: knowledge base agumented with parameterized rule templates
    """
    kb_copy = copy.deepcopy(kb)

    def suffix_rule_parameters(rule, num_rule, num_copy):
        new_rule = []
        for predicate, args in rule:
            if is_parameter(predicate):
                new_rule.append(Atom("%s_%d_%d" %
                                     (predicate, num_rule, num_copy), args))
            else:
                new_rule.append(Atom(predicate, args))
        return new_rule

    for i, (rule_template, num) in enumerate(rule_templates):
        for j in range(num):
            kb_copy.append(suffix_rule_parameters(rule_template, i, j))
    return kb_copy


def partition(kb):
    """
    Creates kb dictionary with rule type as keys
    Args:
        kb: a knowledge base with symbolic representations
    Returns: dictionary kb with rule type as keys
    """    
    kb_partitioned = OrderedDict()
    for rule in kb:
        struct = rule2struct(rule)
        if struct not in kb_partitioned:
            kb_partitioned[struct] = [rule]
        else:
            kb_partitioned[struct].append(rule)
    return kb_partitioned


def kb2ids(kb):
    """Transforms non-variable symbols to ids in the kb"""
     
    kb_ids = OrderedDict()

    vocab = Vocab()

    predicate_ids = {}
    constant_ids = []

    for struct in kb:
        rules = kb[struct]
        kb_stacked = []

        for rule in rules:
            for i, (predicate, args) in enumerate(rule):
                if not len(args) in predicate_ids:
                    predicate_ids[len(args)] = []
                if len(kb_stacked) < i + 1:
                    kb_stacked.append([])
                symbols = [x for x in [predicate] + args] 
                for j, sym in enumerate(symbols):
                    if not is_variable(sym):
                        if j == 0 and sym not in vocab:
                            predicate_ids[len(args)].append(vocab(sym))
                        elif j > 0 and sym not in vocab:
                            constant_ids.append(vocab(sym))

                    if len(kb_stacked[i]) < j + 1:
                        kb_stacked[i].append([])
                    kb_stacked[i][j].append(sym)

        # mapping to ids and stacking as numpy array
        for i, atom in enumerate(kb_stacked):
            for j, symbols in enumerate(atom):
                if not is_variable(symbols[0]):
                    kb_stacked[i][j] = np.hstack(vocab(symbols))
                else:
                    kb_stacked[i][j] = symbols

        kb_ids[struct] = kb_stacked
    return kb_ids, vocab, predicate_ids, constant_ids

def kb_ids2known_facts(kb_ids):
    """Creates list of all known facts from kb dict"""

    facts = set()
    for struct in kb_ids:
        arrays = kb_ids[struct][0]
        num_facts = len(arrays[0])
        for i in range(num_facts):
            fact = [x[i] for x in arrays]
            facts.add(tuple(fact))
    return facts

def kb2nkb(kb_ids, emb):
    """
    Creates nkb dict matching structs to embeddings from embedding tensor"""

    nkb = OrderedDict()
    for ix, struct in enumerate(kb_ids):
        kb_stacked = kb_ids[struct]

        atoms_embedded = []
        for i, atom in enumerate(kb_stacked):
            atom_embedded = []
            for j in range(len(kb_stacked[i])):
                symbol = kb_stacked[i][j]
                if isinstance(symbol, np.ndarray):
                    atom_embedded.append(tf.nn.embedding_lookup(emb, symbol))
                else:
                    if isinstance(symbol, list):
                        atom_embedded.append("%s%d" % (symbol[0][0], ix))
                    # atom_embedded.append(symbol)
            atoms_embedded.append(atom_embedded)
        nkb[struct] = atoms_embedded

    return nkb    

def initialize_nkb(kb, input_size=10, init=(-1.0, 1.0)):
    """
    Given list of rules and facts in symbol form, create kb dict in id form, and instantiate embeddings for symbols
    Args: 
        kb: list of facts in text form
        input_size: symbol embedding size
        init: initialization range
    Returns:
        nkb: dict matching structs to relevant parts of embedding tensor
        kb_ids: dict matching structs to lists of symbol ids
        vocab: contains dicts matching symbols to text
        embedding_matrix: tensor of symbol embeddings
        predicate_ids: list of all predicate symbol ids
        constant_ids: list of all constant symbol ids
    """
    

    kb_partitioned = partition(kb)

    kb_ids, vocab, predicate_ids, constant_ids = kb2ids(kb_partitioned)
    init_values = tf.random_uniform([len(vocab), input_size], minval=init[0], maxval=init[1])

    embedding_matrix = tf.Variable(init_values, name="embeddings")

    nkb = kb2nkb(kb_ids, embedding_matrix)

    return nkb, kb_ids, vocab, embedding_matrix, predicate_ids, constant_ids

