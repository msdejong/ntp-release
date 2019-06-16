""""""

import re
import collections
import numpy as np

Atom = collections.namedtuple("Atom", ["predicate", "arguments"])


def trim(string):
    """
    :param string: an input string
    :return: the string without trailing whitespaces
    """
    return re.sub("\A\s+|\s+\Z", "", string)


def is_atom(arg):
    return isinstance(arg, Atom)


def is_variable(arg):
    if isinstance(arg, str):
        return arg.isupper()
    else:
        return False


def is_list(arg):
    return isinstance(arg, list)


def is_array(arg):
    return isinstance(arg, np.ndarray)


def is_constant(arg):
    if isinstance(arg, str):
        return arg.islower()
    else:
        return False


def is_parameter(predicate):
    if isinstance(predicate, str):
        return predicate[0] == "#"
    else:
        return False


def atom2string(atom):
    return "%s(%s)" % (atom.predicate, ",".join(atom.arguments))


def rule2string(rule):
    head = atom2string(rule[0])
    body = [atom2string(x) for x in rule[1:]]
    if len(rule) == 1:
        return "%s." % head
    else:
        return "%s :- %s." % (head, ", ".join(body))


def subs2string(substitutions):
    return "{%s}" % ", ".join([key+"/"+val for
                               key, val in substitutions.items()])


def is_ground_atom(atom):
    if is_atom(atom):
        return len([x for x in atom.arguments if is_variable(x)]) == 0
    else:
        return False


def has_free_variables(rule):
    return len([atom for atom in rule if not is_ground_atom(atom)]) > 0


def parse_rules(rules, delimiter="#####", rule_template=False):
    """
    :param rules:
    :param delimiter:
    :return:
    """
    kb = []
    for rule in rules:
        if rule_template:
            splits = re.split("\A\n?([0-9]?[0-9]+)", rule)
            # fixme: should be 0 and 1 respectively
            num = int(splits[1])
            rule = splits[2]
        rule = re.sub(":-", delimiter, rule)
        rule = re.sub("\),", ")"+delimiter, rule)
        rule = [trim(x) for x in rule.split(delimiter)]
        rule = [x for x in rule if x != ""]
        if len(rule) > 0:
            atoms = []
            for atom in rule:
                splits = atom.split("(")
                predicate = splits[0]
                args = [x for x in re.split("\s?,\s?|\)", splits[1]) if x != ""]
                atoms.append(Atom(predicate, args))
            if rule_template:
                kb.append((atoms, num))
            else:
                kb.append(atoms)
    return kb


def load_from_file(path, rule_template=False):
    with open(path, "r") as f:
        text = f.readlines()
        text = [x for x in text if not x.startswith("%") and x.strip() != ""]
        text = "".join(text)
        rules = [x for x in re.split("\.\n|\.\Z", text) if x != "" and
                 x != "\n" and not x.startswith("%")]
        kb = parse_rules(rules, rule_template=rule_template)

    return kb

def load_from_list(text, rule_template=False):
    text = [x for x in text if not x.startswith("%") and x.strip() != ""]
    text = "".join(text)
    rules = [x for x in re.split("\.\n|\.\Z", text) if x != "" and
                 x != "\n" and not x.startswith("%")]
    kb = parse_rules(rules, rule_template=rule_template)
    return kb

def normalize(kb):
    counter = 0
    normalized_kb = []

    def suffix_variables(atom, suffix):
        new_args = []
        for arg in atom.arguments:
            if is_variable(arg):
                new_args.append(arg+suffix)
            else:
                new_args.append(arg)
        return Atom(atom.predicate, new_args)

    for rule in kb:
        if has_free_variables(rule):
            normalized_kb.append([suffix_variables(atom, str(counter))
                                  for atom in rule])
            counter += 1
        else:
            normalized_kb.append(rule)
    return normalized_kb

def relationship_id_to_symbol(relationships):
    symbol_relationships = {}
    for head, body in relationships.items():
        symbol_relationships["Predicate" + str(head)] = set(["Predicate" + str(body_pred) for body_pred in body])
    return symbol_relationships