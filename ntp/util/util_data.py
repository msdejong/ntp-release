"""Utility for data processing"""

from time import gmtime, strftime
import os
import json

import numpy as np
import tensorflow as tf


def save_conf(path, conf):
    """save conf file in path"""
    with open(path, "w") as f_out:
        splits = path.split("/")
        dir = "/".join(splits[:-1]) + "/"
        conf["meta"]["experiment_dir"] = dir
        json.dump(conf, f_out, indent=4, sort_keys=True)
        f_out.close()


def deep_merge(dict1, dict2):
    """overrides entries in dict1 with entries in dict2 recursively"""
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        tmp = {}
        for key in dict1:
            if key not in dict2:
                tmp[key] = dict1[key]
            else:
                tmp[key] = deep_merge(dict1[key], dict2[key])
        for key in dict2:
            if key not in dict1:
                tmp[key] = dict2[key]
        return tmp
    else:
        return dict2

# Load config file into dict
def load_conf(path):
    """load conf file from path"""
    file_name = path.split("/")[-1]

    with open(path, 'r') as f:
        conf = eval(f.read())

        if "meta" not in conf:
            conf["meta"] = {}

        conf["meta"]["conf"] = path
        conf["meta"]["name"] = file_name.split(".")[0]
        conf["meta"]["file_name"] = file_name
        
        # load parent dict, override with highest level dict
        if "parent" in conf["meta"] and conf["meta"]["parent"] is not None:
            parent = load_conf(conf["meta"]["parent"])
            conf = deep_merge(parent, conf)  # {**parent, **conf}

        f.close()

        return conf

class Vocab(object):

    def __init__(self):
        """
        Creates Vocab object. Used as a dictionary between symbols and symbol ids
        """
        self.next_neg = -1
        self.unk = "<UNK>"
        self.emb = lambda _:None #if emb is None: same behavior as for o-o-v words

        self.sym2id = {}
        # with pos and neg indices
        self.id2sym = {}
        self.next_pos = 0
        self.sym2freqs = {}

        self.sym2id[self.unk] = 0
        # with pos and neg indices
        self.id2sym[0] = self.unk
        self.next_pos = 1
        self.sym2freqs[self.unk] = 0
        
        self.emb_length = None


    def get_id(self, sym):
        """
        returns internal id, that is, positive for out-of-vocab symbol, negative for symbol found in `self.emb`. 
        If `sym` is a new symbol, it is added to the Vocab.
        Args:
            `sym`: symbol (e.g., token)
        """
        
        vec = self.emb(sym)
        if self.emb_length is None and vec is not None:
            self.emb_length = len(vec) if isinstance(vec, list) else vec.shape[0]
        if sym not in self.sym2id:
            if vec is None:
                self.sym2id[sym] = self.next_pos
                self.id2sym[self.next_pos] = sym
                self.next_pos += 1
            else:
                self.sym2id[sym] = self.next_neg
                self.id2sym[self.next_neg] = sym
                self.next_neg -= 1
            self.sym2freqs[sym] = 1
        else:
            self.sym2freqs[sym] += 1
        if sym in self.sym2id:
            return self.sym2id[sym]
        else:
            if self.unk in self.sym2id:
                return self.sym2id[self.unk]
            # can happen for `Vocab` initialized with `unk` argument set to `None`
            else:
                return None

    def __call__(self, *args, **kwargs):
        """
        calls the `get_id` function for the provided symbol(s), which adds symbols to the Vocab if needed and allowed,
        and returns their id(s).
        Args:
            *args: a single symbol, a list of symbols, or multiple symbols
        """
        symbols = args
        if len(args) == 1:
            if isinstance(args[0], list):
                symbols = args[0]
            else:
                return self.get_id(args[0])
        return [self.get_id(sym) for sym in symbols]

    def __contains__(self, sym):
        """checks if `sym` already in the Vocab object"""
        return sym in self.sym2id

    def __len__(self):
        """returns number of unique symbols (including the unknown symbol)"""
        return len(self.id2sym)
