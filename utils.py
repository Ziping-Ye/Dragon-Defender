
import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json


def mapping(in_seq):
    """
    This function creates a mapping from the original values of selected categorical features to a new range.

    Args:
        in_seq (list): a sequence of categorical features of the original values

    Return: 
        (list): a sequence of categorical values of the mapped values
    """
    unique_elems = list(OrderedDict.fromkeys(in_seq))
    
    # support at most 8 unique tracking area code / cell id within each window
    r = min(len(unique_elems), 8)
    
    mapped_elems = list(map(chr, range(97, 97+r)))

    # map the remaining unique elements to "unkown"
    if len(unique_elems) > r:
        mapped_elems += ["unknown"] * (len(unique_elems) - r)
    
    assert len(mapped_elems) == len(unique_elems), "length of mapped elements and unique elements should be the same"
    mapping = dict(zip(unique_elems, mapped_elems))

    return [mapping[elem] for elem in in_seq]


def exclude_attacks(pretrain_train_trace_list, num, attack_names):
    """
    This function is used to exclude certain number of attacks from pretrain and train dataset to demonstrate
    the generalization ability of the model.
    Args:
        pretrain_train_trace_list (list): list of traces used for pretrain and train
        num (int): number of attacks to be excluded.
    Returns:
        filtered_pretrain_train_trace_list (list): list of traces after excluding attacks.
    """
    ex_attacks = random.sample(attack_names, num)
    print("will exclude the following attacks from pretrain and train")
    print(ex_attacks)

    filtered_pretrain_train_trace_list = []
    dropped_pretrain_train_trace_list = []

    for trace in pretrain_train_trace_list:
        keep = True
        for ex_attack in ex_attacks:
            if ex_attack in trace:
               keep = False
               break
        
        if keep:
            filtered_pretrain_train_trace_list.append(trace)
        else:
            dropped_pretrain_train_trace_list.append(trace)

    print(f"before excluding {num} attacks, there are {len(pretrain_train_trace_list)} traces")
    print(f"after excluding, there are {len(filtered_pretrain_train_trace_list)} traces")

    assert len(filtered_pretrain_train_trace_list) + len(dropped_pretrain_train_trace_list) == len(pretrain_train_trace_list), "kept and dropped traces should be add up to total number of pretrain_trace_traces"

    return filtered_pretrain_train_trace_list, dropped_pretrain_train_trace_list, ex_attacks


def transform(lst):
    """
    This function takes in a list of string where each string is one feature sequence, 
    convert each string to list, and transpose the list in order to use sklearn one-hot encoder.
    
    Note: only use for categorical features

    Args:
        lst (list): a list of string where each string is one feature sequence

    Return:
        transpose_lst (list): a list of list
    """
    # step1: convert to a list of list
    for i, feat_seq in enumerate(lst):
        lst[i] = feat_seq.split()

    # step2: transpose the list of list (2 dimensional matrix)
    np_arr = np.array(lst, dtype=object)
    transpose = np_arr.T
    transpose_lst = transpose.tolist()
    return transpose_lst


def construct_one_hot_encoder(base_dir, categorical_features):
    """
       This function constructs a one-hot encoder with spedified values for each categorical feature.
    """
    # step 1: construct a dummy example
    with open(os.path.join(base_dir, "helpers/categorical_features.json"), 'r') as f:
        example_dict = json.load(f)

    # print("example dictionary = ", example_dict)

    # find out the max sequence length and pad shorter sequence
    max_seq_len = max([len(lst) for lst in list(example_dict.values())]) # 97
    # print("max sequence length = ", max_seq_len) # 97

    # pad every feature_seq to max_seq_len
    for key, value in example_dict.items():
        # print("value type:", type(value)) # list
        if len(value) < max_seq_len:
            pad = max_seq_len - len(value)
            value = value + [value[0]] * pad

        example_dict[key] = value
    # print("example dictionary after padding = ", example_dict)

    dummy_example = list(example_dict.values())
    # print(dummy_example)
    # transpose dummy example in order to use sklearn one-hot encoder
    np_arr = np.array(dummy_example)
    transpose = np_arr.T
    dummy_example = transpose.tolist()
    # print(dummy_example)
    
    # step 2: fit the one-hot encoder    
    OneHotEnc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OneHotEnc.fit(dummy_example)
    # print("one hot encoder categories:")
    # print(OneHotEnc.categories_)
    return OneHotEnc

