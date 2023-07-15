"""
    Define customized dataset for supervised contrastive learning and classification.
"""

from lib2to3.pgen2 import token
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowEncoderDataset(Dataset):
    """Customized dataset for supervised contrastive learning.

    Args:
        df: pandas dataframe that contains data for supervised contrastive learning.
    """

    def __init__(self, df):
        self.samples = df.loc[:, "window"]
        self.labels = df.loc[:, "label"]

        self.n_samples = len(df)

    def __getitem__(self, index): # dataset[index]
        return self.samples[index], int(self.labels[index])

    def __len__(self): # len(dataset)
        return self.n_samples


class MessageTaggingDataset(Dataset):

    def __init__(self, df, sliding_window_length, in_dim, max_trace_length, val):

        self.traces = df.loc[:, "trace"]
        self.labels = df.loc[:, "label"]
        if val:
            self.attack_types = df.loc[:, "attack_type"]
        self.n_samples = len(df)
        self.sliding_window_length = sliding_window_length
        self.in_dim = in_dim
        self.max_trace_length = max_trace_length
        self.max_num_tokens = self.max_trace_length - self.sliding_window_length + 1
        self.val = val # if validation dataset

    def __getitem__(self, index): # dataset[index]

        # fetch trace (sentence) and label
        trace, label = self.traces[index], self.labels[index]
        if self.val:
            attack_type = self.attack_types[index]
        # trace: [trace_length, in_dim], label: [trace_length], attack_type: [trace_length]

        # cut the trace if longer than max_trace_length
        if len(trace) > self.max_trace_length:
            trace = trace[:self.max_trace_length]
            label = label[:self.max_trace_length]
            if self.val:
                attack_type = attack_type[:self.max_trace_length]

        # tokenize trace into windows (tokens)
        if self.val:
            token_list, label_list, attack_type_list = self.tokenize(trace, label, attack_type)
        else:
            token_list, label_list = self.tokenize(trace, label)
        # token_list: [num_tokens, window_length, in_dim], label_list: [num_tokens], attack_type_list: [num_tokens]

        # checks to ensure there is a tag for each token
        assert len(token_list) == len(label_list), "there should be a tag for each token"
        if self.val:
            assert len(label_list) == len(attack_type_list), "there should be an attack_type for each token"

        # prepare a numpy array with the data, initializing the data with pad_ind and all labels with -1
        # initializing labels to -1 differentiates tokens with tags from PADding tokens
        tokens = torch.zeros(self.max_num_tokens, self.sliding_window_length + 1, self.in_dim)
        labels = -1 * torch.ones(self.max_num_tokens, dtype=torch.long)
        if self.val:
            attack_types = -1 * torch.ones(self.max_num_tokens, dtype=torch.long)

        # copy the data to the numpy array
        cur_num_tokens = len(label_list) # number of tokens in the current trace

        tokens[:cur_num_tokens, :, :] = token_list
        labels[:cur_num_tokens] = label_list
        if self.val:
            attack_types[:cur_num_tokens] = attack_type_list
        
        if self.val:
            return tokens, labels, attack_types
        else:
            return tokens, labels
        

    def __len__(self): # len(dataset)
        return self.n_samples


    def tokenize(self, trace, label, attack_type=None):
        """Tokenize a given trace into windows. Each window is equivalent to one token in NLP.
        Also inject a tensor of all zeros as the initial value of window embedding for each token (i.e. window).

        Args:
            trace (tensor): [trace_length, in_dim]
            label (list): len = trace_length
            attack_type (list, optional): len = trace_length

        Return:
            token_list (tensor): [num_tokens, window_length, in_dim]
            label_list (tensor): [num_tokens]
            attack_type_list (tensor, optional): [num_tokens]
        """

        token_list = []
        label_list = []
        if attack_type:
            attack_type_list = []

        num_tokens = len(trace) - self.sliding_window_length + 1
        
        # if num_token is less than zero, add padding
        if num_tokens < 0:
            pad = self.sliding_window_length - len(trace)
            padding_token = torch.zeros(pad, self.in_dim)
            trace = torch.cat([padding_token, trace], dim=0)
            # the padding label does not matter since window_label = last message label
            label.extend([0] * pad)
            if attack_type:
                attack_type.extend([0] * pad)
            num_tokens = 1

        assert num_tokens > 0, "num_tokens should be greater than zero"

        for i in range(num_tokens):
            token = torch.cat([torch.zeros(1, self.in_dim), trace[i : i + self.sliding_window_length, :]])
            # token label (window label) = last message label in that window
            token_label = int(label[i + self.sliding_window_length - 1])
            token_list.append(token)
            label_list.append(token_label)
            if attack_type:
                token_attack_type = int(attack_type[i + self.sliding_window_length - 1])
                attack_type_list.append(token_attack_type)
        
        if attack_type:
            return torch.stack(token_list, dim=0), torch.tensor(label_list), torch.tensor(attack_type_list)
            # [num_tokens, window_length, in_dim], [num_tokens], [num_tokens]
        else:
            return torch.stack(token_list, dim=0), torch.tensor(label_list)
            # [num_tokens, window_length, in_dim], [num_tokens]
