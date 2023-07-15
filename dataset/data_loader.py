"""
    This file implements customized data loader for LSTM + linear classifier.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

class DataLoader(object):

    def __init__(self, batch_size, num_epochs, sliding_window_length, in_dim):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sliding_window_length = sliding_window_length
        self.in_dim = in_dim
        self.data = {}

    def tokenize(self, trace, label):
        """Tokenize a given trace into windows. Each window is equivalent to one token in NLP.
        Also inject a tensor of all zeros as the initial value of window embedding for each token (i.e. window).

        Args:
            trace (tensor): [trace_length, in_dim]
            label (list): len = trace_length

        Return:
            token_list (tensor): [num_tokens, window_length, in_dim]
            label_list (tensor): [num_tokens]
        """

        token_list = []
        label_list = []
        num_tokens = len(trace) - self.sliding_window_length + 1

        for i in range(num_tokens):
            token = torch.cat([torch.zeros(1, self.in_dim), trace[i : i + self.sliding_window_length, :]])
            # token label (window label) = last message label in that window
            token_label = int(label[i + self.sliding_window_length - 1])
            token_list.append(token)
            label_list.append(token_label)

        return torch.stack(token_list, dim=0), torch.tensor(label_list)


    def load_traces_and_labels(self, df):

        traces = []
        labels = []

        for _, row in df.iterrows():

            trace, label = row.loc['window'], row.loc['label']

            # ignore traces shorter than 32
            if len(label) < 32:
                continue

            token_list, label_list = self.tokenize(trace, label)
            traces.append(token_list)
            labels.append(label_list)

        # checks to ensure there is a tag for each token
        assert len(traces) == len(labels)
        for i in range(len(labels)): # for each sentence, the number of tokens and number of tages should be equal
            assert len(traces[i]) == len(labels[i]), "there should be a tag for each token"

        # store traces and labels to self.data (dict)
        self.data["traces"] = traces
        self.data["labels"] = labels
        self.data["size"] = len(traces) # number of examples (sentences, i.e traces)


    def data_iterator(self, shuffle=False):

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(self.data['size']))

        if shuffle:
            random.seed(2023)
            random.shuffle(order)

        # one pass over the data (aka, one epoch)
        num_steps = (self.data['size']+1) // self.batch_size # entire data will be splited into these many of batches

        for epoch in range(self.num_epochs):

            for i in range(num_steps):
                # fetch traces and labels
                batch_traces = [self.data['traces'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
                batch_tags = [self.data['labels'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
            
                # compute length of longest sentence in batch
                batch_max_len = max([len(s) for s in batch_traces])

                # prepare a numpy array with the data, initializing the data with pad_ind and all labels with -1
                # initializing labels to -1 differentiates tokens with tags from PADding tokens
                batch_data = np.zeros((len(batch_traces), batch_max_len, self.sliding_window_length + 1, self.in_dim))
                batch_labels = -1 * np.ones((len(batch_traces), batch_max_len))

                # copy the data to the numpy array
                for j in range(len(batch_traces)): # for each trace in this batch
                    cur_len = len(batch_traces[j]) # length of the current trace
                    batch_data[j][:cur_len] = batch_traces[j]
                    batch_labels[j][:cur_len] = batch_tags[j]
                
                batch_data, batch_labels = torch.Tensor(batch_data), torch.LongTensor(batch_labels)

                batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

                yield batch_data, batch_labels
