"""
    1. customize BERT config
        reference:
            https://huggingface.co/docs/transformers/main_classes/configuration
            https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig
    2. define the model
        input: a tensor of shape (batch_size, max sequence length, in_dim) 
               each tensor is the concatenation of vector representation of all features in a message
        output: window embedding - a tensor (batch_size, embedding_dim)
"""

import torch
import torch.nn as nn 
from transformers import BertModel, BertConfig, PretrainedConfig


class Projection_BERT(nn.Module):

    def __init__(self,
                 in_dim, # dimension of the tensor which is the concatenation of vector representation of all features in a message
                 num_attention_heads, # number of attention heads for multi-head self-attention layer
                 num_hidden_layers, # number of encoder blocks
                 embedding_dim, # message embedding dimension
                 max_length # max length of the input sequence
                 ): 

        super().__init__()

        # project the tensor that represents all features of a message to the pre-defined raw message embedding dimension
        self.fc = nn.Linear(in_dim, embedding_dim)
        # maybe try multiple fc layers
        # self.fc = nn.Sequential(
        #         nn.Linear(in_dim, intermidiate_dim),
        #         nn.ReLU(),
        #         nn.Linear(intermidiate_dim, embedding_dim)
        # )

        # BERT model with user defined config hyperparameters
        # Initializing a BERT bert-base-uncased style configuration
        config_dict = BertConfig().to_dict()

        config_dict["num_attention_heads"] = num_attention_heads 
        config_dict["num_hidden_layers"] = num_hidden_layers
        config_dict["hidden_size"] = embedding_dim
        config_dict["max_length"] = max_length

        # Initializing a model from the user defined configuration
        bert = BertModel(PretrainedConfig.from_dict(config_dict))

        # confirm the config is updated with user defined value
        # print(bert.config)

        self.bert = bert


    def forward(self, window): 

        # input a list of tensors
        # window: [batch_size, window length, in_dim]
        # each tensor in the list will pass through a shared fc layer to project to the embedding dimension
        # print("window type and shape:", type(window), window.shape)
        # print("projected msg type and shape:", type(self.fc(window[0])), self.fc(window[0]).shape)

        projected = torch.stack([self.fc(msg_seq) for msg_seq in window], dim=0)
        # print("projected type and shape:", type(projected), projected.shape)
        # projected: [batch_size, window length, embedding_dim]

        # an alternative approach
        # projected = []
        # for batch in range(window.shape[0]):
        #     projected.append(torch.stack([self.fc(msg) for msg in window[batch,:,:]], dim=0))
        # projected = torch.stack(projected, dim=0)

        embedded = self.bert(inputs_embeds=projected)[0]
        # embedded: [batch_size, window length, embedding_dim]
        # print("embedded shape:", embedded.shape)

        # return window embedding
        # print("right before returning:", embedded[:,0,:].shape)
        return embedded[:, 0, :]
