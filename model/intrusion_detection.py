"""
    Intrusion detection is modeled as a classification of sliding windows in a trace.
    We use LSTM + linear_classifier to classify/tag each window as intrusion detection system (IDS).
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

from .supConPretrain import SupConPretrain
from .metrics import AttackTypeMetric 


class IntrusionDetection(pl.LightningModule):

    def __init__(self, 
                 window_encoder_path, 
                 embedding_dim, 
                 hidden_dim, 
                 lr, 
                 attack_type_dict,
                 num_layers=2, 
                 dropout=0.25, 
                 num_classes=2
                 ):

        super().__init__()
        self.lr = lr
        self.attack_type_dict = attack_type_dict
        self.save_hyperparameters()
        # Automatically loads the model with the saved hyperparameters
        self.window_encoder = SupConPretrain.load_from_checkpoint(window_encoder_path).base_model 
        
        # freeze parameters in window encoder
        for param in self.window_encoder.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_dim,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout if num_layers < 2 else dropout)

        self.cls = nn.Linear(hidden_dim, num_classes)

        # define metrics to evaluate the model performance
        # accuracy
        self.train_acc = Accuracy(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
        self.val_acc = Accuracy(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
        
        # precision
        self.train_pre = Precision(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
        self.val_pre = Precision(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
       
        # recall
        self.train_recall = Recall(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
        self.val_recall = Recall(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
       
        # f1 score
        self.train_f1score = F1Score(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
        self.val_f1score = F1Score(task="binary", num_classes=num_classes, ignore_index=-1, validate_args = False)
       
        # confusion matrix
        # self.confmat = ConfusionMatrix(num_classes=num_classes)

        # attack specific metric
        for attack_type, attack_type_label in self.attack_type_dict.items():
            setattr(self, f"{attack_type}_metric", AttackTypeMetric(attack_type_label))

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):

        traces, labels = batch
        # print("traces shape: ", traces.shape)
        # [batch_size, seq_length, window_length, in_dim]
        # print("labels shape: ", labels.shape)
        # [batch_size, seq_length]

        window_embedding_list = []

        with torch.no_grad():
            for i in range(traces.shape[1]): # seq_length
                window = traces[:,i,:,:]
                # print("window shape: ", window.shape)
                # [batch_size, window_length, in_dim]
                window_embedding = self.window_encoder(window)
                # print("window_embedding shape: ", window_embedding.shape)
                # [batch_size, embedding_dim]
                window_embedding_list.append(window_embedding)
        
        window_embeddings = torch.stack(window_embedding_list, dim=1)
        # print("window_embeddings shape: ", window_embeddings.shape)
        # [batch_size, seq_length, embedding_dim]
        
        embeds, _ = self.lstm(window_embeddings)
        # print("embeds shape (before view): ", embeds.shape)
        # [batch_size, seq_length, hidden_dim]
        embeds = embeds.contiguous()

        # reshape the embeds so that each row contains one token (window)
        embeds = embeds.view(-1, embeds.shape[2])
        # print("embeds shape (after view): ", embeds.shape)
        # [batch_size * seq_length, hidden_dim]        

        preds = self.cls(embeds)
        # print("preds shape: ", preds.shape)
        # [batch_size * seq_length, num_classes]
        preds = F.log_softmax(preds, dim=1)
        # print("preds shape: ", preds.shape)
        # [batch_size * seq_length, num_classes]

        loss = self.calculate_loss(preds, labels)

        self.log("train_loss", loss, on_epoch=True, on_step=False)

        labels = labels.view(-1)
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)

        # since the torchmetrics ignore_index is not working for negative index currently (https://github.com/Lightning-AI/metrics/issues/1040)
        # let's manually ignore the paddings and their labels
        mask = torch.where(labels >= 0, True, False)
        indices = torch.nonzero(mask).flatten()
        preds = torch.index_select(preds, 0, indices)
        labels = torch.index_select(labels, 0, indices)

        self.train_pre(preds, labels)
        self.log("train_precision", self.train_pre, on_epoch=True, on_step=False)

        self.train_recall(preds, labels)
        self.log("train_recall", self.train_recall, on_epoch=True, on_step=False)

        self.train_f1score(preds, labels)
        self.log("train_f1score", self.train_f1score, on_epoch=True, on_step=False)

        return loss

    def calculate_loss(self, outputs, labels):
        """
        Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
        for PADding tokens.
        Args:
            outputs: dimension batch_size*seq_len x num_class - log softmax output of the model
            labels: dimension batch_size x seq_len where each element is either a label in [0, 1],
                    or -1 in case it is a PADding token.
        Returns:
            loss: cross entropy loss for all tokens in the batch
        Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
            demonstrates how you can easily define a custom loss function.
        """

        # reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.view(-1)

        # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
        mask = (labels >= 0).float()

        # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
        # number. This does not affect training, since we ignore the PADded tokens with the mask.
        labels = labels % outputs.shape[1]

        num_tokens = int(torch.sum(mask))

        # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
        return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens

    def validation_step(self, batch, batch_idx):
        
        traces, labels, attack_types = batch
        # print("traces shape: ", traces.shape) # [batch_size, seq_length, window_length, in_dim]
        # print("labels shape: ", labels.shape) # [batch_size, seq_length]
        # print("attack_types shape: ", attack_types.shape) # [batch_size, seq_length]

        window_embedding_list = []

        for i in range(traces.shape[1]):
            window = traces[:,i,:,:] # [batch_size, window_length, in_dim]
            # print("window shape: ", window.shape)
            window_embedding = self.window_encoder(window) # [batch_size, embedding_dim]
            # print("window_embedding shape: ", window_embedding.shape)
            window_embedding_list.append(window_embedding)
        
        window_embeddings = torch.stack(window_embedding_list, dim=1) # [batch_size, seq_length, embedding_dim]
        embeds, _ = self.lstm(window_embeddings) # [batch_size, seq_length, hidden_dim]
        embeds = embeds.contiguous()
        embeds = embeds.view(-1, embeds.shape[2]) # [batch_size * seq_length, hidden_dim]   
        preds = self.cls(embeds) # [batch_size * seq_length, num_classes]
        preds = F.log_softmax(preds, dim=1) # [batch_size * seq_length, num_classes]
        val_loss = self.calculate_loss(preds, labels)

        self.log("val_loss", val_loss, on_epoch=True, on_step=False)
        labels = labels.view(-1)
        attack_types = attack_types.view(-1)
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, on_epoch=True, on_step=False)

        # since the torchmetrics ignore_index is not working for negative index currently (https://github.com/Lightning-AI/metrics/issues/1040)
        # let's manually ignore the paddings and their labels
        mask = torch.where(labels >= 0, True, False)
        indices = torch.nonzero(mask).flatten()
        preds = torch.index_select(preds, 0, indices)
        labels = torch.index_select(labels, 0, indices)
        attack_types = torch.index_select(attack_types, 0, indices)

        self.val_pre(preds, labels)
        self.log("val_precision", self.val_pre, on_epoch=True, on_step=False)

        self.val_recall(preds, labels)
        self.log("val_recall", self.val_recall, on_epoch=True, on_step=False)

        self.val_f1score(preds, labels)
        self.log("val_f1score", self.val_f1score, on_epoch=True, on_step=False)

        # self.confmat(preds, labels)
        # self.log("confusion_matrix", self.confmat, on_epoch=True, on_step=False)

        # attack specific metric
        for attack_type, attack_type_label in self.attack_type_dict.items():
            getattr(self, f"{attack_type}_metric").update(preds, labels, attack_types)
            
    
    def validation_epoch_end(self, outputs):
        for attack_type, attack_type_label in self.attack_type_dict.items():
            self.log(f"{attack_type}_metric", getattr(self, f"{attack_type}_metric").compute())
            getattr(self, f"{attack_type}_metric").reset()


    def forward(self, trace): # Use for inference only

        window_embedding_list = []

        for i in range(trace.shape[1]):
            window = trace[:,i,:,:] # [batch_size, window_length, in_dim]
            # print("window shape: ", window.shape)
            window_embedding = self.window_encoder(window) # [batch_size, embedding_dim]
            # print("window_embedding shape: ", window_embedding.shape)
            window_embedding_list.append(window_embedding)
        
        window_embeddings = torch.stack(window_embedding_list, dim=1) # [batch_size, seq_length, embedding_dim]
        embeds, _ = self.lstm(window_embeddings) # [batch_size, seq_length, hidden_dim]
        embeds = embeds.contiguous()
        embeds = embeds.view(-1, embeds.shape[2]) # [batch_size * seq_length, hidden_dim]   
        preds = self.cls(embeds) # [batch_size * seq_length, num_classes]
        preds = F.log_softmax(preds, dim=1) # [batch_size * seq_length, num_classes]

