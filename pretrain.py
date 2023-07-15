"""
    Supervised contrastive pre-training of Projection_BERT model (window encoder) following the SupContrast: Supervised Contrastive Learning.
"""

import os
import sys

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, early_stopping

from model import Projection_BERT, SupConPretrain
from preprocessing import pretrain_preprocess
from dataset import ContrastivePretrainingDataset
from visualization import visualization


def pretrain_fun(encoder, max_epochs, log_dir, embedding_dim, lr, pretrain_DataLoader):

    tb_logger = TensorBoardLogger(log_dir, name="")

    checkpoint_callback = ModelCheckpoint(filename="{epoch}_{train_loss:.4f}",
                                          mode='min',
                                          save_top_k=3, 
                                          monitor='train_loss', 
                                          auto_insert_metric_name=True)
    early_stopping_callback = early_stopping.EarlyStopping(monitor="train_loss", mode='min')

#fast_dev_run=True, 
    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[checkpoint_callback, LearningRateMonitor('epoch'), early_stopping_callback],
                         log_every_n_steps=50,
                         logger=tb_logger)

    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    model = SupConPretrain(encoder, embedding_dim, lr)
    trainer.fit(model, pretrain_DataLoader)
    
    return checkpoint_callback.best_model_path


if __name__ == '__main__':

    base_dir = r"/home/zipingye/cellular-ids"
    # base_dir = r"/Users/zipingye/Desktop/cellular-ids"
    
    # It is recommended to use as many workers as possible in a data loader, which corresponds to the number of CPU cores
    num_workers = os.cpu_count()
    # num_workers = 4

    sliding_window_length = 31
    in_dim = 157

    # model architecture hyperparameters (bert config) 
    number_attention_heads = 4 # number of attention heads for multi-head self-attention layer
    number_hidden_layers = 6 # number of encoder blocks
    embedding_dimension = 32 # message embedding dimension (called hidden_size in bert config)
    max_length = sliding_window_length + 1 # max length of the input sequence

    # dirs = [f"exclude_2_attacks_version_{i}" for i in range(6)]
    dirs = ["exclude_0_attacks_version_1"]
    
    for dir in dirs:
        print("\n\n**************************************************")
        print(f"pretrain for dataset {dir}")
        # feature preprocessing
        pretrain_file = os.path.join(base_dir, "traces", dir, "pretrain.csv")
        # pretrain_output = os.path.join(base_dir, "traces/processed_pretrain.csv")
        print("\n***** preprocessing features *****")
        pretrain_df = pretrain_preprocess(pretrain_file)
        print("\n***** preprocessing finished *****")

        pretrain_Dataset = ContrastivePretrainingDataset(pretrain_df)

        pretrain_log_dir = os.path.join(base_dir, "logs", dir, "pretrained_models")

        vis_file = os.path.join(base_dir, "traces", dir, "visualization.csv")
        print("\n***** preprocessing features *****")
        vis_df = pretrain_preprocess(vis_file)
        num_samples = len(vis_df)
        print("\n***** preprocessing finished *****")

        vis_Dataset = ContrastivePretrainingDataset(vis_df)

        # instantiate a Projection_BERT model
        encoder = Projection_BERT(in_dim, # dimension of the tensor which is the concatenation of vector representation of all features in a message
                                  number_attention_heads, # number of attention heads for multi-head self-attention layer
                                  number_hidden_layers, # number of encoder blocks
                                  embedding_dimension, # message embedding dimension
                                  max_length # max length of the input sequence
                                  )

        # A common observation in contrastive learning is that the larger the batch size, the better the models perform. 
        batch_size = 256
            
        pretrain_DataLoader = DataLoader(dataset=pretrain_Dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        vis_DataLoader = DataLoader(dataset=vis_Dataset, batch_size=batch_size, num_workers=num_workers)

        model_path = pretrain_fun(encoder=encoder,
                                max_epochs=5,
                                log_dir=pretrain_log_dir,
                                embedding_dim=embedding_dimension,
                                lr=5e-4,
                                pretrain_DataLoader=pretrain_DataLoader
                                )

        print("***** finish pretraining *****")
        print(f"best model path: {model_path}")
        print("*** check the visualization of the pretrained embeddings ***")

        visualization(vis_DataLoader, model_path, num_samples, embedding_dimension, 2)
        visualization(vis_DataLoader, model_path, num_samples, embedding_dimension, 3)

        print(f"DONE -> pretrain for dataset {dir} <- DONE")
        print("**************************************************")