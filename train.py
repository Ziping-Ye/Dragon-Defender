"""
    Supervised training of linear classifier on window embeddings. i.e. fine-tuning
"""

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, early_stopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from model import IntrusionDetection
from preprocessing import train_preprocess
from dataset import WindowTaggingDataset


def train_fun(IDS, max_epochs, base_dir, log_dir, train_DataLoader, val_DataLoader):

    # tb_logger = TensorBoardLogger(log_dir, name="tb_log")
    csv_logger = CSVLogger(log_dir, name="csv_log")

    metric_csv_path = os.path.join(log_dir, "csv_log", "metrics.csv")
    # log metric csv file path
    with open(os.path.join(base_dir, "metric_csv_paths.txt"), 'a') as f:
        f.write(metric_csv_path)

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(log_dir, "checkpoints"),
                                          filename="{epoch}_{train_loss:.4f}",
                                          mode='min',
                                          save_top_k=1, 
                                          monitor='train_loss', 
                                          auto_insert_metric_name=True)

    early_stopping_callback = early_stopping.EarlyStopping(monitor="train_loss", mode='min')

    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[checkpoint_callback, LearningRateMonitor('epoch'), early_stopping_callback],
                         logger=[csv_logger])
    
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    trainer.fit(IDS, train_DataLoader, val_DataLoader)
    # trainer.tune(intrusion_detection, train_loader, val_loader)
    return checkpoint_callback.best_model_path


if __name__ == '__main__':

    pl.seed_everything(42) # To be reproducable

    base_dir = r"/home/zipingye/cellular-ids"
    # base_dir = r"/Users/zipingye/Desktop/cellular-ids"

    batch_size = 8
    # num_workers = os.cpu_count()
    num_workers = 1
    max_trace_length = 128

    sliding_window_length = 31
    in_dim = 157
    num_epochs = 20
    embedding_dimension = 32
    lstm_hidden_dim = 256
    train_lr = 0.03

    dir_name = "exclude_0_attacks_version_1"
    train_log_dir = os.path.join(base_dir, "logs", dir_name, "intrusion_detection_system")

    train_file = os.path.join(base_dir, "traces", dir_name, "train.csv")
    val_file = os.path.join(base_dir, "traces", dir_name, "validation.csv")

    print("***** preprocessing features for training *****")
    train_df = train_preprocess(train_file, False)
    print("***** preprocessing finished *****\n")
    print("***** preprocessing features for validation *****")
    val_df = train_preprocess(val_file, True)
    print("***** preprocessing finished *****")

    window_encoder_path = r"/home/zipingye/cellular-ids/logs/exclude_0_attacks_version_1/pretrained_models/version_0/checkpoints/epoch=11_train_loss=4.9307.ckpt"
    IDS = IntrusionDetection(window_encoder_path, embedding_dimension, lstm_hidden_dim, train_lr)

    train_dataset = WindowTaggingDataset(train_df, sliding_window_length, in_dim, max_trace_length, False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = WindowTaggingDataset(val_df, sliding_window_length, in_dim, max_trace_length, True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    best_model_path = train_fun(IDS, num_epochs, base_dir, train_log_dir, train_loader, val_loader)

    print("excellent job!!!")
    print(f"check out best model path at {best_model_path}")

