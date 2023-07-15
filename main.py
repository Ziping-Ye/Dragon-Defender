"""
    Full automation from preparing dataset, pretrain, visualization and train.
"""

import os
import sys
import argparse
import random

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import Projection_BERT, IntrusionDetection
from preprocessing import pretrain_preprocess, train_preprocess, prepare_one_dataset
from dataset import ContrastivePretrainingDataset, WindowTaggingDataset
from pretrain import pretrain_fun
from train import train_fun
from visualization import visualization


if __name__ == "__main__":

    # *** parse command line arguments
    parser = argparse.ArgumentParser("argument for full automation")
    parser.add_argument('--ex_num', type=int, default=0,
                        help='how many attack(s) want to exclude from pretrain and train')
    parser.add_argument('--version', type=int, default=1,
                        help='version number of the current experiment')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='number of pretrain epochs')
    parser.add_argument('--train_epochs', type=int, default=100,
                        help='number of train epochs')
    parser.add_argument('--sw_length', type=int, default=31,
                        help='length of the sliding window')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='dimension of the window embedding space') #called hidden_size in bert config
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='number of attention heads in the multi-head self-attention mechanism')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='number of encoder layers in BERT')
    parser.add_argument('--lstm_hidden_dim', type=int, default=128,
                        help='dimension of the LSTM hidden state')

    args = parser.parse_args()
    exclude_attack_num = args.ex_num
    version_num = args.version
    pretrain_epochs = args.pretrain_epochs
    train_epochs = args.train_epochs
    sliding_window_length = args.sw_length
    embedding_dimension = args.embedding_dim
    number_attention_heads = args.num_attention_heads # number of attention heads for multi-head self-attention layer
    number_hidden_layers = args.num_encoder_layers # number of encoder blocks
    lstm_hidden_dim = args.lstm_hidden_dim

    # *** parameters ***
    base_dir = r"/home/zipingye/cellular-ids"
    # base_dir = r"/Users/zipingye/Desktop/cellular-ids"
    in_dim = 157
    # It is recommended to use as many workers as possible in a data loader, which corresponds to the number of CPU cores
    # num_workers = os.cpu_count()
    num_workers = 4
    attack_type_dict = {
        "benign" : 0,
        "AKA_Bypass" : 1,
        "Attach_Reject" : 2, 
        "EMM_Information" : 3, 
        "IMEI_Catching" : 4,
        "IMSI_Catching" : 5, 
        "Malformed_Identity_Request" : 6, 
        "Null_Encryption" : 7,
        "Numb_Attack" : 8, 
        "Repeat_Security_Mode_Command" : 9, 
        "RLF_Report" : 10,
        "Service_Reject" : 11, 
        "TAU_Reject" : 12,
        "IMSI_Cracking" : 13,
        "IMSI_Cracking_Reduced" : 14,
        "Paging_Channel_Hijacking" : 15
    }

    # ***** prepare dataset *****
    # *** prepare dataset parameters ***
    val_portion = 0.075
    attack_names = ["AKA_Bypass", "Attach_Reject", "EMM_Information", "IMEI_Catching",
                    "IMSI_Catching", "Malformed_Identity_Request", "Null_Encryption",
                    "Numb_Attack", "Repeat_Security_Mode_Command", "RLF_Report",
                    "Service_Reject", "TAU_Reject", "IMSI_Cracking", "IMSI_Cracking_Reduced", "Paging_Channel_Hijacking"]
    
    attack_names_to_count = ["AKA_Bypass", "Attach_Reject", "EMM_Information", "IMEI_Catching",
                            "IMSI_Catching", "Malformed_Identity_Request", "Null_Encryption",
                            "Numb_Attack", "Repeat_Security_Mode_Command", "RLF_Report",
                            "Service_Reject", "TAU_Reject", "Paging_Channel_Hijacking"]

    feature_list = ["message name", "Attach with IMSI", "Null encryption", 
                    "Enable IMEISV", "Cell ID", "TAC", "EMM state", 
                    "EMM substate", "EMM cause", "paging_record_number"]

    benign_trace_dir = os.path.join(base_dir, "traces/benign_traces")
    attack_trace_dir = os.path.join(base_dir, "traces/attack_traces")
    traces_directory = {"benign": benign_trace_dir, "attack": attack_trace_dir}

    print("\n******************************************************")
    print("***** prepare dataset *****")
    dir_name = prepare_one_dataset(exclude_attack_num, version_num, attack_names, attack_names_to_count,
                                   feature_list, traces_directory, val_portion, sliding_window_length)

    # ***** pretrain *****
    print("\n\n******************************************************")
    print("***** pretraining *****")
    pl.seed_everything(42) # To be reproducable
    # *** pretrain parameters ***
    pretrain_log_dir = os.path.join(base_dir, "logs", dir_name, "pretrained_models")
    pretrain_lr = 5e-4
    # A common observation in contrastive learning is that the larger the batch size, the better the models perform. 
    pretrain_batch_size = 256

    # window encoder architecture hyperparameters (bert config) 
    max_length = sliding_window_length + 1 # max length of the input sequence

    # feature preprocessing
    pretrain_file = os.path.join(base_dir, "traces", dir_name, "pretrain.csv")
    print("\n***** preprocessing features for pretraining *****")
    pretrain_dataframe = pretrain_preprocess(pretrain_file)
    print("***** preprocessing finished *****")
    pretrain_Dataset = ContrastivePretrainingDataset(pretrain_dataframe)

    vis_file = os.path.join(base_dir, "traces", dir_name, "visualization.csv")
    print("\n***** preprocessing features for visualization *****")
    vis_dataframe = pretrain_preprocess(vis_file)
    num_samples = len(vis_dataframe)
    print("***** preprocessing finished *****")
    vis_Dataset = ContrastivePretrainingDataset(vis_dataframe)

    # instantiate a Projection_BERT model as window encoder
    window_encoder = Projection_BERT(in_dim, # dimension of the tensor which is the concatenation of vector representation of all features in a message
                                     number_attention_heads, # number of attention heads for multi-head self-attention layer
                                     number_hidden_layers, # number of encoder blocks
                                     embedding_dimension, # message embedding dimension
                                     max_length) # max length of the input sequence
        
    pretrain_DataLoader = DataLoader(dataset=pretrain_Dataset, batch_size=pretrain_batch_size, shuffle=True, num_workers=num_workers)
    vis_DataLoader = DataLoader(dataset=vis_Dataset, batch_size=pretrain_batch_size, num_workers=num_workers)

    # window_encoder_path is the absolute path
    window_encoder_path = pretrain_fun(window_encoder,
                                       pretrain_epochs,
                                       pretrain_log_dir,
                                       embedding_dimension,
                                       pretrain_lr,
                                       pretrain_DataLoader)

    print("***** finish pretraining *****")
    print("*** check the visualization of the pretrained embeddings ***")
    
    # *** visualization ***
    visualization(vis_DataLoader, window_encoder_path, num_samples, embedding_dimension, 2)
    visualization(vis_DataLoader, window_encoder_path, num_samples, embedding_dimension, 3)

    # ***** train *****
    print("\n\n******************************************************")
    print("***** training *****")
    # *** train parameters ***
    train_log_dir = os.path.join(base_dir, "logs", dir_name, "intrusion_detection_system")
    train_batch_size = 16
    max_trace_length = 256
    train_lr = 0.01

    train_file = os.path.join(base_dir, "traces", dir_name, "train.csv")
    val_file = os.path.join(base_dir, "traces", dir_name, "validation.csv")

    print("***** preprocessing features for training *****")
    train_dataframe = train_preprocess(train_file, False)
    print("***** preprocessing finished *****\n")
    print("***** preprocessing features for validation *****")
    val_dataframe = train_preprocess(val_file, True)
    print("***** preprocessing finished *****")

    intrusion_detction_sys = IntrusionDetection(window_encoder_path, embedding_dimension, lstm_hidden_dim, train_lr)

    train_dataset = WindowTaggingDataset(train_dataframe, sliding_window_length, in_dim, max_trace_length, False)
    train_DataLoader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = WindowTaggingDataset(val_dataframe, sliding_window_length, in_dim, max_trace_length, True)
    val_DataLoader = DataLoader(dataset=val_dataset, batch_size=train_batch_size, num_workers=num_workers)

    IDS_path = train_fun(intrusion_detction_sys, train_epochs, base_dir, train_log_dir, train_DataLoader, val_DataLoader)

    print("******* training finished *******")
    print(f"check out best model path at {IDS_path}")

