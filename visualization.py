"""
    This file creates a 2D/3D visualization of the learned embedding.
"""

import os
import sys

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import ContrastivePretrainingDataset
from model import SupConPretrain
from preprocessing import pretrain_preprocess


def visualization(VDLooader, model_path, num_samples, embedding_dimension, vis_dim, num_classes=2):

    encoder = SupConPretrain.load_from_checkpoint(model_path).base_model
    encoder.eval()

    # instantiate TSNE
    tsne = TSNE(vis_dim, verbose=1, init='pca', random_state=42)

    total_window_embeddings = torch.zeros((0, embedding_dimension))
    total_labels = []

    for windows, labels in VDLooader:
        window_embeddings = encoder(windows)

        total_window_embeddings = torch.cat((total_window_embeddings, window_embeddings), dim=0)
        total_labels.extend(labels.tolist())

    total_window_embeddings = total_window_embeddings.detach().numpy()
    total_labels = np.array(total_labels)

    # project to 2D space
    tsne_proj = tsne.fit_transform(total_window_embeddings) #numpy.ndarray
    # print(tsne_proj.shape) # (num_samples, 2)

    fig = plt.figure()
    if vis_dim == 2:
        ax = fig.add_subplot()

        # Plot those points as a scatter plot and label them based on the pred labels
        for i in range(num_classes):
            ax.scatter(tsne_proj[total_labels[:num_samples]==i,1], tsne_proj[total_labels[:num_samples]==i,0])
    else: 
        ax = fig.add_subplot(projection='3d')

        # Plot those points as a scatter plot and label them based on the pred labels
        for i in range(num_classes):
            ax.scatter(tsne_proj[total_labels[:num_samples]==i,2], tsne_proj[total_labels[:num_samples]==i,1], tsne_proj[total_labels[:num_samples]==i,0])

    plt.legend([str(i) for i in range(num_classes)])
    fig_dir = "/".join(model_path.split('/')[:-1]) + "/" + f"visualization_{vis_dim}D.png"
    plt.savefig(fig_dir)


if __name__ == '__main__':

    base_dir = r"/home/zipingye/cellular-ids"
    # base_dir = r"/Users/zipingye/Desktop/cellular-ids"

    embedding_dimension = 32

    # load data
    # feature preprocessing
    vis_file = os.path.join(base_dir, "traces/exclude_0_attack/visualization.csv")
    # pretrain_output = os.path.join(base_dir, "traces/sample_pretrain.csv")
    print("***** preprocessing features *****")
    vis_df = pretrain_preprocess(vis_file)
    print("***** preprocessing finished *****")
    # A common observation in contrastive learning is that the larger the batch size, the better the models perform. 
    batch_size = 256
    # It is recommended to use as many workers as possible in a data loader, which corresponds to the number of CPU cores
    # num_workers = os.cpu_count()
    num_workers = 1
    num_classes = 2

    vis_Dataset = ContrastivePretrainingDataset(vis_df)
    vis_DataLoader = DataLoader(dataset=vis_Dataset, batch_size=batch_size, num_workers=num_workers)

    # load model
    # instantiate the model by loading weights
    model_path = os.path.join(base_dir, "logs/tb_logs/lightning_logs/version_0/checkpoints/epoch=29_train_loss=4.87.ckpt")
    num_samples = len(vis_df)

    visualization(vis_DataLoader, model_path, num_samples, 2)
    visualization(vis_DataLoader, model_path, num_samples, 3)
