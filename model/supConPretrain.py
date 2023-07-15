"""
    Supervised contrastive pre-training of Projection_BERT model following the SupContrast: Supervised Contrastive Learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


class SupConPretrain(pl.LightningModule):
    
    def __init__(self, 
                 encoder, 
                 embedding_dim,
                 lr,
                 contrast_mode="all", 
                 temperature=0.07, 
                 base_temperature=0.07,
                 weight_decay=1e-4):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.save_hyperparameters()

        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'

        # used to compute contrastive loss
        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.base_temperature = base_temperature

        # Base model f(.)
        self.base_model = encoder

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.mlp = nn.Sequential(nn.Linear(self.embedding_dim , 4*self.embedding_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4*self.embedding_dim , self.embedding_dim))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                     T_max=self.hparams.max_epochs,
        #                                                     eta_min=self.hparams.lr/50)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}

    def SupConLoss(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, embedding_dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.

        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, embedding_dim],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def training_step(self, batch, batch_idx):

        windows, labels = batch
        
        window_embeddings = self.base_model(windows).view(-1, 1, self.embedding_dim)
        # print("window_embeddings shape:", window_embeddings.shape)

        loss = self.SupConLoss(window_embeddings, labels=labels)
        # Logging loss
        self.log('train_loss', loss, on_epoch=True, on_step=True) # log epoch-level metrics

        return {"loss": loss}

    def forward(self, batch): # define inference action
        pass
