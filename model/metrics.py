"""
    Implementation of a custom metric that computes the detection rate for each type of attack.
"""

import torch
from torchmetrics import Metric


class AttackTypeMetric(Metric):
    """
        Compute detection rate for each type of attack.
    """

    def __init__(self, attack_type_label):
        super().__init__()
        self.attack_type_label = attack_type_label # an integer that represents a specific attack type
        # dist_reduce_fx="sum" Function to reduce state across multiple processes in distributed mode
        self.add_state("correct", default=torch.tensor(0)) # for the given attack type, our model sucessfully detected
        self.add_state("total", default=torch.tensor(0)) # total number of given attack type (in validation traces)

    def update(self, preds, labels, attack_types):
        if preds.shape != labels.shape:
            preds = torch.argmax(preds, dim=1)
        assert preds.shape == labels.shape == attack_types.shape, "shape mismatch"

        # only care about preds and labels where it is the given attack type
        mask = (attack_types == self.attack_type_label)
        indices = torch.nonzero(mask).flatten()
        preds = torch.index_select(preds, 0, indices)
        labels = torch.index_select(labels, 0, indices)
        self.correct += torch.sum(preds == labels)
        self.total += torch.sum(mask)

    def compute(self):
        return self.correct.float() / self.total
