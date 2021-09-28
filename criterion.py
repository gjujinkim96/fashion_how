import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import weightedtau
from itertools import permutations
import numpy as np


class WeightedByTauLoss(nn.Module):
    def __init__(self, num_rnk, use_tau):
        super(WeightedByTauLoss, self).__init__()

        self.use_tau = use_tau

        y = np.array(list(permutations(range(num_rnk), num_rnk)))
        tot = []
        for i in y:
            tmp = [weightedtau(2 - i, 2 - j)[0] for j in y]
            tot.append(tmp)
        tot = torch.tensor(tot).float()
        self.tau_weight = tot
        if not self.use_tau:
            self.tau_weight = (self.tau_weight + 1) / 2
        self.dim_size = len(y)
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input, target):
        # dim_size 곱해야 cross entropy랑 같아짐)
        self.tau_weight = self.tau_weight.to(input.device)
        loss = self.criterion(F.log_softmax(input, dim=-1), self.tau_weight[target])

        return loss


class CE_WBT(nn.Module):
    def __init__(self, num_rnk, use_tau):
        super(CE_WBT, self).__init__()

        self.criterion1 = WeightedByTauLoss(num_rnk, use_tau)
        self.criterion2 = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion1(input, target) + self.criterion2(input, target)


def get_criterion(setting):
    if setting.criterion == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif setting.criterion == 'WeightedByTauLoss':
        return WeightedByTauLoss(setting.num_rnk, setting.use_tau)
    elif setting.criterion == 'CE_WBT':
        return CE_WBT(setting.num_rnk, setting.use_tau)
    else:
        raise NotImplementedError()
