import torch.nn as nn


class PoolerHead(nn.Module):
    def __init__(self, hid_dim, output_dim=None, pooler_act='Tanh'):
        super().__init__()

        if output_dim is None:
            output_dim = hid_dim

        self.dense = nn.Linear(hid_dim, output_dim)
        self.has_pooler = True

        if pooler_act == 'Tanh':
            self.act = nn.Tanh()
        elif pooler_act == 'ReLU':
            self.act = nn.ReLU()
        elif pooler_act == 'GELU':
            self.act = nn.GELU()
        elif pooler_act == 'none':
            self.has_pooler = False
        else:
            raise ValueError()

    def forward(self, x, extract=True, avg=False):
        if extract:
            if avg:
                x = x.mean(dim=1)
            else:
                x = x[:, 0]
        x = self.dense(x)

        if self.has_pooler:
            x = self.act(x)
        return x
