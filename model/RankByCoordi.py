import math

import torch.nn as nn


class RankByCoordi(nn.Module):
    def __init__(self, setting):
        super().__init__()

        self.output_size = math.factorial(setting.num_rnk)
        self.dropout = nn.Dropout(setting.one3_output_dropout)
        self.use_layernorm = setting.one3_output_layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(setting.one3_hid_dim * setting.num_rnk)

        self.ranking = nn.Linear(setting.num_rnk * setting.one3_hid_dim, self.output_size)

    def forward(self, coordis):
        # coordis = B R HID

        coordis = coordis.reshape(coordis.size(0), -1)

        if self.use_layernorm:
            coordis = self.layernorm(coordis)
        coordis = self.dropout(coordis)
        output = self.ranking(coordis)
        return output
