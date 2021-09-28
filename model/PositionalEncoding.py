import math

import torch
import torch.nn as nn

class JustPositionalEncoding(nn.Module):
    def __init__(self, hid_dim, max_len: int = 1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2) * (-math.log(10000.0) / hid_dim))
        pe = torch.zeros(1, max_len, hid_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]

