import torch.nn.functional as F
from torch.nn import *


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class CustomTransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 ff_mode='linear'):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.dropout = Dropout(dropout)

        self.use_conv = ff_mode in ['conv', 'conv_linear']
        self.use_conv_linear = ff_mode == 'conv_linear'
        self.use_both = ff_mode == 'conv+linear'
        self.use_gru = ff_mode == 'gru'
        if self.use_conv:
            if self.use_conv_linear:
                self.conv1 = Conv1d(d_model, d_model, 3, padding=1)
                self.conv2 = Conv1d(d_model, dim_feedforward, 3, padding=1)
                self.conv_lin = Linear(dim_feedforward, d_model)
            else:
                self.conv1 = Conv1d(d_model, dim_feedforward, 3, padding=1)
                self.conv2 = Conv1d(dim_feedforward, d_model, 3, padding=1)
        elif self.use_both:
            self.linear1 = Linear(d_model, dim_feedforward)
            self.linear2 = Linear(dim_feedforward, d_model)
            self.conv1 = Conv1d(d_model, dim_feedforward, 3, padding=1)
            self.conv2 = Conv1d(dim_feedforward, d_model, 3, padding=1)
        elif self.use_gru:
            self.gru1 = GRU(d_model, d_model // 2, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.linear1 = Linear(d_model, dim_feedforward)
            self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.use_conv:
            src2 = self.conv2(self.dropout(self.activation(self.conv1(src.transpose(1, 2))))).transpose(1, 2)
            if self.use_conv_linear:
                src2 = self.conv_lin(self.dropout(self.activation(src2)))
        elif self.use_both:
            src2 = self.conv2(self.dropout(self.activation(self.conv1(src.transpose(1, 2))))).transpose(1, 2)
            src2 = src2 + self.linear2(self.dropout(self.activation(self.linear1(src))))
        elif self.use_gru:
            src2, _ = self.gru1(src)
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
