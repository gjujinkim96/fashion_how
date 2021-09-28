import torch
import torch.nn as nn

from model.CustomTransformer import CustomTransformerEncoderLayer
from model.PoolerHead import PoolerHead
from model.PositionalEncoding import JustPositionalEncoding


class DlgTurnSummarizer(nn.Module):
    def __init__(self, setting):
        super().__init__()

        self.pos_encoding = JustPositionalEncoding(setting.one3_hid_dim, max_len=512)

        self.use_layernorm = setting.one3_layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(setting.one3_hid_dim)

        self.dropout = nn.Dropout(setting.one3_dropout)

        encoder_layer = CustomTransformerEncoderLayer(setting.one3_hid_dim,
                                                      nhead=setting.one3_attn_n_head,
                                                      dim_feedforward=setting.one3_dim_feedforward,
                                                      activation=setting.one3_tr_act,
                                                      ff_mode=setting.one3_ff_mode_low_level)

        self.transformer = nn.TransformerEncoder(encoder_layer, setting.one3_tf_n_layer)
        self.pooler = PoolerHead(setting.one3_hid_dim, pooler_act=setting.one3_pooler_act)

    def forward(self, dlgs, dlgs_seq_masks, dlgs_turn_mask):
        batch_size, turn_len, dlgs_seq_len, hid_dim = dlgs.shape

        # dlg prepare
        # pytorch 1.8에서 batch_first 없음
        dlgs_seq_masks = dlgs_seq_masks.reshape(-1, dlgs_seq_len)
        dlgs_turn_mask = dlgs_turn_mask.reshape(-1)

        dlgs = dlgs.reshape(-1, dlgs_seq_len, hid_dim)
        dlgs = self.pos_encoding(dlgs)

        if self.use_layernorm:
            dlgs = self.layernorm(dlgs)
        dlgs = self.dropout(dlgs)

        dlgs = self.transformer(dlgs.transpose(0, 1), src_key_padding_mask=dlgs_seq_masks).transpose(0, 1)
        dlgs = dlgs.masked_fill(dlgs_turn_mask[:, None, None], 0)
        dlgs = self.pooler(dlgs)
        dlgs = dlgs.reshape(batch_size, turn_len, hid_dim)

        return dlgs
