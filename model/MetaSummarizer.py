import torch.nn as nn

from model.CustomTransformer import CustomTransformerEncoderLayer
from model.PoolerHead import PoolerHead
from model.PositionalEncoding import JustPositionalEncoding


class MetaSummarizer(nn.Module):
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

    def forward(self, metas, metas_masks):
        batch_size, rnk_len, coordi_len, meta_type, meta_seq_len, hid_dim = metas.shape

        # meta prepare
        metas = metas.reshape(-1, meta_seq_len, hid_dim)
        metas_masks = metas_masks.reshape(-1, meta_seq_len)

        metas = self.pos_encoding(metas)
        if self.use_layernorm:
            metas = self.layernorm(metas)
        metas = self.dropout(metas)

        metas = self.transformer(metas.transpose(0, 1), src_key_padding_mask=metas_masks).transpose(0, 1)
        metas = self.pooler(metas)
        metas = metas.reshape(batch_size, rnk_len, coordi_len, meta_type, hid_dim)

        return metas
