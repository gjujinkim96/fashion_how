import torch.nn as nn
from model.PoolerHead import PoolerHead


class ImgFeatExtractor(nn.Module):
    def __init__(self, setting):
        super().__init__()

        self.use_layernorm = setting.one3_layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(setting.img_feats_size)
        self.dropout = nn.Dropout(setting.one3_dropout)
        self.pooler = PoolerHead(setting.img_feats_size, output_dim=setting.one3_hid_dim,
                                 pooler_act=setting.one3_pooler_act)

    def forward(self, feat):
        if self.use_layernorm:
            feat = self.layernorm(feat)
        feat = self.dropout(feat)

        feat = self.pooler(feat, extract=False)
        return feat

