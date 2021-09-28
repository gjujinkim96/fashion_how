import torch
import torch.nn as nn

from model.CustomTransformer import CustomTransformerEncoderLayer
from model.PoolerHead import PoolerHead
from model.PositionalEncoding import JustPositionalEncoding


class CoordiSummarizer(nn.Module):
    def __init__(self, setting):
        super().__init__()

        self.rnk_len = setting.num_rnk
        self.hid_dim = setting.one3_hid_dim
        self.start_end_emb = nn.Embedding(2, setting.one3_hid_dim)

        self.pos_encoding = JustPositionalEncoding(setting.one3_hid_dim, max_len=512)

        self.use_layernorm = setting.one3_coordi_sum_layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(setting.one3_hid_dim)

        self.dropout = nn.Dropout(setting.one3_dropout)

        encoder_layer = CustomTransformerEncoderLayer(setting.one3_hid_dim,
                                                      nhead=setting.one3_attn_n_head,
                                                      dim_feedforward=setting.one3_dim_feedforward,
                                                      activation=setting.one3_tr_act,
                                                      ff_mode=setting.one3_ff_mode_high_level)

        self.transformer = nn.TransformerEncoder(encoder_layer, setting.one3_tf_n_layer)

        self.pooler = PoolerHead(setting.one3_hid_dim, pooler_act=setting.one3_pooler_act)

    def make_inputs(self, dlgs, img_feats=None, metas=None):
        dlgs = self.pos_encoding(dlgs)

        dlgs = dlgs.repeat(self.rnk_len, 1, 1)

        start = self.start_end_emb(torch.zeros(dlgs.size(0), 1, dtype=torch.long, device=dlgs.device))
        end = self.start_end_emb(torch.ones(dlgs.size(0), 1, dtype=torch.long, device=dlgs.device))

        input_pieces = [start, dlgs, end]

        if img_feats is not None:
            img_feats = img_feats.transpose(0, 1)
            img_feats = img_feats.reshape(dlgs.size(0), -1, self.hid_dim)
            input_pieces.append(img_feats)

        if metas is not None:
            metas = metas.transpose(0, 1)
            metas = metas.reshape(dlgs.size(0), -1, self.hid_dim)
            input_pieces.append(metas)

        inputs = torch.cat(input_pieces, dim=1)
        return inputs

    def make_mask(self, dlgs_turn_mask, inputs_len):
        dlgs_turn_mask = dlgs_turn_mask.repeat(self.rnk_len, 1)
        front_mask = torch.zeros(dlgs_turn_mask.size(0), 1, dtype=dlgs_turn_mask.dtype, device=dlgs_turn_mask.device)
        back_mask = torch.zeros(dlgs_turn_mask.size(0), inputs_len - dlgs_turn_mask.size(1) - 1,
                                dtype=dlgs_turn_mask.dtype, device=dlgs_turn_mask.device)

        masks = torch.cat((front_mask, dlgs_turn_mask, back_mask), dim=1)
        return masks

    def forward(self, dlgs, dlgs_turn_mask, img_feats=None, metas=None):
        batch_size, turn_len, hid_dim = dlgs.shape

        inputs = self.make_inputs(dlgs, img_feats, metas)
        masks = self.make_mask(dlgs_turn_mask, inputs.size(1))

        if self.use_layernorm:
            inputs = self.layernorm(inputs)
        inputs = self.dropout(inputs)

        # B*R S HID
        output = self.transformer(inputs.transpose(0, 1), src_key_padding_mask=masks).transpose(0, 1)
        pooler_output = self.pooler(output)

        output = output.reshape(self.rnk_len, batch_size, -1, hid_dim).transpose(0, 1)
        pooler_output = pooler_output.reshape(self.rnk_len, batch_size, -1).transpose(0, 1)  # B R HID

        return output, pooler_output


