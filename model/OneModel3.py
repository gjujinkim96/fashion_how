import torch
import torch.nn as nn

from .CoordiSummarizer import CoordiSummarizer
from .DlgTurnSummarizer import DlgTurnSummarizer
from .ImgFeatExtractor import ImgFeatExtractor
from .MetaSummarizer import MetaSummarizer
from .RankByCoordi import RankByCoordi


class OneModel3(nn.Module):
    def __init__(self, setting, data_provider):
        super().__init__()

        self.dlg_turn_summarizer = DlgTurnSummarizer(setting)
        self.img_feat_extractor = ImgFeatExtractor(setting)
        self.meta_summarizer = MetaSummarizer(setting)
        self.coordi_summarizer = CoordiSummarizer(setting)
        self.rank_output = RankByCoordi(setting)

        self.multi_sample = setting.one3_multi_sample_dropout

    def forward(self, dlgs, dlgs_seq_masks, dlgs_turn_mask, img_feats, metas, metas_masks):
        # dlgs: Batch Turn Seq Dlg_Hid
        # img_feats: Batch Rank Coordi Img_Type Feat_Hid
        # metas: Batch Rank Coordi Meta_Type Seq Meta_Hid

        dlgs = self.dlg_turn_summarizer(dlgs, dlgs_seq_masks, dlgs_turn_mask)
        img_feats = self.img_feat_extractor(img_feats)
        metas = self.meta_summarizer(metas, metas_masks)

        _, output = self.coordi_summarizer(dlgs, dlgs_turn_mask, img_feats, metas)

        if self.training and self.multi_sample > 1:
            outputs = []
            for i in range(self.multi_sample):
                outputs.append(self.rank_output(output))
            output = torch.cat(outputs, dim=0)
        else:
            output = self.rank_output(output)
        return output

    def output_from_batch(self, batch, criterion=None, is_train=True):
        rnk = batch[-1]
        batch = batch[:-1]
        logits = self(*batch)

        if criterion is not None:
            if is_train and self.multi_sample > 1:
                rnk = rnk.repeat(self.multi_sample)
            loss = criterion(logits, rnk)
            return logits, rnk, loss
        else:
            return logits, rnk

    def pred_from_output(self, logits, is_train=True):
        logits = logits.detach().cpu()
        pred = torch.argmax(logits, -1)
        return pred
