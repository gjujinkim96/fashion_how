import numpy as np
import os
import torch
import math
from torch.utils.data import Dataset

from custom_file_io import shuffle_coordi_and_ranking_for_one, Shuffler
from file_io import *
from .padding_utils import *


class OneDataset(Dataset):
    def __init__(self, data_dialog, data_coordi, data_rank, data_provider, mode, dialogs=None, show=True):
        super().__init__()
        setting = data_provider.setting
        meta_holder = data_provider.meta_holder
        swer = data_provider.swer

        self.dialogs = dialogs

        if self.dialogs is None:
            vec_dialog = vectorize(data_dialog, swer, return_words=True)
            self.dlg = vec_dialog
        else:
            self.dialogs = vectorize(self.dialogs, swer, return_words=True)
            self.dlg = data_dialog

        idx_coordi = indexing_coordi(data_coordi, setting.coordi_size, meta_holder.item2idx, show=show)

        zero_pad_idx = swer._subw_dic['ZERO_PAD']
        self.zero_pad_emb = torch.tensor(swer._emb_np[zero_pad_idx])
        self.rnk = torch.tensor(data_rank, dtype=torch.long)
        self.coordi = idx_coordi
        self.coordi_size = setting.coordi_size
        self.metadata = meta_holder.metadata
        self.img_feats = meta_holder.feats

        self.num_rnk = 3 if setting.mode != 'train' else setting.num_rnk
        self.is_test = mode in ['test', 'test_only']
        self.perm_from_dataset = True if self.is_test else setting.perm_from_dataset
        self.perm_iteration = math.factorial(self.num_rnk) if self.is_test else setting.permutation_iteration
        self.perm_random = False if self.is_test else setting.perm_random

        if self.perm_from_dataset:
            self.shuffler = Shuffler(self.num_rnk)

    def __len__(self):
        mult = self.perm_iteration if self.perm_from_dataset else 1
        return len(self.dlg) * mult

    def __getitem__(self, idx):
        if self.perm_from_dataset:
            item = idx // self.perm_iteration
            perm_idx = idx % self.perm_iteration

            if self.perm_random:
                perm_idx = np.random.randint(self.shuffler.perm_len)
        else:
            item = idx

        if self.dialogs is None:
            dlg = self.dlg[item]
        else:
            dlg = self.dialogs[self.dlg[item]]
        rnk = self.rnk[item]

        cur_coordis = self.coordi[item]

        img_feat = []
        for i in range(self.num_rnk):
            cur_coordi = cur_coordis[i]
            vec_coordi = []
            for j in range(self.coordi_size):
                img_idx = cur_coordi[j]
                vec_coordi.append(self.img_feats[j][img_idx])
            img_feat.append(vec_coordi)
        img_feat = torch.tensor(img_feat)

        meta = []
        for i in range(self.num_rnk):
            cur_coordi = cur_coordis[i]
            vec_meta = []
            for j in range(self.coordi_size):
                img_idx = cur_coordi[j]
                vec_meta.append(self.metadata[j][img_idx])
            meta.append(vec_meta)
        meta = np.array(meta, dtype='object')

        if self.is_test:
            img_feat, meta, rnk = shuffle_coordi_and_ranking_for_one(img_feat, meta, self.num_rnk)
        elif self.perm_from_dataset:
            img_feat, meta, rnk = self.shuffler.change_by_add(perm_idx, img_feat, meta, rnk)

        return dlg, img_feat, meta, rnk

    def collate_fn(self, batch):
        dlgs = [b[0] for b in batch]
        dlgs, dlgs_seq_masks, dlgs_turn_mask = pad_turns(dlgs, self.zero_pad_emb)

        img_feats = torch.stack([b[1] for b in batch])
        # rnk coordi meta_type seq dim
        metas = [b[2] for b in batch]
        metas, metas_masks = pad_metas(metas, self.zero_pad_emb)
        rnks = torch.tensor([b[3] for b in batch])

        return dlgs, dlgs_seq_masks, dlgs_turn_mask, img_feats.float(), metas.float(), metas_masks, rnks
