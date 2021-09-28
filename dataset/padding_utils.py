import numpy as np
import torch


def pad_seq(seq, max_seq, pad_emb):
    seq = np.array(seq, dtype=np.float32)
    ret = torch.tensor(seq)
    pad_mask = torch.zeros(len(ret), dtype=torch.bool)
    pad_len = max_seq - len(ret)
    if pad_len > 0:
        pad_part = pad_emb.repeat(pad_len, 1)
        ret = torch.cat((ret, pad_part))

        mask_pad_part = torch.ones(pad_len, dtype=torch.bool)
        pad_mask = torch.cat((pad_mask, mask_pad_part))

    trunc_len = len(ret) - max_seq
    if trunc_len > 0:
        ret = ret[:max_seq]
        pad_mask = pad_mask[:max_seq]

    return ret, pad_mask


def pad_turn(turn, max_turn, max_seq, pad_emb):
    ret = []
    for s in turn:
        ret.append(pad_seq(s, max_seq, pad_emb))
    ret, pad_masks = [torch.stack(x) for x in zip(*ret)]
    pad_turns = torch.zeros(len(turn), dtype=torch.bool)

    pad_len = max_turn - len(ret)
    if pad_len > 0:
        pad_part = pad_emb.repeat(pad_len, max_seq, 1)
        ret = torch.cat((ret, pad_part))

        # 여기는 1로 하면 nan으로 계산되서 나옴
        # 그냥 나온 결과를 turn_mask로 무시하자
        mask_pad_part = torch.zeros((pad_len, max_seq), dtype=torch.bool)
        pad_masks = torch.cat((pad_masks, mask_pad_part))

        turns_pad_part = torch.ones(pad_len, dtype=torch.bool)
        pad_turns = torch.cat((pad_turns, turns_pad_part))

    trunc_len = len(ret) - max_turn
    if trunc_len > 0:
        ret = ret[:max_turn]
        pad_turns = pad_turns[:max_turn]
    return ret, pad_masks, pad_turns


def pad_turns(turns, pad_emb):
    max_turn = min(max([len(turn) for turn in turns]), 500)
    max_seq = min(max([len(sent) for turn in turns for sent in turn]), 500)

    ret = []
    for turn in turns:
        ret.append(pad_turn(turn, max_turn, max_seq, pad_emb))
    dlgs, pad_masks, pad_turns = [torch.stack(x) for x in zip(*ret)]
    return dlgs, pad_masks, pad_turns


def pad_recursive(data, max_seq, pad_emb, level):
    if level == 0:
        data = torch.tensor(data)
        return pad_seq(data, max_seq, pad_emb)
    else:
        ret = []
        for datum in data:
            ret.append(pad_recursive(datum, max_seq, pad_emb, level - 1))
        ret_data, ret_masks = [torch.stack(x) for x in zip(*ret)]
        return ret_data, ret_masks


def max_recursive(data, level):
    if level == 0:
        return len(data)
    else:
        return max([max_recursive(x, level - 1) for x in data])


def pad_metas(metas, pad_emb):
    max_seq = min(max_recursive(metas, 4), 500)
    return pad_recursive(metas, max_seq, pad_emb, 4)
