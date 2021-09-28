from file_io import *
import torch

def categorize_for_one(name, vec_item, coordi_size):
    """
    function: categorize fashion items
    """
    slot_item = []
    slot_name = []
    for i in range(coordi_size):
        slot_item.append([])
        slot_name.append([])
    for i in range(len(name)):
        pos = position_of_fashion_item(name[i])
        slot_item[pos].append(vec_item[i])
        slot_name[pos].append(name[i])
    return slot_name, slot_item


def load_fashion_feature_for_one(file_name, slot_name, coordi_size, feat_size):
    """
    function: load image features
    """
    with open(file_name, 'r') as fin:
        data = json.load(fin)
        suffix = '.jpg'

        feats = []
        for i in range(coordi_size):
            feat = []
            for n in slot_name[i]:
                if n[0:4] == 'NONE':
                    feat.append(np.zeros((3, feat_size)))
                else:
                    img_name = n + suffix
                    feat.append(np.stack(data[img_name]))
            feats.append(np.stack(feat))
        feats = np.array(feats, dtype=object)
        return feats


def make_metadata_for_one(in_file_fashion, swer, coordi_size, meta_size,
                          use_multimodal, in_file_img_feats, feat_size):
    """
    function: make metadata for training and test
    """
    print('\n<Make metadata>')
    if not os.path.exists(in_file_fashion):
        raise ValueError('{} do not exists.'.format(in_file_fashion))

    # load metadata DB
    name, data_item = load_fashion_item(in_file_fashion, coordi_size, meta_size)
    print('vectorizing data')
    emb_size = swer.get_emb_size()

    # for computing vec_similarities
    vec_item = vectorize_dlg(swer, data_item, return_words=False)
    vec_item = vec_item.reshape((-1, meta_size * emb_size))
    slot_name, slot_item = categorize(name, vec_item, coordi_size)
    vec_similarities = []
    # calculation cosine similarities
    for i in range(coordi_size):
        item_sparse = sparse.csr_matrix(slot_item[i])
        similarities = cosine_similarity(item_sparse)
        vec_similarities.append(similarities)
    vec_similarities = np.array(vec_similarities, dtype=object)

    # embedding
    vec_item = vectorize_dlg(swer, data_item, return_words=True)
    vec_item = [vec_item[4 * i:4 * (i + 1)] for i in range(len(vec_item) // 4)]

    # categorize fashion items
    slot_name, slot_item = categorize_for_one(name, vec_item, coordi_size)
    slot_feat = None
    if use_multimodal:
        slot_feat = load_fashion_feature_for_one(in_file_img_feats,
                                                 slot_name, coordi_size, feat_size)

    idx2item = []
    item2idx = []
    item_size = []
    for i in range(coordi_size):
        idx2item.append(dict((j, m) for j, m in enumerate(slot_name[i])))
        item2idx.append(dict((m, j) for j, m in enumerate(slot_name[i])))
        item_size.append(len(slot_name[i]))
    return slot_item, idx2item, item2idx, item_size, \
           vec_similarities, slot_feat


def make_metadata_from_setting(setting, swer):
    return make_metadata_for_one(setting.in_file_fashion, swer,
                                 setting.coordi_size, setting.meta_size,
                                 setting.use_multimodal, setting.in_file_img_feats,
                                 setting.img_feats_size)


def make_raw_io_data_no_ranking(in_file_dialog, setting):
    dialog, coordi, reward, delim_dlg, delim_crd, delim_rwd = \
        load_trn_dialog(in_file_dialog, setting.filter_by_short)
    # per episode
    dialog = episode_slice(dialog, delim_dlg)
    coordi = episode_slice(coordi, delim_crd)
    reward = episode_slice(reward, delim_rwd)

    if setting.small_data:
        dialog = dialog[:10]
        coordi = coordi[:10]
        reward = reward[:10]

    return dialog, coordi, reward

def make_raw_io_data(setting, meta_holder, mode):
    if mode == 'train' and setting.load_from_dialog is not None:
        with open(setting.load_from_dialog, 'rb') as f:
            saved = pickle.load(f)

        data_dialog = saved['dialog']
        data_dialog_idx = saved['dialog_index']
        data_coordi = saved['coordi']
        data_rank = [0] * len(data_dialog_idx)

        idx = np.arange(setting.num_rnk)
        rank_lst = np.array(list(permutations(idx, setting.num_rnk)))
        new_coordi = []
        new_rank = []
        for crd in data_coordi:
            rank, rand_crd = shuffle_one_coordi_and_ranking(rank_lst, crd, setting.num_rnk)
            new_coordi.append(rand_crd)
            new_rank.append(rank)

        return data_dialog, new_coordi, new_rank, data_dialog_idx

    in_file_dialog = setting.in_file_trn_dialog if mode == 'train' else setting.in_file_tst_dialog

    print('\n<Make raw input & output data>')
    if not os.path.exists(in_file_dialog):
        raise ValueError('{} do not exists.'.format(in_file_dialog))

    if mode == 'train':
        dialog, coordi, reward = make_raw_io_data_no_ranking(in_file_dialog, setting)

        # prepare DB for evaluation
        if setting.perm_from_dataset:
            print(f'using 1 for permutation_iteration instead of {setting.permutation_iteration}')
            print('because perm_from_from_dataset is on')
            perm_iteration = 1
        else:
            perm_iteration = setting.permutation_iteration

        data_dialog, data_coordi, data_rank, data_dialog_idx = \
            make_ranking_examples(dialog, coordi, reward,
                                  meta_holder.item2idx, meta_holder.idx2item, meta_holder.similarities,
                                  setting.num_rnk, perm_iteration, setting.num_augmentation, setting.corr_thres,
                                  setting.lower_thres, setting.adjust_upward, setting.adjust_downward,
                                  rank_mode=setting.rank_mode, filter_by_short=setting.filter_by_short)
    else:
        # load test dialog DB
        data_dialog, data_coordi, data_rank = \
            load_eval_dialog(in_file_dialog, setting.num_rnk)
        data_dialog_idx = list(range(len(data_dialog)))

    if setting.small_data:
        data_coordi = data_coordi[:10]
        data_rank = data_rank[:10]
        data_dialog_idx = data_dialog_idx[:10]

    data_rank = np.array(data_rank, dtype='int32')

    return data_dialog, data_coordi, data_rank, data_dialog_idx


def shuffle_coordi_and_ranking_for_one(img_feat, meta, num_rank):
    idx = np.arange(num_rank)
    np.random.shuffle(idx)
    rank_lst = np.array(list(permutations(np.arange(num_rank), num_rank)))
    for i in range(len(rank_lst)):
        if np.array_equal(idx, rank_lst[i]):
            rank = i
            break

    new_feats = img_feat[rank_lst[rank]]
    new_meta = meta[rank_lst[rank]]

    return new_feats, new_meta, rank


class Shuffler:
    def __init__(self, num_rnk):
        self.perms = torch.tensor(list(permutations(range(num_rnk), num_rnk)))
        self.rev_perm = torch.zeros(len(self.perms), num_rnk, dtype=torch.long)
        for i in range(self.rev_perm.size(0)):
            for j in range(self.rev_perm.size(1)):
                self.rev_perm[i, self.perms[i, j]] = j
        self.perm_len = len(self.perms)

    def change_rnk(self, dest, img_feat, meta, rnk):
        img_feat = img_feat[self.rev_perm[rnk]][self.perms[dest]]
        meta = meta[self.rev_perm[rnk]][self.perms[dest]]
        rnk = dest
        return img_feat, meta, rnk

    def change_by_add(self, add, img_feat, meta, rnk):
        return self.change_rnk((rnk + add) % self.perm_len, img_feat, meta, rnk)
