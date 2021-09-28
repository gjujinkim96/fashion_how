from dataclasses import dataclass
from typing import Optional


@dataclass
class Setting:
    mode: str  # train, test, pred, coordi_train, coordi_pred, make_train

    # file path related
    in_file_trn_dialog: str
    in_file_tst_dialog: str
    in_file_fashion: str
    in_file_img_feats: str
    subWordEmb_path: str
    pred_csv_save_name: str

    # data constant
    coordi_size: int
    meta_size: int
    img_feats_size: int
    img_feat_type: int
    num_rnk: int

    # model save related
    model_name: str
    model_save_dir: str
    model_file: Optional[str]
    save_freq: int
    save_best: bool

    # dataset
    dataset: str
    perm_from_dataset: bool
    perm_random: bool

    # general model setting
    learning_rate: float
    max_grad_norm: float
    batch_size: int
    epochs: int
    seed: int

    # general data setting
    small_data: bool
    use_amp: bool
    cuda_pref: bool
    num_workers: int

    corr_thres: float
    lower_thres: float
    adjust_upward: bool
    adjust_downward: bool
    rank_mode: str  # only for old ranking # use_all, use_ready_made_only, use_append_only

    permutation_iteration: int
    num_augmentation: int
    use_multimodal: bool

    # optimizer
    optimizer: str
    momentum: float
    weight_decay: float
    amsgrad: bool

    # lr scheduler
    lr_scheduler: str
    warmup_ratio: float
    warmup_steps: Optional[int]
    step_size_n_epoch: int
    step_size_gamma: float
    num_cycles: int

    # criterion
    criterion: str
    use_tau: bool

    # wandb
    should_log: bool
    project: str
    entity: str
    name: str

    # one model 3 stuff
    one3_hid_dim: int
    one3_dim_feedforward: int
    one3_attn_n_head: int
    one3_tf_n_layer: int

    one3_dropout: float
    one3_layernorm: bool
    one3_pooler_act: str  # Tanh, ReLU, none
    one3_tr_act: str  # relu, gelu
    one3_coordi_sum_layernorm: bool
    one3_output_dropout: float
    one3_output_layernorm: bool

    one3_multi_sample_dropout: int

    one3_ff_mode_low_level: str
    one3_ff_mode_high_level: str

    use_swa: bool
    swa_start: int
    swa_lr: float

    filter_by_short: bool

    load_from_dialog: Optional[str]