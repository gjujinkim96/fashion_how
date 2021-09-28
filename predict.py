import os

os.system('cd /home/work/model')

import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == '__main__':
    libs_to_install = ['transformers', 'madgrad', 'wandb', 'cached_property']
    for lib in libs_to_install:
        print(f'########## Downloading {lib}')
        install(lib)

    from setting import Setting
    import main

    setting = Setting(
        mode='pred',  # train, test, pred
        ## 수정 필요
        in_file_trn_dialog='/home/work/data/ddata.wst.txt',
        in_file_tst_dialog='/home/work/data/ac_eval_t1.wst.dev',
        in_file_fashion='/home/work/data/mdata.wst.txt',
        in_file_img_feats='/home/work/data/extracted_feat.json',
        subWordEmb_path='/home/work/model/sstm_v0p5_deploy/sstm_v4p49_np_final_n36134_d128_r_eng_upper.dat',
        pred_csv_save_name='/home/work/model/prediction.csv',
        # 수정 필요

        coordi_size=4,
        meta_size=4,
        img_feats_size=4096,
        img_feat_type=3,
        num_rnk=3,

        model_name='OneModel3',
        model_save_dir='/home/work/model/one3',   # <--- 수정 필요
        model_file='OneModel3-best.pt',
        save_freq=100,
        save_best=False,

        dataset='OneDataset',
        perm_from_dataset=False,
        perm_random=False,

        learning_rate=0.0001,
        max_grad_norm=0.5,
        batch_size=8,
        epochs=10,
        seed=42,

        small_data=False,
        use_amp=True,
        cuda_pref=True,
        num_workers=0,

        corr_thres=0.7,
        lower_thres=0.6,
        adjust_upward=False,
        adjust_downward=True,
        rank_mode='use_all',

        permutation_iteration=1,
        num_augmentation=5,
        use_multimodal=True,

        optimizer='AdamW',
        momentum=0.0,
        weight_decay=0.001,
        amsgrad=False,

        lr_scheduler='get_cosine_schedule_with_warmup',  # get_constant_schedule, get_cosine_schedule_with_warmup
        warmup_ratio=0.2,
        warmup_steps=None,
        step_size_n_epoch=1000,
        step_size_gamma=0.5,
        num_cycles=1,

        criterion='WeightedByTauLoss',  # WeightedByTauLoss, CrossEntropyLoss
        use_tau=True,

        should_log=False,
        project='fashion',
        entity='usr_name',
        name='one3_run',

        one3_hid_dim=128,
        one3_dim_feedforward=256,
        one3_attn_n_head=8,
        one3_tf_n_layer=6,

        one3_dropout=0.1,
        one3_layernorm=True,
        one3_pooler_act='Tanh',
        one3_tr_act='relu',
        one3_coordi_sum_layernorm=True,
        one3_output_dropout=0.3,
        one3_output_layernorm=False,

        one3_multi_sample_dropout=4,

        one3_ff_mode_low_level='linear',
        one3_ff_mode_high_level='linear',

        use_swa=False,
        swa_start=5,
        swa_lr=0.05,

        filter_by_short=False,

        load_from_dialog=None,
    )

    main.main(setting)
