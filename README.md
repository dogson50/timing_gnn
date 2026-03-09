# 生成数据集

进入目录 './src'，执行（默认参数见 options.py）：

    python build_dataset.py \
      --src_lib ../data/lib/Nangate45 \
      --tgt_lib ../data/lib/ASAP7 \
      --src_spi ../data/spi/Nangate45 \
      --tgt_sp  ../data/spi/ASAP7 \
      --out_dir ../output

数据集会保存到 PATH_OUT，包含 dataset.pkl、scaler_stats.json、y_scaler.json 以及各类 CSV 切分文件。

关键建数参数（完整列表见 options.py）：

    --tgt_split_ratios 0.14 0.14 0.72
    --split_seed 42
    --dataset_pkl_name dataset.pkl

## 训练（HGAT + MLP 回归）
进入目录 './src'，执行：

    python train.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

本周推荐基线（HGAT v2 readout）：

        python train.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl \
            --hgat_layers 2 --hgat_dropout 0.1 --hgat_use_net_readout

做什么：目标域上的基线 cell delay 回归，HGAT 作为设计/结构嵌入，MLP 做回归，可选冻结 HGAT。
注意：data_save_path 默认是 ../output（与 build_dataset 的 out_dir 相同）。

关键训练参数（完整列表见 options.py）：

    --design_dim 64    # HGAT design embedding dim
    --hgat_hid   64    # HGAT hidden dim
    --hgat_heads 1     # HGAT attention heads
    --hgat_layers 2
    --hgat_dropout 0.1
    --hgat_use_net_readout
    --hgat_type_attn_readout
    --batch_size 1350
    --learning_rate 1e-3

查看全部超参说明：

    python train.py --help

## 训练（HGAT + MLP，预训练 + 微调）
进入目录 './src'，执行：

    python train_base_pt_ft.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

本周推荐基线（HGAT v2 readout）：

        python train_base_pt_ft.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl \
            --hgat_layers 2 --hgat_dropout 0.1 --hgat_use_net_readout

做什么：若有源域数据，先在源域预训练，再在目标域微调以提升迁移效果。
说明：

- Pretrain uses src_df if present; finetune uses tgt_train_df.
- Use --pretrain_epochs to control pretrain length (0 means half of num_epoch).

Key options (see options.py for full list):

    --pretrain_epochs 0
    --num_epoch 1000
    --design_dim 64
    --hgat_hid 64
    --hgat_heads 1

## 训练（HGAT + MLP，45nm/7nm 平衡采样）
进入目录 './src'，执行：

    python train_balanced_sampling.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

本周推荐基线（HGAT v2 readout）：

        python train_balanced_sampling.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl \
            --hgat_layers 2 --hgat_dropout 0.1 --hgat_use_net_readout

做什么：交替采样源/目标批次以平衡 45nm/7nm（或 src/tgt），缓解域偏移。
说明：

- 7nm uses tgt_train_df; 45nm uses src_df (if present).
- Balanced sampling uses --sample_45_num and --loss_weight_45.

Key options (see options.py for full list):

    --sample_45_num 1
    --loss_weight_45 1.0

## 训练（HGAT + 双 MLP 头，src/tgt 平衡采样）
进入目录 './src'，执行：

    python train_balanced_sampling_sep_mlp.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

本周主线基线（concat + NET readout）：

        python train_balanced_sampling_sep_mlp.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl \
            --hgat_layers 2 --hgat_dropout 0.1 --hgat_use_net_readout \
            --use_arc_cond --arc_sep_domain_emb --arc_vocab_scope tgt --arc_cond_mode concat

做什么：共享 HGAT 编码器，同时为源/目标域使用独立 MLP 头；并进行跨域平衡采样。
说明：

- Separate MLP heads for target and source domains (tgt/src).
- Target uses tgt_train_df; source uses src_df (fallback to tgt_train_df if src_df missing).
- Balanced sampling uses --sample_45_num and --loss_weight_45.

Key options (see options.py for full list):

    --in_dim 20
    --design_dim 64
    --hgat_hid 64
    --hgat_heads 1
    --hgat_layers 2
    --hgat_dropout 0.1
    --hgat_use_net_readout
    --sample_45_num 1
    --loss_weight_45 1.0

Arc-condition options (for sep_mlp baseline):

    --use_arc_cond
    --arc_sep_domain_emb
    --arc_vocab_scope tgt
    --arc_cond_mode concat

## 训练（HGAT + 双 MLP 头，按 cell_type 解耦）
进入目录 './src'，执行：

    python train_balanced_sampling_sep_mlp_disentangle_cell_type.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

推荐带 HGAT v2 readout 参数：

        python train_balanced_sampling_sep_mlp_disentangle_cell_type.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl \
            --hgat_layers 2 --hgat_dropout 0.1 --hgat_use_net_readout

做什么：加入基于 cell_type 的解耦目标，鼓励类型相关因素分离。
说明：

- 仍然使用与 train_balanced_sampling_sep_mlp.py 相同的平衡采样。
- 目标是减少 cell_type 信息泄漏到“域不变”表示中。

## 训练（HGAT + 双 MLP 头，按 domain 解耦）
进入目录 './src'，执行：

    python train_balanced_sampling_sep_mlp_disentangle_domain.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

推荐带 HGAT v2 readout 参数：

        python train_balanced_sampling_sep_mlp_disentangle_domain.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl \
            --hgat_layers 2 --hgat_dropout 0.1 --hgat_use_net_readout

## 训练（HGAT + 共享主干 + 校正头，src/tgt 平衡采样）
进入目录 './src'，执行：

        python train_balanced_sampling_sep_mlp_shared_calib.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

做什么：共享主干回归器并叠加域校正分支，在保持跨域平衡采样的同时减少双头完全分离带来的参数冗余。
说明：

- 训练采样策略与 train_balanced_sampling_sep_mlp.py 一致。
- HGAT 编码器口径与本周基线一致。

推荐带 HGAT v2 readout 参数：

        python train_balanced_sampling_sep_mlp_shared_calib.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl \
            --hgat_layers 2 --hgat_dropout 0.1 --hgat_use_net_readout

做什么：加入基于源/目标域的解耦目标，鼓励域不变因素分离。
说明：

- 仍然使用与 train_balanced_sampling_sep_mlp.py 相同的平衡采样。
- 目标是降低共享表示中的域特定偏差。

## 脚本对比（速览）
| 脚本 | 用途 | 额外目标 |
|---|---|---|
| train.py | 目标域基线 HGAT + MLP 回归 | 无 |
| train_base_pt_ft.py | 源域预训练 → 目标域微调 | 无 |
| train_balanced_sampling.py | 单头 + 源/目标平衡采样 | 可选源域加权损失 |
| train_balanced_sampling_sep_mlp.py | 双头 + 源/目标平衡采样 | 无 |
| train_balanced_sampling_sep_mlp_shared_calib.py | 共享主干 + 校正头 + 源/目标平衡采样 | 域校正分支 |
| train_balanced_sampling_sep_mlp_disentangle_cell_type.py | 双头 + 按 cell_type 解耦 | 解耦损失（cell_type） |
| train_balanced_sampling_sep_mlp_disentangle_domain.py | 双头 + 按 domain 解耦 | 解耦损失（domain） |
