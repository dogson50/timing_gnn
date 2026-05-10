#!/bin/bash
# Exp-B: half-shared architecture (src/tgt separate backbone inside shared_calib.py)
set -euo pipefail

cd "$(dirname "$0")/../.."

python -u src/train_balanced_sampling_sep_mlp_shared_calib.py \
  --model_saving_dir model_sep_hgat_expB_half_share \
  --data_save_path output \
  --dataset_pkl_name dataset.pkl \
  --gpu 0 \
  --seed 9294 \
  --freeze_hgat \
  --dedup_z \
  --num_epoch 260 \
  --batch_size 3072 \
  --learning_rate 3e-3 \
  --enc_lr_scale 0.05 \
  --weight_decay 2e-5 \
  --mlp_dropout 0.15 \
  --hgat_l2_norm \
  --z_noise_std 0.01 \
  --loss_weight_45 1.0 \
  --src_loss_anneal_start 80 \
  --src_loss_anneal_end 280 \
  --src_loss_final_scale 0.8
