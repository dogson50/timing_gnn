#!/bin/bash
# Exp-A: target-only baseline (keep HGAT encoder path, disable source loss)
set -euo pipefail

cd "$(dirname "$0")/../.."

python -u src/train_balanced_sampling_sep_mlp_shared_calib_step5a_feat_opt.py \
  --model_saving_dir model_sep_hgat_expA_tgt_only \
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
  --enrich_parasitic_net_feat \
  --hgat_net_feat_mode parasitic_append \
  --hgat_par_cap_weight 0.6 \
  --loss_weight_45 0.0 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 3 \
  --plateau_threshold 3e-4 \
  --plateau_min_lr 1e-6 \
  --early_stop_patience 10 \
  --early_stop_min_delta 8e-4 \
  --num_workers 0 \
  --val_eval_interval 5 \
  --test_eval_interval 100
