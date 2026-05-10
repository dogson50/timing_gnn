#!/bin/bash
#SBATCH --job-name=TimingPred
#SBATCH --mail-user=xyzhang21@link.cuhk.edu.hk
#SBATCH --output=./log/reproduce-base-130-ft-mlp-gnn-cell-seed6666-lr-1e-4-ep200-bs-2048-pt-jpeg,link,spi,usbf-seed-1357.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=proj[73-79,193]

cd ./src
python -u train_base_pt_ft.py --model_saving_dir ../output/reproduce/base-130-ft-mlp-gnn-cell-seed6666-lr-1e-4-ep200-bs-2048-pt-jpeg,link,spi,usbf-seed-1357 \
--data_save_path ../datasets/130-designs \
--num_epoch 50 \
--batch_size 2048 \
--seed 6666 \
--cell_feat_dim 137 \
--learning_rate 1e-4 \
--training_set smallboom \
--test_set arm9,chacha,hwacha,or1200,sha3 \
--train_node 7 \
--test_node 7 \
--load_ckpt_path ../output/pretrain \
--ft_mode mlp_gnn_cell
#--time_unit_trans 7_to_130
