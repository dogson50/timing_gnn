#!/bin/bash
#SBATCH --job-name=TimingPred
#SBATCH --mail-user=xyzhang21@link.cuhk.edu.hk
#SBATCH --output=./log/dag/reproduce-no-retain-sep-bayesian-residual-disentangle-learn-alpha-weightkl-200-klstart0-sampling-ema001-wclr100-wcmd01-temp02-130-sample1-weight1000-seed6666-lr2e-4-ep200-bs2048-jpeg,link,spi,usbf.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=proj[73-79,193]

cd ./src
python -u train_balanced_sampling_sep_mlp_disentangle_bayesian_res.py --model_saving_dir ../output/reproduce/no-retain-sep-bayesian-residual-disentangle-learn-alpha-weightkl-200-klstart0-sampling-ema001-wclr100-wcmd01-temp02-130-sample1-weight1-seed6666-lr1e-4-ep200-bs2048-jpeg,link,spi,usbf \
--data_save_path ../datasets/130-designs \
--num_epoch 200 \
--batch_size 2048 \
--seed 6666 \
--cell_feat_dim 137 \
--learning_rate 2e-4 \
--training_set jpeg,linkruncca,spiMaster,usbf_device \
--test_set arm9,chacha,hwacha,or1200,sha3 \
--sample_130_num 1 \
--loss_weight_130 1000. \
--weight_kl 200.0 \
--kl \
--sampling \
--con_temp 0.2 \
--weight_clr 100.0 \
--weight_cmd 10.0 \
--norm_clr \
--kl_start_ep 0 \
--not_retain_graph
