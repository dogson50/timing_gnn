#!/bin/bash
#SBATCH --job-name=TimingPred
#SBATCH --mail-user=xyzhang21@link.cuhk.edu.hk
#SBATCH --output=./DAC23-reproduce-ep200-bs1350.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=proj200

cd ./src
python -u train.py --model_saving_dir ../output/reproduce-dac23-ep200-bs1350 \
--data_save_path ../datasets/asap7-designs-repro \
--num_epoch 200 \
--batch_size 1350