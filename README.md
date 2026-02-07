# Generating dataset

Go to directory './src', and execute (defaults are defined in options.py):

    python build_dataset.py \
      --src_lib ../data/lib/Nangate45 \
      --tgt_lib ../data/lib/ASAP7 \
      --src_spi ../data/spi/Nangate45 \
      --tgt_sp  ../data/spi/ASAP7 \
      --out_dir ../output

The dataset will be saved under PATH_OUT, including dataset.pkl, scaler_stats.json, y_scaler.json and CSV splits.

Key dataset options (see options.py for full list):

    --tgt_split_ratios 0.14 0.14 0.72
    --split_seed 42
    --dataset_pkl_name dataset.pkl

## Training (HGAT + MLP regression)
Go to directory './src', and execute:

    python train.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

Note: default data_save_path is ../output (same as build_dataset out_dir).

Key training options (see options.py for full list):

    --design_dim 64    # HGAT design embedding dim
    --hgat_hid   64    # HGAT hidden dim
    --hgat_heads 1     # HGAT attention heads
    --batch_size 1350
    --learning_rate 1e-3

For hyper-parameters, run:

    python train.py --help

## Training (HGAT + MLP, pretrain + finetune)
Go to directory './src', and execute:

    python train_base_pt_ft.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

Notes:

- Pretrain uses src_df if present; finetune uses tgt_train_df.
- Use --pretrain_epochs to control pretrain length (0 means half of num_epoch).

Key options (see options.py for full list):

    --pretrain_epochs 0
    --num_epoch 1000
    --design_dim 64
    --hgat_hid 64
    --hgat_heads 1

## Training (HGAT + MLP, balanced sampling 45nm/7nm)
Go to directory './src', and execute:

    python train_balanced_sampling.py --model_saving_dir PATH_MODEL --data_save_path PATH_OUT --dataset_pkl_name dataset.pkl

Notes:

- 7nm uses tgt_train_df; 45nm uses src_df (if present).
- Balanced sampling uses --sample_45_num and --loss_weight_45.

Key options (see options.py for full list):

    --sample_45_num 1
    --loss_weight_45 1.0
