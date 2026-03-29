import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help='the learning rate for training. Type: float.',
                        default=1e-3)
    parser.add_argument("--enc_lr_scale", type=float, default=0.2,
                        help='encoder LR scale w.r.t. learning_rate when encoder is trainable')
    parser.add_argument("--batch_size", type=int, help='the number of samples in each training batch. Type: int',
                        default=1350)
    parser.add_argument("--num_epoch", type=int,
                        help='Type: int; number of epoches that the training procedure runs. Type: int', default=500)
    # HGAT/MLP regression settings
    parser.add_argument("--in_dim", type=int,
                        help='input feature dimension (must equal len(NUMERIC_COLS)=20).', default=20)
    parser.add_argument("--out_dim", type=int, help='the dimension of the output embedding. Type: int', default=64)
    parser.add_argument("--hidden_dim", type=int, help='the dimension of the intermediate MLP layers. Type: int',
                        default=256)
    parser.add_argument("--design_dim", type=int, default=64, help='HGAT design embedding dimension')
    parser.add_argument("--hgat_hid", type=int, default=64, help='HGAT hidden dimension')
    parser.add_argument("--hgat_heads", type=int, default=1, help='HGAT attention heads')
    parser.add_argument("--hgat_layers", type=int, default=2, help='HGAT message passing layers')
    parser.add_argument("--hgat_dropout", type=float, default=0.1, help='HGAT dropout rate')
    parser.add_argument("--hgat_use_net_readout", action="store_true",
                        help="include NET node statistics in HGAT graph readout")
    parser.add_argument("--hgat_type_attn_readout", action="store_true",
                        help="use learned node-type attention when fusing HGAT type summaries")
    parser.add_argument("--hgat_l2_norm", action="store_true",
                        help="apply L2 normalization on the final HGAT embedding")
    parser.add_argument("--z_noise_std", type=float, default=0.0,
                        help="std of Gaussian noise added to HGAT z during training only")
    parser.add_argument("--use_arc_cond", action="store_true",
                        help="enable arc conditioning in MLP heads using from_pin/to_pin/pol embeddings")
    parser.add_argument("--pin_emb_dim", type=int, default=8,
                        help="embedding dim for from_pin/to_pin when --use_arc_cond")
    parser.add_argument("--pol_emb_dim", type=int, default=2,
                        help="embedding dim for polarity when --use_arc_cond")
    parser.add_argument("--sense_emb_dim", type=int, default=4,
                        help="embedding dim for timing_sense when --use_arc_cond")
    parser.add_argument("--cond_emb_dim", type=int, default=8,
                        help="embedding dim for when/sdf condition token when --use_arc_cond")
    parser.add_argument("--arc_vocab_scope", type=str, default="tgt",
                        choices=["tgt", "src_tgt"],
                        help="pin vocabulary scope for arc conditioning")
    parser.add_argument("--arc_sep_domain_emb", action="store_true",
                        help="use separate arc embeddings for target/source heads")
    parser.add_argument("--arc_cond_mode", type=str, default="concat",
                        choices=["concat", "film"],
                        help="arc conditioning fusion mode: concat or film")
    parser.add_argument("--disable_src_arc_cond", action="store_true",
                        help="disable arc conditioning on source head (target head still uses arc cond)")
    # HGAT runtime controls (shared across training scripts)
    parser.add_argument("--freeze_hgat", action="store_true",
                        help="freeze HGAT encoder and precompute z (train MLP only)")
    parser.add_argument("--enc_update_interval", type=int, default=1,
                        help="when HGAT is trainable, update encoder every N target batches (N>1 speeds up training)")
    parser.add_argument("--dedup_z", action="store_true",
                        help="deduplicate cell_type within batch when building z")
    parser.add_argument("--pretrain_epochs", type=int, default=0,
                        help="Number of pretrain epochs (0 = half of num_epoch)")
    parser.add_argument("--gcn_dropout", type=float, help='dropout rate for GNN layers. Type: float', default=0)
    parser.add_argument("--mlp_dropout", type=float, help='dropout rate for mlp. Type: float', default=0)
    parser.add_argument("--weight_decay", type=float, help='weight decay. Type: float', default=0)
    parser.add_argument("--model_saving_dir", type=str, help='the directory to save the trained model. Type: str',
                        default='../models/asap7-designs')
    parser.add_argument("--preprocess",
                        help='decide whether to run the preprocess procedure or not. If set True, then a preprocess procedure'
                             ' (generating dataset + initialize model)will be carried out; '
                             'Else a normal training procedure will be carried out.'
                             'Type: sote_true', action='store_true')
    parser.add_argument("--n_fcn", type=int, help='the number of full connected layers of the mlp. Type: int',
                        default=3)
    parser.add_argument("--alpha", type=float, help='the weight of the cost-sensitive learning. Type: float',
                        default=1.0)
    parser.add_argument("--change_lr", help='Decide to change to learning rate. Type: float', action='store_true')
    parser.add_argument("--change_alpha", help='Decide to change alpha. Type: float', action='store_true')
    parser.add_argument("--gpu", type=int, help='index of gpu. Type: int', default=0)
    # parser.add_argument('--balanced',action='store_true',help = 'decide whether to balance the training dataset (using oversampling) or not; Type: store_true')
    parser.add_argument('--nlabels', type=int, help='number of prediction classes. Type: int', default=1)
    parser.add_argument('--os_rate', help='the oversampling rate. Type: int', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.5,
                        help='choose the threshold for binary classification to make a trade-off between recall and precision. Type: float')
    parser.add_argument('--data_save_path', type=str,
                        help='directory containing dataset.pkl (build_dataset output).', default='../output')
    parser.add_argument('--rawdata_path', type=str, default='../rawdata/example')
    # parser.add_argument('--data_info_txt',type=str, help='the file that saves the information of the data to parse')
    # parser.add_argument('--data_usage',type=str,help='decide whether to generate train dataset or test dataset')
    parser.add_argument('--predict_path', type=str, help='the directory used to save the prediction result. Type: str',
                        default='../prediction/example')
    parser.add_argument('--droplast', action='store_true')
    parser.add_argument('--feat_reduce', type=int, nargs='+', default=[0, 0])
    parser.add_argument('--masking', type=str, default='critical')
    parser.add_argument('--design', type=str)
    parser.add_argument('--unet', action='store_true', help='decide whether the layoutnet use the unet architecture')
    parser.add_argument('--pooling', type=str, default='max', help='the pooling type for layoutnet')
    parser.add_argument('--norm', action='store_true', help='decide whether to normalize the input feature or not')
    parser.add_argument('--task', type=str, default='reg',
                        help="decide a classification task or regression task, valid values: ['cls','reg']")
    parser.add_argument('--attn', action='store_true', help='decide whether to apply attention mechanism in GNN')
    parser.add_argument('--num_heads', type=int, default=1, help='Decide the number of heads for attention mechanism')
    parser.add_argument('--training_set', type=str, default=None)
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--reproducibility', action='store_true')
    parser.add_argument('--seed', type=int, default=9294)
    # transfer setting
    parser.add_argument('--train_node', type=str, default='7')
    parser.add_argument('--test_node', type=str, default='7')
    parser.add_argument('--load_ckpt_path', type=str, default=None)
    parser.add_argument('--ft_mode', type=str, default=None)
    parser.add_argument('--time_unit_trans', type=str, default=None)
    # batch balanced training
    parser.add_argument('--sample_45_num', type=int, default=1,
                        help='number of source-domain batches per target batch')
    parser.add_argument('--loss_weight_45', type=float, default=1.0,
                        help='weight for source-domain loss in balanced training')
    parser.add_argument('--src_loss_anneal_start', type=int, default=-1,
                        help='epoch (1-based) to start annealing source loss weight; <0 disables')
    parser.add_argument('--src_loss_anneal_end', type=int, default=-1,
                        help='epoch (1-based) to end annealing source loss weight; <0 disables')
    parser.add_argument('--src_loss_final_scale', type=float, default=1.0,
                        help='final scale on loss_weight_45 after annealing (e.g., 0.3)')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='early stop patience on val_r2 (0 disables early stop)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum val_r2 improvement to reset early-stop counter')
    parser.add_argument('--lr_scheduler', type=str, default='none',
                        choices=['none', 'cosine_wr', 'cosine', 'cosineannealingwarmrestarts', 'plateau', 'reduce_on_plateau'],
                        help='learning-rate scheduler type')
    parser.add_argument('--cosine_t0', type=int, default=0,
                        help='T_0 for cosine warm restarts; <=0 means auto (num_epoch//4)')
    parser.add_argument('--cosine_t_mult', type=int, default=2,
                        help='T_mult for cosine warm restarts')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6,
                        help='minimum LR for cosine warm restarts')
    parser.add_argument('--plateau_factor', type=float, default=0.5,
                        help='LR decay factor for ReduceLROnPlateau')
    parser.add_argument('--plateau_patience', type=int, default=5,
                        help='patience (eval rounds) for ReduceLROnPlateau')
    parser.add_argument('--plateau_threshold', type=float, default=1e-4,
                        help='minimum metric change to qualify as improvement for ReduceLROnPlateau')
    parser.add_argument('--plateau_min_lr', type=float, default=1e-6,
                        help='minimum LR for ReduceLROnPlateau')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='DataLoader prefetch factor (effective only when num_workers > 0)')
    parser.add_argument('--persistent_workers', type=int, default=1,
                        help='DataLoader persistent workers flag (1 enable, 0 disable)')
    parser.add_argument('--val_eval_interval', type=int, default=1,
                        help='run validation every N epochs')
    parser.add_argument('--test_eval_interval', type=int, default=50,
                        help='run test evaluation every N epochs')
    # disentangle and alignment
    parser.add_argument('--node_feat_dim', type=int, default=128)
    parser.add_argument('--con_temp', type=float, default=1.0)
    parser.add_argument('--cmd_k', type=int, default=5)
    parser.add_argument('--clr_proj_dim', type=int, default=64,
                        help='projection dim for CLR head; CLR is applied on a dedicated projection head')
    parser.add_argument('--cmd_proj_dim', type=int, default=64,
                        help='projection dim for CMD residual alignment head')
    parser.add_argument('--weight_clr', type=float, default=0.0)
    parser.add_argument('--weight_cmd', type=float, default=0.0)
    parser.add_argument('--clr_label_mode', type=str, default='semantic_arc',
                        choices=['semantic_arc', 'topology', 'cell_type'],
                        help='CLR positive labels: semantic_arc uses topology + canonical arc role; topology merges drive variants; cell_type keeps exact types')
    parser.add_argument('--cmd_label_mode', type=str, default='semantic_arc',
                        choices=['semantic_arc', 'topology', 'cell_type'],
                        help='grouping key for conditional CMD; semantic_arc is recommended')
    parser.add_argument('--not_retain_graph', action='store_true', default=False)
    parser.add_argument('--norm_clr', action='store_true', default=False)
    # bayesian learning
    parser.add_argument('--mc_num', default=10, type=int)
    parser.add_argument('--kl', action='store_true', default=False)
    parser.add_argument('--gaussian_prior', action='store_true', default=False)
    parser.add_argument('--weight_kl', type=float, default=1.)
    parser.add_argument('--sampling', action='store_true', default=False)
    parser.add_argument('--ema_momentum', type=float, default=0.01)
    parser.add_argument('--fix_alpha', action='store_true', default=False)
    parser.add_argument('--res_alpha', type=float, default=0.0)
    parser.add_argument('--kl_start_ep', type=int, default=0)
    parser.add_argument('--test_bayesian', action='store_true', default=False)
    parser.add_argument('--test_bayesian_only', action='store_true', default=False)
    # test the paths w. highest delay
    parser.add_argument('--test_delay_ratio', type=float, default=1.)
    # test for adv only model
    parser.add_argument('--adv_only', action='store_true')
    # build_dataset settings
    parser.add_argument("--src_lib", type=str, default="../data/lib/Nangate45",
                        help="Nangate45 lib root dir (for build_dataset)")
    parser.add_argument("--tgt_lib", type=str, default="../data/lib/ASAP7",
                        help="ASAP7 lib root dir (for build_dataset)")
    parser.add_argument("--src_spi", type=str, default="../data/spi/Nangate45",
                        help="Nangate45 SPI root dir (for build_dataset)")
    parser.add_argument("--tgt_sp", type=str, default="../data/spi/ASAP7",
                        help="ASAP7 SP root dir or file (for build_dataset)")
    parser.add_argument("--out_dir", type=str, default="../output",
                        help="Output dir for build_dataset")
    parser.add_argument("--target_label_ratio", type=float, default=1,
                        help="Ratio of labeled data in target train pool (for build_dataset)")
    parser.add_argument("--tgt_split_ratios", type=float, nargs=3, default=[0.14, 0.14, 0.72],
                        help="Target split ratios for train/val/test (for build_dataset)")
    parser.add_argument("--tgt_split_mode", type=str, default="random",
                        choices=["cell_type", "random", "stratified", "table_group"],
                        help="Target split mode: cell_type | random | stratified | table_group")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for dataset splitting (for build_dataset)")
    parser.add_argument("--dataset_pkl_name", type=str, default="dataset.pkl",
                        help="dataset pkl filename (used by build_dataset and training)")
    options = parser.parse_args(args)
    return options
