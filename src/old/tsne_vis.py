r"""
Baseline: Pretraining on 130nm data, finetuning on 7nm data.
1. Configure model cell feat dim to 34+95+8=137
2. Load data and shift the cell node feat
"""

import torch as th
import random
import os
import numpy as np

from torchmetrics import R2Score
from importlib.resources import path
from lib2to3.pytree import Node
from tkinter import N
from tracemalloc import start
from dataset import *
from options import get_options
from model_adapt import *
from TimeConv import *
from Unet import UNet
import dgl
import pickle
import os
from time import time
from random import shuffle
import itertools
from MyDataloader import *
import tee
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# with open('../rawdata/ctype2id.json', 'r') as f:
#     ctype2id = json.load(f)
#     num_ctypes = len(ctype2id)

idx2design = {}

device = th.device("cuda:" + str(get_options().gpu) if th.cuda.is_available() else "cpu")
R2_score = R2Score().to(device)
Loss = nn.CrossEntropyLoss() if get_options().task == 'cls' else nn.MSELoss()

DATA_SAVE_PATH = {
    '7': '../datasets/asap7-designs-repro',
    '130': '../datasets/130-designs'
}


def init_model(options):
    r"""

    initialize the model

    :param options:
        some additional parameters
    :return:
        param: options
        gnn : initialized gnn
        cnn: initialized gnn
        mlp: initialized mlp
    """

    # initialize the GNN model
    print('Intializing models...')
    # print(options.no_cnn,options.no_gnn)
    assert not options.no_cnn or not options.no_gnn, 'GNN and CNN model can not be both None!'
    mlp_dim = 0
    if options.no_gnn:
        gnn = None
    else:
        gnn = PathConv(
            in_feats_dim=options.out_dim,
            out_feats_dim=options.out_dim,
            cell_feat_dim=options.cell_feat_dim,
            net_feat_dim=options.net_feat_dim,
            flag_attn=options.attn,
            num_heads=options.num_heads
        )
        mlp_dim += gnn.out_feats_dim
    # initialize the cnn model
    if options.no_cnn:
        cnn = None
        fcn = None
    else:
        cnn = UNet(options.pooling) if options.unet else LayoutNet(options.pooling)
        fcn = nn.Linear(options.map_size * options.map_size, options.cnn_outdim)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(fcn.weight, gain=gain)
        mlp_dim += options.cnn_outdim
    # initialze the MLP
    mlp_7 = MLP2(
        in_dim=mlp_dim,
        # in_dim = options.out_dim,
        out_dim=options.nlabels,
        nlayers=options.n_fcn,
        dropout=options.mlp_dropout
    )

    mlp_130 = MLP2(
        in_dim=mlp_dim,
        # in_dim = options.out_dim,
        out_dim=options.nlabels,
        nlayers=options.n_fcn,
        dropout=options.mlp_dropout
    )
    # fcn=None
    model = SepPathModel(gnn, fcn, mlp_7, mlp_130)
    print("creating model in:", options.model_saving_dir)
    print('Path Model', model)
    print('cnn:', cnn)
    # save the model and create a file to save the results
    if os.path.exists(os.path.join(options.model_saving_dir, 'model.pkl')) is False:
        with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
            parameters = options
            pickle.dump((parameters, model, cnn), f)
        with open(os.path.join(options.model_saving_dir, 'res.txt'), 'w') as f:
            pass
    print('Model Initializing is accomplished!')

    return parameters, model, cnn


def load_model(device, options):
    r"""
    Load the model

    :param device:
        the target device that the model is loaded on
    :param options:
        some additional parameters
    :return:
        param: new options
        gnn : loaded gnn
        cnn: loaded gnn
        mlp: loaded mlp
    """
    print('----------------Loading the model and hyper-parameters----------------')
    model_dir = options.model_saving_dir
    # if there is no model in the target directory, break
    if os.path.exists(os.path.join(model_dir, 'model.pkl')) is False:
        param, model, cnn = init_model(options)
    else:
        # read the pkl file that saves the hype-parameters and the model.
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            # param: hyper-parameters, e.g., learning rate;
            # load the model from pickle file
            param, model, cnn = pickle.load(f)
            param.model_saving_dir = options.model_saving_dir
            # make some changes to the options
            if options.change_lr:
                param.learning_rate = options.learning_rate
            if options.change_alpha:
                param.alpha = options.alpha
    print(f'Model successfully initialized!')
    # model = nn.DataParallel(model, device_ids=[0,1])
    # cnn = nn.DataParallel(cnn, device_ids=[0,1])
    if options.load_ckpt_path is not None:
        with open(os.path.join(options.load_ckpt_path, 'model.pkl'), 'rb') as f:
            _, model, cnn = pickle.load(f)
        print(f'Model and hyper-parameters successfully loaded from {options.load_ckpt_path}')
    model = model.to(device)
    if cnn is not None:
        cnn = cnn.to(device)
    return param, model, cnn


def extract_feat(loader, device, model, cnn, beta, options):

    overall_loss, overall_acc, overall_recall, overall_precision, overall_f1, overall_r2 = 0, 0.0, 0, 0, 0, 0
    # runtime = 0
    res = []
    with th.no_grad():
        # load validation data, one batch at a time
        # each time we sample some central nodes, together with their input neighborhoods (in_blocks) \
        # and output neighborhoods (out_block).
        # The dst_nodes of the last block of in_block/out_block is the central nodes.
        print('validate:')
        case_idx = 0
        all_dataset_feat = []
        for path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks in loader:

            start_time = time()
            cnn_inputs = cnn_inputs.reshape((1, cnn_inputs.shape[0], cnn_inputs.shape[1], cnn_inputs.shape[2]))
            feat_map = cnn(cnn_inputs.to(device)).reshape((1, -1)) if cnn is not None else None
            # path_mask = path_mask.to_sparse()
            # transfer the data to GPU
            graph = graph.to(device)
            count_target = 0
            label_hats = None
            all_feat = None
            target_list = []
            # first_level_nodes = topo_levels[0][0]
            # init_message = graph.ndata['h'][first_level_nodes]
            path_loader = DataLoader(path_dataset, batch_size=len(path_dataset.paths), shuffle=False)
            for path_ids in path_loader:
                path_ids = list(set(path_ids.numpy().tolist()))
                sampled_ends, sampled_paths = {}, {}
                for i, pathid in enumerate(path_ids):
                    level = path2level[pathid]
                    endpoint = path2endpoint[pathid]
                    # assert endpoints2path[endpoint] == pathid
                    sampled_ends[level] = sampled_ends.get(level, [])
                    sampled_ends[level].append(endpoint)
                    sampled_paths[level] = sampled_paths.get(level, [])
                    sampled_paths[level].append(pathid)

                for level_id, level in enumerate(topo_levels):
                    nodes, eids = level[:2]
                    if len(level) == 4:
                        eids = eids.to(device)
                    targets = sampled_ends.get(level_id, [])
                    paths = sampled_paths.get(level_id, [])
                    target_list.extend(targets)
                    count_target += len(target_list)

                    if options.no_cnn or len(paths) == 0:
                        path_map = None
                    else:
                        path_mask = th.index_select(path_masks, 0, th.tensor(paths)).to(device)
                        path_map = path_mask.to_dense() * feat_map
                        # path_map = path_map.view(-1,path_map.shape[1]*path_map.shape[2])

                    output = model(graph, nodes, eids, targets, level_id, path_map, return_feat=True)

                    if len(paths) == 0:
                        continue
                    cur_label_hats, feat = output
                    print(f'feat shape: {feat.shape}')
                    if all_feat is None:
                        all_feat = feat
                    else:
                        all_feat = th.cat((all_feat, feat), dim=0)
            all_dataset_feat.append(all_feat)
    return all_dataset_feat


def split_dataset(paths, critical_paths):
    non_critical_paths = list(set(paths) - set(critical_paths))
    shuffle(critical_paths)
    val_paths = critical_paths[:int(len(critical_paths) / 5)]
    test_paths = critical_paths[int(len(critical_paths) / 5):]
    shuffle(non_critical_paths)
    val_paths.extend(non_critical_paths[:int(len(non_critical_paths) / 5)])
    test_paths.extend(non_critical_paths[int(len(non_critical_paths) / 5):])

    return val_paths, test_paths

def minMax_scalar(a):
    min_a = th.min(a)
    max_a = th.max(a)
    return (a - min_a) / (max_a - min_a)


def norm(feature, start_idx):
    num_feat = feature.shape[1]
    for i in range(start_idx, num_feat):
        feature[:, i: i + 1] = minMax_scalar(feature[:, i]).reshape(-1, 1)
    return feature


def load_data(data_path, usage, init_feat_dim, os_rate, feat_reduce, if_norm, design_set=None, node='7',
              time_unit_trans=None):
    """

    load the data

    :param dataset_file: str
            a pickle file that saves the dataset
    :param init_feat_dim: int
            the dimension of the initial feature
    :return:
        updated_dataset: List[(graph, topo_levels)]
            where graph is the DAG representation of a circuit,
            and topo_levels are the calculated topological levels
    """
    assert usage in ['train', 'test'], "Wrong data usage! Should be either 'train' or 'test'."
    if design_set is None:
        design_list_file = os.path.join(data_path, '{}data_list.txt'.format(usage))
        assert os.path.exists(design_list_file), \
            "Can not find the traindata list txt '{}'".format(design_list_file)

        with open(design_list_file, 'r') as f:
            lines = f.readlines()
            design_list = [l.replace('\n', '') for l in lines]
    else:
        design_list = design_set.split(',')
    print('--- {} designs: '.format(usage), design_list)
    datasets = []
    for i, design in enumerate(design_list):
        dataset_file = os.path.join(data_path, '{}.pkl'.format(design))
        graph, topo_levels, path_masks, path2level, path2endpoint, critical_paths, cnn_inputs = th.load(dataset_file)
        # with open(dataset_file,'rb') as f:
        # graph, topo_levels, path_masks,path2level,path2endpoint,critical_paths,cnn_inputs = pickle.load(f)
        print(path_masks.shape, graph.ndata['cell_feat'].shape)

        # relocate the cell feature for merge 130nm and 7nm feature
        original_cell_feat = graph.nodes['pin'].data['cell_feat']
        print(f'original cell feat for node {node}: {original_cell_feat.shape}')
        n_nodes, _ = original_cell_feat.shape
        new_shape = (n_nodes, 137)  # 34+95+8
        new_cell_feat = th.zeros(new_shape, dtype=original_cell_feat.dtype, device=original_cell_feat.device)
        if node == '7':
            # put the 7nm cell type feat first
            new_cell_feat[:, :34] = original_cell_feat[:, :34]
            # print(f'pin info feat: {original_cell_feat[:, -8:].shape}')
            new_cell_feat[:, -8:] = original_cell_feat[:, -8:]
        elif node == '130':
            # put the 130nm cell type feat after the 7nm feat
            cell_type_feat_130 = original_cell_feat[:, :-8]
            # print(f'130 cell type feat shape: {cell_type_feat_130.shape}')
            # print(new_cell_feat[:, 34:-8].shape)
            new_cell_feat[:, 34:-8] = original_cell_feat[:, :-8]
            # print(f'pin info feat: {original_cell_feat[:, -8:].shape}')
            new_cell_feat[:, -8:] = original_cell_feat[:, -8:]
        else:
            raise ValueError
        graph.nodes['pin'].data['cell_feat'] = new_cell_feat
        # shape_ = graph.ndata['cell_feat'].shape
        # print(f'After feat transform, the new cell feat {shape_}')
        # exit()

        # change the unit of the arrival time (ps <-> ns)
        if time_unit_trans is not None:
            end = graph.nodes['pin'].data['end']
            indices = th.where(end == 1)[0]
            arrival_time = graph.nodes['pin'].data['arrival_time']
            print(f'arrival time shape: {arrival_time.shape} | elements: {arrival_time[indices]}')
            if time_unit_trans == '130_to_7':
                "ns -> ps"
                new_arrival_time = arrival_time * 1000.
                print(f'new arrival time: {new_arrival_time[indices]}')
            elif time_unit_trans == '7_to_130':
                "ps -> ns"
                new_arrival_time = arrival_time / 1000.
                print(f'new arrival time: {new_arrival_time[indices]}')
            elif time_unit_trans == 'div_10':
                new_arrival_time = arrival_time / 10.
                print(f'new arrival time: {new_arrival_time[indices]}')
            elif time_unit_trans == 'div_100':
                new_arrival_time = arrival_time / 100.
                print(f'new arrival time: {new_arrival_time[indices]}')
            else:
                raise ValueError
            graph.nodes['pin'].data['arrival_time'] = new_arrival_time
            arr_time = graph.ndata['arrival_time']
            print(f'after mapping: the new arrival time is: {arr_time[indices]}')
        graph.ndata['h'] = th.zeros((graph.number_of_nodes(), init_feat_dim), dtype=th.float)
        graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float)
        # import pdb
        # pdb.set_trace()
        # net_feat = graph.ndata['net_feat']
        # print(f'Before reduce: {net_feat.shape}')
        if feat_reduce is not None:
            if feat_reduce[1] != 0:
                graph.ndata['net_feat'] = graph.ndata['net_feat'][:, :-feat_reduce[1]]
            if feat_reduce[0] != 0:
                graph.ndata['cell_feat'] = graph.ndata['cell_feat'][:, :-feat_reduce[0]]
            # print(graph.ndata['cell_feat'][:5])
        # net_feat = graph.ndata['net_feat']
        # print(f'After reduce: {net_feat.shape}')
        # exit()
        # normalize all the features, so that the value of different feature will not differ greatly
        # graph.ndata['cell_feat'] = transform_cellfeat(graph.ndata['cell_feat'])
        if if_norm:
            graph.ndata['cell_feat'] = norm(graph.ndata['cell_feat'], num_ctypes)
            graph.ndata['net_feat'] = norm(graph.ndata['net_feat'], num_ctypes)

        if type(cnn_inputs) == np.ndarray:
            cnn_inputs = th.from_numpy(cnn_inputs).float()
        # cnn_inputs = th.unsqueeze(cnn_inputs,dim=0)
        paths = list(range(len(graph.ndata['end'][graph.ndata['end'].squeeze() == 1])))
        # non_critical_paths = list(set(paths)-set(critical_paths))
        num_neg = len(paths) - len(critical_paths)
        num_pos = len(critical_paths)
        ratio = num_neg / num_pos - 1

        # if usage == 'test':
        #
        #     split_file = os.path.join(data_path, '{}_split.pkl'.format(design))
        #     if os.path.exists(split_file):
        #         with open(split_file, 'rb') as f:
        #             val_paths, test_paths = pickle.load(f)
        #     else:
        #         val_paths, test_paths = split_dataset(paths, critical_paths)
        #         with open(split_file, 'wb') as f:
        #             pickle.dump((val_paths, test_paths), f)
        #     paths = val_paths
        #     print(len(paths), len(critical_paths))

        if usage == 'train':
            idx2design[i] = design
        if usage == 'train' and os_rate != 0 and ratio > 1:
            # shuffle(critical_paths)
            for _ in range(os_rate):
                paths.extend(critical_paths)
            # while ratio>=1:
            #     paths.extend(critical_paths)
            #     ratio -= 1
            # shuffle(critical_paths)
            # paths.extend(critical_paths[:int(ratio*num_pos)])
        path_dataset = PathDataset(paths)

        datasets.append(
            (path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks)
        )

    return datasets


def judge_critical(pred_arr_time, required_time):
    pred_slack = required_time - pred_arr_time
    is_critical = th.ones(pred_arr_time.shape)
    is_critical[pred_slack >= 0] = 0
    return is_critical


def sample_paths(path_ids, path2level, path2endpoint):
    # sample level2path mapping
    sampled_ends, sampled_paths = {}, {}
    for _, pathid in enumerate(path_ids):
        level = path2level[pathid]
        endpoint = path2endpoint[pathid]
        # assert endpoints2path[endpoint] == pathid
        sampled_ends[level] = sampled_ends.get(level, [])
        sampled_ends[level].append(endpoint)
        sampled_paths[level] = sampled_paths.get(level, [])
        sampled_paths[level].append(pathid)
    return sampled_ends, sampled_paths


def set_path_loader(graph, path_dataset, path2level, batch_size, device):
    # prepare the path dataloader, feat map and graph
    graph = graph.to(device)
    # print(graph)
    if len(path2level) <= batch_size:
        path_loader = DataLoader(path_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        path_loader = DataLoader(path_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_batch = len(path_loader)
    return graph, path_loader, num_batch


def path_batch_forward(path_ids, path2level, path2endpoint, topo_levels, feat_map, model, graph, path_masks, node='7'):
    # forward of a batch of paths in the netlist
    path_ids = list(path_ids.numpy().tolist())
    sampled_ends, sampled_paths = sample_paths(path_ids, path2level, path2endpoint)
    count_target = 0
    label_hats = None
    target_list = []
    for level_id, level in enumerate(topo_levels):
        nodes, eids = level[:2]
        if len(level) == 2:
            eids = eids.to(device)
        targets = sampled_ends.get(level_id, [])
        paths = sampled_paths.get(level_id, [])
        # print(level_id,len(targets))
        target_list.extend(targets)
        count_target += len(target_list)
        # print(level_id,len(nodes))
        if options.no_cnn or len(paths) == 0:
            path_map = None
        else:
            # path_mask = path_masks[paths].to(device)
            path_mask = th.index_select(path_masks, 0, th.tensor(paths)).to(device)
            path_map = path_mask.to_dense() * feat_map
            # path_map = path_map.view(-1,path_map.shape[1]*path_map.shape[2])
        # print(path_map.shape)
        # path_map = None
        # print(level_id,path_map.shape)
        cur_label_hats = model(graph, nodes, eids, targets, level_id, path_map, node=node)

        if len(paths) == 0:
            continue

        if label_hats is None:
            label_hats = cur_label_hats
        else:
            label_hats = th.cat((label_hats, cur_label_hats), dim=0)
    # labels = graph.ndata['label'][target_list].squeeze()
    arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
    return label_hats, arrival_time


def get_130_design(bid, len_130_designs, sample_num):
    # sample the design indices given the batch id
    start_idx = bid * sample_num
    sampled_idx = []
    for i in range(start_idx, start_idx + sample_num):
        sampled_idx.append(i % len_130_designs)
    return sampled_idx


def main(options, seed):
    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

    # you can define your dataset file here
    # data_save_path = options.data_save_path
    train_data_save_path = DATA_SAVE_PATH[options.train_node]
    test_data_save_path = DATA_SAVE_PATH[options.test_node]
    print(f'train data save path: {train_data_save_path}')
    print(f'test data save path: {test_data_save_path}')

    # load the model
    options.cell_feat_dim -= options.feat_reduce[0]
    options.net_feat_dim -= options.feat_reduce[1]
    options, model, cnn = load_model(device, options)
    with open(os.path.join(options.model_saving_dir, 'seed.txt'), 'a') as f:
        f.write(str(seed))

    print("----------------Loading data----------------")
    # train_data_file = os.path.join(data_save_path, 'train.pkl')

    # train_dataset_7 = load_data('../datasets/asap7-designs-repro', 'train', options.out_dim, options.os_rate,
    #                             options.feat_reduce,
    #                             options.norm, design_set='smallboom', node='7')
    train_dataset_130 = load_data('../datasets/130-designs', 'train', options.out_dim, options.os_rate,
                                  options.feat_reduce,
                                  options.norm, design_set=options.training_set, node='130')

    # val_data_file = os.path.join(data_save_path, 'test.pkl')
    val_dataset = load_data(test_data_save_path, 'test', options.out_dim, options.os_rate, options.feat_reduce,
                            options.norm, design_set=options.test_set, node=options.test_node,
                            time_unit_trans=options.time_unit_trans)

    # split the validation set and test set
    beta = options.beta

    all_feat_7 = extract_feat(val_dataset, device, model, cnn, beta, options)
    all_feat_130 = extract_feat(train_dataset_130, device, model, cnn, beta, options)
    names_7 = ['arm9', 'chacha', 'hwacha', 'or1200', 'sha3', 'rocket', 'smallboom', 'steecore']
    names_130 = ['jpeg', 'linkruncca', 'spiMaster', 'usbf_device']
    tsne(all_feat_7, all_feat_130, names_7, names_130, load=False, save=True, sample=1000)


def merge_feature(feat_all, label_all):
    return np.concatenate(feat_all), np.concatenate(label_all)


def draw_separate(all_feat, all_label, names_7, names_130, size_7, size_130, tsne_path):
    start = 0
    for i, n in enumerate(names_7):
        dataset_path = os.path.join(tsne_path, f'{n}-7nm.png')
        dataset_len = size_7[i]
        feat_i = all_feat[start: start+dataset_len]
        label_i = all_label[start: start+dataset_len]
        start += dataset_len
        print(f'feat i shape: {feat_i.shape}')
        plt.scatter(feat_i[:, 0], feat_i[:, 1], s=0.5)
        plt.savefig(dataset_path)
        plt.close()
    for i, n in enumerate(names_130):
        dataset_path = os.path.join(tsne_path, f'{n}-130nm.png')
        dataset_len = size_130[i]
        feat_i = all_feat[start: start+dataset_len]
        label_i = all_label[start: start+dataset_len]
        start += dataset_len
        plt.scatter(feat_i[:, 0], feat_i[:, 1], s=0.5)
        plt.savefig(dataset_path)
        plt.close()


def tsne(all_feat_7, all_feat_130, names_7, names_130, load=False, save=False, sample=500, separate=False):
    # concat all the features
    tsne_path = '/research/d4/gds/xyzhang21/Timing/DAC23-Timing-Prediction/src/tsne/raw'
    if load:
        all_feat = torch.load(os.path.join(tsne_path, 'all_feat.pt'))
        all_label = torch.load(os.path.join(tsne_path, 'all_label.pt'))
        # all_feat_np = np.load(os.path.join(tsne_path, 'all_feat_np.npy'))
        # all_label_np = np.load(os.path.join(tsne_path, 'all_label.npy'))
    else:
        assert len(all_feat_7) == len(names_7)
        assert len(all_feat_130) == len(names_130)
        idx2name = {}
        # total_feat_num_7 = sum([d.shape[0] for d in all_feat_7])
        # total_feat_num_130 = sum([d.shape[0] for d in all_feat_130])
        # print(total_feat_num_7, total_feat_num_130)
        feat_dim = all_feat_7[0].shape[1]
        total_feat_num_7 = len(names_7) * sample
        total_feat_num_130 = len(names_130) * sample
        print(total_feat_num_7, total_feat_num_130)
        all_feat = torch.zeros((total_feat_num_7+total_feat_num_130, feat_dim))
        all_label = torch.zeros((total_feat_num_7+total_feat_num_130))
        print(f'info dataset 7nm')
        idx = 0
        label = 0
        size_7, size_130 = [], []
        for i, dataset_7 in enumerate(all_feat_7):
            print(f'len of dataset: {dataset_7.shape[0]}')
            dataset_7 = dataset_7[:sample]
            size_7.append(dataset_7.shape[0])
            for f in dataset_7:
                idx2name[idx] = names_7[i]
                all_feat[idx] = f
                all_label[idx] = label
                idx += 1
            label += 1
        print(f'info dataset 130nm')
        for i, dataset_130 in enumerate(all_feat_130):
            print(f'len of dataset: {dataset_130.shape[0]}')
            dataset_130 = dataset_130[:sample]
            size_130.append(dataset_130.shape[0])
            for f in dataset_130:
                idx2name[idx] = names_130[i]
                all_feat[idx] = f
                all_label[idx] = label
                idx += 1
            label += 1
        print(f'all feat shape: {all_feat.shape} idx: {idx} label: {label}')
        # all_feat_np = all_feat.numpy()
        # all_label_np = all_label.numpy()
        if save:
            th.save(all_feat, os.path.join(tsne_path, 'all_feat.pt'))
            th.save(all_label, os.path.join(tsne_path, 'all_label.pt'))
            # np.save(os.path.join(tsne_path, 'all_feat_np.npy'), all_feat_np)
            # np.save(os.path.join(tsne_path, 'all_label.npy'), all_label_np)
    print(f'all feat shape: {all_feat.shape}')
    print(f'all label shape: {all_label.shape}')
    all_feat_np = all_feat.numpy()
    all_label_np = all_label.numpy()
    print(f'all label: {np.unique(all_label_np)}')
    cmap = plt.cm.get_cmap('viridis', np.unique(all_label_np).size)
    tsne = TSNE(n_components=2)
    feat_2d = tsne.fit_transform(all_feat_np)
    print(f'feat 2d shape: {feat_2d.shape}')
    # Draw all the feat
    draw_separate(feat_2d, all_label_np, names_7, names_130, size_7, size_130, tsne_path)
    exit()
    save_path = os.path.join(tsne_path, 'tsne_all.png')
    plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=all_label_np, s=0.5)
    plt.legend(['arm9', 'chacha', 'hwacha', 'or1200', 'sha3', 'rocket', 'smallboom', 'steecore', 'jpeg', 'linkruncca', 'spiMaster', 'usbf_device'])
    plt.savefig(save_path)
    exit()
    filtered_feat_all = []
    filtered_labels_all = []
    if separate:
        label_idx = 0
        for i, n in enumerate(names_7):
            feat = feat_2d[i*sample:(i+1)*sample]
            # filter the feat
            if n == 'chacha':
                feat_filtered = feat[(feat[:, 0] > 0) & (feat[:, 0] < 50) & (feat[:, 1] > 20)]
                feat_filtered = feat_filtered[:500]
                len_filtered_feat = len(feat_filtered)
                label_filtered = th.zeros(len_filtered_feat).numpy()
                label_filtered[:] = label_idx
                label_idx += 1
                filtered_feat_all.append(feat_filtered)
                filtered_labels_all.append(label_filtered)
                print(f'len chacha: {len_filtered_feat}')
            elif n == 'or1200':
                feat_filtered = feat[(feat[:, 0] < -20) & (feat[:, 1] > 30)]
                feat_filtered = feat_filtered[:500]
                len_filtered_feat = len(feat_filtered)
                label_filtered = th.zeros(len_filtered_feat).numpy()
                label_filtered[:] = label_idx
                filtered_feat_all.append(feat_filtered)
                filtered_labels_all.append(label_filtered)
                print(f'len or1200: {len_filtered_feat}')
                label_idx += 1
            elif n == 'steelcore':
                feat_filtered = feat[(feat[:, 0] > -30) & (feat[:, 0] < 10) & (feat[:, 1] > 20)]
                len_filtered_feat = len(feat_filtered)
                label_filtered = th.zeros(len_filtered_feat).numpy()
                label_filtered[:] = label_idx
                label_idx += 1
                filtered_feat_all.append(feat_filtered)
                filtered_labels_all.append(label_filtered)
                print(f'len steelcore: {len_filtered_feat}')
            else:
                continue

        for i, n in enumerate(names_130):
            feat = feat_2d[(len(names_7)+i)*sample:(len(names_7)+i+1)*sample]
            if n == 'jpeg':
                feat_filtered = feat[(feat[:, 0] > 0) & (feat[:, 0] < 40) & (feat[:, 1] < 0)]
                len_filtered_feat = len(feat_filtered)
                label_filtered = th.zeros(len_filtered_feat).numpy()
                label_filtered[:] = label_idx
                label_idx += 1
                filtered_feat_all.append(feat_filtered)
                filtered_labels_all.append(label_filtered)
                print(f'len jpeg: {len_filtered_feat}')
            elif n == 'spiMaster':
                feat_filtered = feat[(feat[:, 0] < -20) & (feat[:, 0] > -75) & (feat[:, 1] < -20)]
                len_filtered_feat = len(feat_filtered)
                label_filtered = th.zeros(len_filtered_feat).numpy()
                label_filtered[:] = label_idx
                filtered_feat_all.append(feat_filtered)
                filtered_labels_all.append(label_filtered)
                print(f'len spiMaster: {len_filtered_feat}')
                label_idx += 1
            elif n == 'linkruncca':
                feat_filtered = feat[(feat[:, 0] > -25) & (feat[:, 0] < 10) & (feat[:, 1] < -15)]
                len_filtered_feat = len(feat_filtered)
                label_filtered = th.zeros(len_filtered_feat).numpy()
                label_filtered[:] = label_idx
                label_idx += 1
                filtered_feat_all.append(feat_filtered)
                filtered_labels_all.append(label_filtered)
                print(f'len link: {len_filtered_feat}')
            else:
                continue
        filtered_feat_all, filtered_label_all = merge_feature(filtered_feat_all, filtered_labels_all)
        # print(f'after filter: {filtered_feat_all.size} {filtered_label_all.size}')
        # plt.scatter(filtered_feat_all[:, 0], filtered_feat_all[:, 1], c=filtered_label_all, cmap=cmap, s=0.5)
        # plt.colorbar(ticks=range(np.unique(filtered_label_all).size))
        # plt.savefig('tsne-all.png')
        # exit()
        markers = ['^', '^', '^', 'o',  'o',  'o']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        for i in range(6):
            xi = filtered_feat_all[filtered_label_all == i, 0]
            yi = filtered_feat_all[filtered_label_all == i, 1]
            plt.scatter(xi, yi, marker=markers[i], color=colors[i], s=6)
        ax = plt.gca()
        # Hide x-axis
        ax.get_xaxis().set_visible(False)
        # Hide y-axis
        ax.get_yaxis().set_visible(False)
        plt.legend(['chacha-7nm', 'or1200-7nm', 'steelcore-7nm', 'jpeg-130nm', 'spiMaster-130nm', 'linkruncca-130nm'])
        plt.savefig('tsne-all.png')
    else:
        plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=all_label_np, cmap=cmap, s=0.5)
        plt.colorbar(ticks=range(np.unique(all_label_np).size))
        plt.savefig('tsne.png')


if __name__ == "__main__":
    options = get_options()
    seed = options.seed
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    stdout_f = '{}/stdout.log'.format(options.model_saving_dir)
    stderr_f = '{}/stderr.log'.format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    names_7 = ['arm9', 'chacha', 'hwacha', 'or1200', 'sha3', 'rocket', 'smallboom', 'steelcore']
    names_130 = ['jpeg', 'linkruncca', 'spiMaster', 'usbf_device']
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        main(options, seed)
        # tsne(None, None, names_7, names_130, load=True, save=False, sample=1000, separate=True)
        # 7nm: chacha, or1200, steelcore
        # 130nm: jpeg, spi, link