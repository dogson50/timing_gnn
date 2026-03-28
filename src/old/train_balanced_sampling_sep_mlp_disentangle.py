r"""
Baseline: Pretraining on 130nm data, finetuning on 7nm data.
1. Configure model cell feat dim to 34+95+8=137
2. Load data and shift the cell node feat
"""
import torch
import torch as th
import random
import os
import numpy as np
import torch.nn.functional as F
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

from time import time
from random import shuffle
import itertools
from MyDataloader import *
import tee
from torch.utils.data import DataLoader

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

    mlp_node = MLP2(
        in_dim=mlp_dim,
        # in_dim = options.out_dim,
        out_dim=options.node_feat_dim,
        nlayers=options.n_fcn,
        dropout=options.mlp_dropout
    )

    mlp_others = MLP3(
        in_dim=mlp_dim,
        # in_dim = options.out_dim,
        out_dim=mlp_dim-options.node_feat_dim,
        nlayers=options.n_fcn,
        dropout=options.mlp_dropout
    )

    # fcn=None
    # model = SepPathModel(gnn, fcn, mlp_7, mlp_130)
    model = DisSepPathModel(gnn, fcn, mlp_7, mlp_130, mlp_node, mlp_others)
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


def validate(loader, device, model, cnn, beta, options):
    r"""

    validate the model

    :param loader:
        the data loader to load the validation dataset
    :param device:
        device
    :param model:
        trained model
    :param mlp:
        trained mlp
    :param Loss:
        used loss function
    :param beta:
        a hyperparameter that determines the thredshold of binary classification
    :param options:
        some parameters
    :return:
        result of the validation: loss, acc,recall,precision,F1_score
    """

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
        for path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks in loader:

            total_num, total_loss, correct, fn, fp, tn, tp, total_r2 = 0, 0.0, 0, 0, 0, 0, 0, 0
            runtime = 0
            # optim.zero_grad()
            start_time = time()
            cnn_inputs = cnn_inputs.reshape((1, cnn_inputs.shape[0], cnn_inputs.shape[1], cnn_inputs.shape[2]))
            feat_map = cnn(cnn_inputs.to(device)).reshape((1, -1)) if cnn is not None else None
            # path_mask = path_mask.to_sparse()
            # transfer the data to GPU

            graph = graph.to(device)
            count_target = 0
            label_hats = None
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

                    output = model(graph, nodes, eids, targets, level_id, path_map)

                    if len(paths) == 0:
                        continue

                    cur_label_hats, _, _ = output
                    if label_hats is None:
                        label_hats = cur_label_hats
                    else:
                        label_hats = th.cat((label_hats, cur_label_hats), dim=0)

            # labels: the ground-truth binary labels, decide whether path is critical or not
            # predict_labels: the predicted binary labels
            # label_hats: the output of the model, len=2 for classification, len=1 for regression
            labels = graph.ndata['label'][target_list].squeeze()
            if options.task == 'cls':
                predict_labels = th.argmax(nn.functional.softmax(label_hats, 1), dim=1)
                test_loss = Loss(label_hats, labels)
                test_r2 = 0
            elif options.task == 'reg':
                required_time = graph.ndata['required_time'][target_list].squeeze()
                arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
                test_loss = Loss(label_hats, arrival_time)
                predict_labels = judge_critical(label_hats, required_time).to(device)
                test_r2 = R2_score(label_hats, arrival_time).to(device)
                total_r2 += test_r2.item()
                # print('R2 score: {}'.format(test_r2))
            # calculate loss

            # print(cnn.down3.maxpool_conv.weights)
            total_num += len(labels)
            total_loss += test_loss.item()

            correct += (
                    predict_labels == labels
            ).sum().item()
            # calculate fake negative, true positive, fake negative, and true negative rate
            fn += ((predict_labels == 0) & (labels != 0)).sum().item()
            tp += ((predict_labels != 0) & (labels != 0)).sum().item()
            tn += ((predict_labels == 0) & (labels == 0)).sum().item()
            fp += ((predict_labels != 0) & (labels == 0)).sum().item()

            acc = correct / total_num
            recall = 0
            precision = 0
            if tp != 0:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
            F1_score = 0
            if precision != 0 or recall != 0:
                F1_score = 2 * recall * precision / (recall + precision)

            overall_loss += total_loss
            overall_r2 += total_r2
            overall_recall += recall
            overall_f1 += F1_score
            overall_acc += acc
            overall_precision += precision

            # overall_correct += correct
            # overall_fn  += fn
            # overall_fp += fp
            # overall_tn += tn
            # overall_tp += tp
            # overall_num += total_num
            # print('case',case_idx)
            # case_idx += 1
            # print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(precision, 3))
            print("\tcase {} \tl:{:.3f}, r2:{:.3f}, rc:{:.3f}, F1:{:.3f}".format(case_idx, test_loss, test_r2, recall,
                                                                                 F1_score))
            case_idx += 1
            res.append([test_loss, test_r2, acc, recall, precision, F1_score])
    # calculate the overall loss / accuracy
    num_case = case_idx
    overall_loss = overall_loss / num_case
    overall_acc = overall_acc / num_case
    overall_r2 = overall_r2 / num_case
    overall_f1 = overall_f1 / num_case
    overall_recall = overall_recall / num_case
    overall_precision = overall_precision / num_case
    # calculate overall recall, precision and F1-score

    # print('overall val')
    # print("\ttp:", overall_tp, " fp:", overall_fp, " fn:", overall_fn, " tn:", overall_tn, " precision:", round(overall_precision, 3))
    print("\toverall r2:{:.3f}, rc:{:.3f}, F1:{:.3f}".format(overall_r2, overall_recall, overall_f1))

    return res, overall_f1, overall_r2


def split_dataset(paths, critical_paths):
    non_critical_paths = list(set(paths) - set(critical_paths))
    shuffle(critical_paths)
    val_paths = critical_paths[:int(len(critical_paths) / 5)]
    test_paths = critical_paths[int(len(critical_paths) / 5):]
    shuffle(non_critical_paths)
    val_paths.extend(non_critical_paths[:int(len(non_critical_paths) / 5)])
    test_paths.extend(non_critical_paths[int(len(non_critical_paths) / 5):])

    return val_paths, test_paths


# def transform_cellfeat(cell_feat):
#     new_cellfeat = th.zeros((cell_feat.shape[0], cell_feat.shape[1] - num_ctypes + 4))
#     cell_type = cell_feat[:, :num_ctypes]
#     cell_type = th.argmax(cell_type, dim=1, keepdim=True)
#     id2celltype = {}
#     for cell, id in ctype2id.items():
#         id2celltype[id] = cell
#
#     # new celltype feat:
#     #   buf: 1000
#     #   inv: 0100
#     #   register: 0010
#     #   others: 0001
#     {"AND": 0, "FA": 1, "HA": 2, "MAJI": 3, "MAJ": 4, "NAND": 5, "NOR": 6, "OR": 7, "TIEHI": 8,
#      "TIELO": 9, "XNOR": 10, "XOR": 11, "ASYNC_DFFH": 12, "DFFHQN": 13, "DFFHQ": 14, "DFFLQN": 15,
#      "DFFLQ": 16, "DHL": 17, "DLL": 18, "ICG": 19, "SDFH": 20, "SDFL": 21, "BUF": 22, "CKINVDC": 23,
#      "HB": 24, "INV": 25, "O2A1O1I": 26, "OA": 27, "OAI": 28, "A2O1A1I": 29, "A2O1A1O1I": 30, "AO": 31, "AOI": 32}
#     for i in range(cell_feat.shape[0]):
#         type = id2celltype[cell_type[i].item()]
#         new_cellfeat[i][0] = type == 'BUFF'
#         new_cellfeat[i][1] = type in ('INV', 'CKINVDC')
#         new_cellfeat[i][2] = type in ('TIEHI', "TIELO", "ASYNC_DFFH", "DFFHQN", "DFFHQ",
#                                       "DFFLQN", "DFFLQ", "DHL", "DLL", "ICG", "SDFH",
#                                       "SDFL", "HB")
#         new_cellfeat[i][3] = not (new_cellfeat[i][0] and new_cellfeat[i][1] and new_cellfeat[i][2])
#         new_cellfeat[i][4:] = cell_feat[i][num_ctypes:]
#         print(type, new_cellfeat[i])
# new_cellfeat[:,0:1] = cell_type == ctype2id['BUF']
# new_cellfeat[:,1:2] = cell_type == ctype2id['INV']
# new_cellfeat[:,2:3] = ( cell_type != ctype2id['INV'] and cell_type!=ctype2id['BUF'])
# new_cellfeat[:,3:] = cell_feat[:,num_ctypes:]


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

        if usage == 'test':

            split_file = os.path.join(data_path, '{}_split.pkl'.format(design))
            if os.path.exists(split_file):
                with open(split_file, 'rb') as f:
                    val_paths, test_paths = pickle.load(f)
            else:
                val_paths, test_paths = split_dataset(paths, critical_paths)
                with open(split_file, 'wb') as f:
                    pickle.dump((val_paths, test_paths), f)
            paths = val_paths
            print(len(paths), len(critical_paths))

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
    label_hats, feat_node, feat_others = None, None, None
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
        output = model(graph, nodes, eids, targets, level_id, path_map, node=node)

        if len(paths) == 0:
            continue
        cur_label_hats, cur_feat_node, cur_feat_others = output
        if label_hats is None:
            label_hats = cur_label_hats
        else:
            label_hats = th.cat((label_hats, cur_label_hats), dim=0)
        if feat_node is None:
            feat_node = cur_feat_node
        else:
            feat_node = th.cat((feat_node, cur_feat_node), dim=0)
        if feat_others is None:
            feat_others = cur_feat_others
        else:
            feat_others = th.cat((feat_others, cur_feat_others), dim=0)

    # labels = graph.ndata['label'][target_list].squeeze()
    arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
    return label_hats, arrival_time, feat_node, feat_others


def get_130_design(bid, len_130_designs, sample_num):
    # sample the design indices given the batch id
    start_idx = bid * sample_num
    sampled_idx = []
    for i in range(start_idx, start_idx + sample_num):
        sampled_idx.append(i % len_130_designs)
    return sampled_idx


def reset_graph_feature(graph, device):
    graph.ndata['h'] = th.zeros((graph.number_of_nodes(), options.out_dim), dtype=th.float).to(device)
    graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(
        device)
    graph.edges['cell'].data['e'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(
        device)


def contrastive_loss(node_feat_7, node_feat_130, temp, device, normalization=False):
    # calculate the contrastive loss
    num_feat_7 = node_feat_7.shape[0]
    num_feat_130 = node_feat_130.shape[0]
    if normalization:
        node_feat_7_ = F.normalize(node_feat_7, dim=1)
        node_feat_130_ = F.normalize(node_feat_130, dim=1)
    else:
        node_feat_7_ = node_feat_7
        node_feat_130_ = node_feat_130
    targets = th.zeros((num_feat_7+num_feat_130), dtype=torch.int32).to(device)
    targets[num_feat_7:] = 1 # assign node 130 label to 1
    all_node_feat = th.cat((node_feat_7_, node_feat_130_), dim=0)
    # print(f'all node feat shape: {all_node_feat.shape}')
    dot_product_tempered = th.mm(all_node_feat, all_node_feat.T) / temp
    # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
    exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
    )

    mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
    # print(f'mask sim class shape: {mask_similar_class.shape}')
    # print(f'mask sim 0: {mask_similar_class[0]}')
    # print(f'mask sim -1: {mask_similar_class[-1]}')
    mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
    mask_combined = mask_similar_class * mask_anchor_out
    cardinality_per_samples = torch.sum(mask_combined, dim=1)

    log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
    supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
    supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

    return supervised_contrastive_loss


def l2diff(x1, x2):
    """
        standard euclidean norm
        """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    """
        difference between moments
        """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    return l2diff(ss1, ss2)


def cmd_loss(others_feat_7, others_feat_130, k=5):
    # calculate the cmd loss
    x1 = others_feat_7
    x2 = others_feat_130
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(k - 1):
        scms.append(moment_diff(sx1, sx2, i + 2))
    return sum(scms)


def train(options, seed):
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

    print('Hyperparameters are listed as follows:')
    print(options)
    print('seed:', seed)
    print('The model architecture is shown as follow:')
    print(model)
    print(cnn)
    print("----------------Loading data----------------")
    # train_data_file = os.path.join(data_save_path, 'train.pkl')

    train_dataset_7 = load_data('../datasets/asap7-designs-repro', 'train', options.out_dim, options.os_rate,
                                options.feat_reduce,
                                options.norm, design_set='smallboom', node='7')
    train_dataset_130 = load_data('../datasets/130-designs', 'train', options.out_dim, options.os_rate,
                                  options.feat_reduce,
                                  options.norm, design_set=options.training_set, node='130')

    # val_data_file = os.path.join(data_save_path, 'test.pkl')
    val_dataset = load_data(test_data_save_path, 'test', options.out_dim, options.os_rate, options.feat_reduce,
                            options.norm, design_set=options.test_set, node=options.test_node,
                            time_unit_trans=options.time_unit_trans)

    # split the validation set and test set
    beta = options.beta

    # set the optimizer
    print(f'Setting the optimizer and weights to update')
    print(f'GNN params')
    for n, p in model.named_parameters():
        print(n)
    print(f'CNN params')
    for n, p in cnn.named_parameters():
        print(n)
    if cnn is not None:
        if options.ft_mode is None or options.ft_mode == "all":
            # optim = th.optim.Adam(
            #     itertools.chain(model.parameters(), cnn.parameters()),
            #     # model.parameters(),
            #     options.learning_rate, weight_decay=options.weight_decay
            # )
            print(f'Optimiza all the parameters!')
            pass
        elif options.ft_mode == 'mlp_gnn_cell':
            # freeze the gnn net, gnn neigh and cnn
            print(f'Optimize the mlp and gnn-cell only!')
            freeze_modules = [model.gnn[0].fc_cell_neigh, model.gnn[0].fc_net_self, model.fcn, cnn]
            for m in freeze_modules:
                for param in m.parameters():
                    param.requires_grad = False

        elif options.ft_mode == 'mlp_gnn_cell_fcn':
            print(f'optimize the mlp, gnn cell and fcn.')
            freeze_modules = [model.gnn[0].fc_cell_neigh, model.gnn[0].fc_net_self, cnn]
            for m in freeze_modules:
                for param in m.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("Wrong ft mode!")

        optim = th.optim.Adam(
            itertools.chain(filter(lambda p: p.requires_grad, model.parameters()),
                            filter(lambda p: p.requires_grad, cnn.parameters())),
            options.learning_rate, weight_decay=options.weight_decay
        )
        cnn.train()
        model.train()
    else:
        optim = th.optim.Adam(
            model.parameters(),
            options.learning_rate, weight_decay=options.weight_decay
        )
        model.train()
    # check the gradients setting
    # print(f'gnn cell req grad: {model.gnn[0].fc_cell_self.layers[0].weight.requires_grad}')
    # print(f'gnn net req grad: {model.gnn[0].fc_net_self.layers[0].weight.requires_grad}')
    # print(f'gnn neigh req grad: {model.gnn[0].fc_cell_neigh.layers[2].weight.requires_grad}')
    # print(f'model fcn req grad: {model.fcn[0].weight.requires_grad}')
    # print(f'model mlp req grad: {model.mlp[0].layers[0].weight.requires_grad}')
    # print(f'cnn req grad {cnn.encode[0].weight.requires_grad}')
    print("----------------Start training---------------")
    pre_loss = 100
    stop_score = 0
    max_F1_score, max_r2 = 0, 0
    print(f'index2design: {idx2design}')
    th.autograd.set_detect_anomaly(True)
    len_130_dataset = len(train_dataset_130)
    dataset_130_index = 0
    batch_size_7 = options.batch_size // 2
    batch_size_130 = options.batch_size // (2 * options.sample_130_num)
    print(
        f'For each batch we sample {batch_size_7} 7nm data and {batch_size_130} 130nm data from {options.sample_130_num} designs')
    for epoch in range(options.num_epoch):
        # assume only one 7nm design is loaded
        path_dataset_7, graph_7, path2level_7, path2endpoint_7, topo_levels_7, cnn_inputs_7, path_masks_7 = \
            train_dataset_7[0]
        graph_7, path_loader_7, num_batch_7 = set_path_loader(graph_7, path_dataset_7, path2level_7, batch_size_7,
                                                              device)
        # initialize the 130nm dataloader
        feat_map_130_all, graph_130_all, path_loader_130_all, path_loader_iter_130_all = [], [], [], []
        path2level_130_all, path2endpoint_130_all, topo_levels_130_all, cnn_inputs_130_all, path_masks_130_all = [], [], [], [], []
        for i, (path_dataset_130, graph_130, path2level_130, path2endpoint_130, topo_levels_130, cnn_inputs_130,
                path_masks_130) in enumerate(train_dataset_130):
            graph_130, path_loader_130, num_batch_130 = set_path_loader(graph_130, path_dataset_130, path2level_130,
                                                                        batch_size_130, device)
            graph_130_all.append(graph_130)
            path_loader_130_all.append(path_loader_130)
            path_loader_iter_130_all.append(iter(path_loader_130))
            path2level_130_all.append(path2level_130)
            path2endpoint_130_all.append(path2endpoint_130)
            topo_levels_130_all.append(topo_levels_130)
            cnn_inputs_130_all.append(cnn_inputs_130)
            path_masks_130_all.append(path_masks_130)

        print(f'{len(graph_130_all)} 130nm data loaded!')

        '''
        For each batch, sample a sub-batch from the 7nm dataset and then one (or more) sub-batches 
        from the 130nm dataset.
        '''
        path_loader_7_iter = iter(path_loader_7)
        for bidx in range(num_batch_7):
            start_time = time()
            # extract the image feat
            feat_map_7 = cnn(cnn_inputs_7.to(device)).reshape((1, -1)) if cnn is not None else None
            feat_map_130_all = [cnn(cnn_input.to(device)).reshape((1, -1)) if cnn is not None else None for cnn_input in
                                cnn_inputs_130_all]
            path_ids_7 = next(path_loader_7_iter)
            label_hats_7, arrival_time_7, feat_node_7, feat_others_7 = path_batch_forward(path_ids_7, path2level_7,
                                                                                          path2endpoint_7,
                                                                                          topo_levels_7,
                                                                                          feat_map_7, model, graph_7,
                                                                                          path_masks_7)
            # sample from 130nm data
            label_hats_130, arrival_time_130, feat_node_130, feat_others_130 = [], [], [], []
            sampled_130_idx = get_130_design(bidx, len(feat_map_130_all), options.sample_130_num)
            for i in sampled_130_idx:
                feat_map_130_ = feat_map_130_all[i]
                graph_130_ = graph_130_all[i]
                path_loader_130_ = path_loader_130_all[i]
                path_loader_iter_130_ = path_loader_iter_130_all[i]
                path2level_130_ = path2level_130_all[i]
                path2endpoint_130_ = path2endpoint_130_all[i]
                topo_levels_130_ = topo_levels_130_all[i]
                path_masks_130_ = path_masks_130_all[i]
                # if running out the dataloader, reinit the dataloader
                try:
                    path_ids_130_ = next(path_loader_iter_130_)
                except StopIteration:
                    path_loader_iter_130_ = iter(path_loader_130_)
                    path_loader_iter_130_all[i] = path_loader_iter_130_
                    path_ids_130_ = next(path_loader_iter_130_)
                label_hats_130_, arrival_time_130_, feat_node_130_, feat_others_130_ = path_batch_forward(path_ids_130_,
                                                                                                          path2level_130_,
                                                                                                          path2endpoint_130_,
                                                                                                          topo_levels_130_,
                                                                                                          feat_map_130_,
                                                                                                          model,
                                                                                                          graph_130_,
                                                                                                          path_masks_130_,
                                                                                                          node='130')
                label_hats_130.append(label_hats_130_)
                arrival_time_130.append(arrival_time_130_)
                feat_node_130.append(feat_node_130_)
                feat_others_130.append(feat_others_130_)
            # calculate the reg loss
            train_loss_7 = Loss(label_hats_7, arrival_time_7)
            train_loss_130_all = [Loss(label_hats_130_, arrival_time_130_) for (label_hats_130_, arrival_time_130_) in
                                  zip(label_hats_130, arrival_time_130)]
            total_loss = (train_loss_7 * batch_size_7 + options.loss_weight_130 * sum(
                train_loss_130_all) * batch_size_130) / (batch_size_7 + len(train_loss_130_all) * batch_size_130)

            train_r2_7 = R2_score(label_hats_7, arrival_time_7).to(device)
            train_r2_130 = [R2_score(label_hats_130_, arrival_time_130_) for (label_hats_130_, arrival_time_130_) in
                            zip(label_hats_130, arrival_time_130)]

            # Calculate the alignment loss
            feat_node_130 = th.cat(feat_node_130, dim=0)
            feat_others_130 = th.cat(feat_others_130, dim=0)
            l_clr = contrastive_loss(feat_node_7, feat_node_130, options.con_temp, device, normalization=options.norm_clr)
            l_cmd = cmd_loss(feat_others_7, feat_others_130, k=options.cmd_k)
            total_loss = total_loss + options.weight_clr * l_clr + options.weight_cmd * l_cmd

            # BP
            optim.zero_grad()
            total_loss.backward(retain_graph=not options.not_retain_graph)
            optim.step()
            end_time = time()

            # Update the graph
            reset_graph_feature(graph_7, device)
            for i in sampled_130_idx:
                reset_graph_feature(graph_130_all[i], device)

            training_sets = options.training_set.split(',')
            designs = [training_sets[i] for i in sampled_130_idx]
            designs_info = ",".join(training_sets[i] for i in sampled_130_idx)
            formatted_loss_130 = ["{:.2f}".format(loss.item()) for loss in train_loss_130_all]
            loss_130_msg = ",".join(d + ": " + formatted_loss_130[i] for i, d in enumerate(designs))
            formatted_r2_130 = ["{:.2f}".format(r2.item()) for r2 in train_r2_130]
            r2_130_msg = ",".join(d + ": " + formatted_r2_130[i] for i, d in enumerate(designs))
            print("e{}, sampled 130 designs: {}, b{}/{}, total loss:{:.2f}, 7nm reg loss: {:.2f}, 130nm reg loss: {}, "
                  "clr loss: {:.2f}. cmd loss: {:.2f} 7nm r2:{:.2f}, 130nm r2: {}, batch time: {:.2f}".format(epoch,
                                                                                designs_info, bidx, num_batch_7,
                                                                                total_loss.item(), train_loss_7.item(),
                                                                                loss_130_msg, l_clr.item(), l_cmd.item(),
                                                                                train_r2_7.item(), r2_130_msg,
                                                                                end_time - start_time))
            if epoch < 10:
                flag = bidx % 50 == 0
            elif epoch < 30:
                flag = bidx % 15 == 0
            elif epoch < 50:
                flag = bidx % 10 == 0
            elif epoch < 100:
                flag = bidx % 15 == 0
            else:
                flag = bidx % 5 == 0
            if flag or bidx == num_batch_7 - 1:
                val_res, val_F1_score, val_r2 = validate(val_dataset, device, model, cnn, beta, options)
                if options.task == 'cls':
                    judgement = val_F1_score > max_F1_score
                elif options.task == 'reg':
                    judgement = val_r2 > max_r2
                else:
                    assert False
                # judgement = True
                if judgement:
                    stop_score = 0
                    max_F1_score = val_F1_score
                    max_r2 = val_r2
                    print("Saving model.... ", os.path.join(options.model_saving_dir))
                    if os.path.exists(options.model_saving_dir) is False:
                        os.makedirs(options.model_saving_dir)
                    with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
                        parameters = options
                        pickle.dump((parameters, model, cnn), f)
                    print("Model successfully saved")


if __name__ == "__main__":
    options = get_options()
    seed = options.seed
    # seed = random.randint(1, 10000)
    # seed = 9294
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # if options.reproducibility:
    #     print(f'---------Reproducibility setting')
    #     th.backends.cudnn.benchmark = False
    #     th.backends.cudnn.deterministic = True
    #     os.environ["PYTHONHASHSEED"] = str(seed)
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    os.makedirs(copilot_log_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")
    stdout_f = '{}/stdout.log'.format(options.model_saving_dir)
    stderr_f = '{}/stderr.log'.format(options.model_saving_dir)
    os.makedirs(options.model_saving_dir, exist_ok=True)
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        train(options, seed)

