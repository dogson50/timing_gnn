r"""
this script is used to train and validate the models
"""
import os
import random
import pickle
import re
import numpy as np
import torch as th
import pandas as pd
import tee

from torch.utils.data import DataLoader, Dataset

from types import SimpleNamespace

from options import get_options

from model import DisentangledRegressor
from hgat import HGATDesignEncoder, build_dgl_graph_from_devs
from spi2graph import parse_transistors_spice, parse_top_subckt_pins
from losses import total_loss

# NOTE:
# This repo originally used train.py for verilog/layout/path-based training.
# Your current task only has lib+sp data, so we add a new mode: --mode cell_delay
# which routes training through HGAT (see train_hgat.py) and dataset.pkl.

# Legacy globals kept as placeholders to avoid NameError if legacy code paths are accidentally called.
ctype2id = {}
num_ctypes = 0
idx2design = {}
R2_score = None
Loss = None


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
    mlp = MLP2(
        in_dim=mlp_dim,
        # in_dim = options.out_dim,
        out_dim=options.nlabels,
        nlayers=options.n_fcn,
        dropout=options.mlp_dropout
    )

    # fcn=None
    model = PathModel(gnn, fcn, mlp)
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
    model = model.to(device)
    if cnn is not None:
        cnn = cnn.to(device)
    # model = nn.DataParallel(model, device_ids=[0,1])
    # cnn = nn.DataParallel(cnn, device_ids=[0,1])

    print('Model and hyper-parameters successfully loaded!')
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

                    cur_label_hats = model(graph, nodes, eids, targets, level_id, path_map)

                    if len(paths) == 0:
                        continue

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


def transform_cellfeat(cell_feat):
    new_cellfeat = th.zeros((cell_feat.shape[0], cell_feat.shape[1] - num_ctypes + 4))
    cell_type = cell_feat[:, :num_ctypes]
    cell_type = th.argmax(cell_type, dim=1, keepdim=True)
    id2celltype = {}
    for cell, id in ctype2id.items():
        id2celltype[id] = cell

    # new celltype feat:
    #   buf: 1000
    #   inv: 0100
    #   register: 0010
    #   others: 0001
    {"AND": 0, "FA": 1, "HA": 2, "MAJI": 3, "MAJ": 4, "NAND": 5, "NOR": 6, "OR": 7, "TIEHI": 8,
     "TIELO": 9, "XNOR": 10, "XOR": 11, "ASYNC_DFFH": 12, "DFFHQN": 13, "DFFHQ": 14, "DFFLQN": 15,
     "DFFLQ": 16, "DHL": 17, "DLL": 18, "ICG": 19, "SDFH": 20, "SDFL": 21, "BUF": 22, "CKINVDC": 23,
     "HB": 24, "INV": 25, "O2A1O1I": 26, "OA": 27, "OAI": 28, "A2O1A1I": 29, "A2O1A1O1I": 30, "AO": 31, "AOI": 32}
    for i in range(cell_feat.shape[0]):
        type = id2celltype[cell_type[i].item()]
        new_cellfeat[i][0] = type == 'BUFF'
        new_cellfeat[i][1] = type in ('INV', 'CKINVDC')
        new_cellfeat[i][2] = type in ('TIEHI', "TIELO", "ASYNC_DFFH", "DFFHQN", "DFFHQ",
                                      "DFFLQN", "DFFLQ", "DHL", "DLL", "ICG", "SDFH",
                                      "SDFL", "HB")
        new_cellfeat[i][3] = not (new_cellfeat[i][0] and new_cellfeat[i][1] and new_cellfeat[i][2])
        new_cellfeat[i][4:] = cell_feat[i][num_ctypes:]
        print(type, new_cellfeat[i])
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


def load_data(data_path, usage, init_feat_dim, os_rate, feat_reduce, if_norm, design_set=None):
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


def train(options, seed):
    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

    # you can define your dataset file here
    data_save_path = options.data_save_path
    print(data_save_path)

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

    train_dataset = load_data(data_save_path, 'train', options.out_dim, options.os_rate, options.feat_reduce,
                              options.norm, design_set=options.training_set)
    # val_data_file = os.path.join(data_save_path, 'test.pkl')
    val_dataset = load_data(data_save_path, 'test', options.out_dim, options.os_rate, options.feat_reduce,
                            options.norm, design_set=options.test_set)

    # split the validation set and test set

    beta = options.beta
    # set the optimizer
    if cnn is not None:
        optim = th.optim.Adam(
            itertools.chain(model.parameters(), cnn.parameters()),
            # model.parameters(),
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

    print("----------------Start training---------------")
    pre_loss = 100
    stop_score = 0
    max_F1_score, max_r2 = 0, 0
    print(f'index2design: {idx2design}')
    th.autograd.set_detect_anomaly(True)
    for epoch in range(options.num_epoch):
        runtime = 0
        total_num, total_loss, correct, fn, fp, tn, tp, total_r2 = 0, 0.0, 0, 0, 0, 0, 0, 0
        # total_nodes = []
        # shuffle(train_dataset)
        for i, (path_dataset, graph, path2level, path2endpoint, topo_levels, cnn_inputs, path_masks) in enumerate(
                train_dataset):
            # for i,(path_dataset,graph,path2level,path2endpoint,topo_levels,cnn_inputs,path_masks) in enumerate(random.sample(train_dataset,len(train_dataset))):
            # feat_map = th.ones((128,128)).to(device)
            design = idx2design[i]
            feat_map = cnn(cnn_inputs.to(device)).reshape((1, -1)) if cnn is not None else None
            # feat_map = feat_map.to_sparse()
            # path_masks = path_masks.to_sparse()
            # print(feat_map.shape)
            # optim.zero_grad()
            # transfer the data to GPU
            # graph = graph.to(device)
            graph = graph.to(device)
            # print(graph)
            if len(path2level) <= options.batch_size:
                path_loader = DataLoader(path_dataset, batch_size=options.batch_size, shuffle=True, drop_last=False)
            else:
                path_loader = DataLoader(path_dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)

            # first_level_nodes = topo_levels[0][0]
            # init_message = graph.ndata['h'][first_level_nodes]
            num_batch = len(path_loader)
            for bidx, path_ids in enumerate(path_loader):
                path_ids = list(path_ids.numpy().tolist())
                # endpoints = [path2endpoint[pid] for pid in path_ids]
                # endpoints = endpoints.numpy().tolist()
                # path_ids2 = [endpoints2path[nd] for nd in endpoints]
                # assert set(path_ids) == set(path_ids2)
                sampled_ends, sampled_paths = {}, {}
                for i, pathid in enumerate(path_ids):
                    level = path2level[pathid]
                    endpoint = path2endpoint[pathid]
                    # assert endpoints2path[endpoint] == pathid
                    sampled_ends[level] = sampled_ends.get(level, [])
                    sampled_ends[level].append(endpoint)
                    sampled_paths[level] = sampled_paths.get(level, [])
                    sampled_paths[level].append(pathid)

                count_target = 0
                label_hats = None
                target_list = []
                # please start from the first level!!!!
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
                    cur_label_hats = model(graph, nodes, eids, targets, level_id, path_map)

                    if len(paths) == 0:
                        continue

                    if label_hats is None:
                        label_hats = cur_label_hats
                    else:
                        label_hats = th.cat((label_hats, cur_label_hats), dim=0)
                # print(len(target_list),target_list[:10])

                labels = graph.ndata['label'][target_list].squeeze()
                # predicted labels
                # label_hats = mlp(target_embeddings)
                if options.task == 'cls':
                    predict_labels = th.argmax(nn.functional.softmax(label_hats, 1), dim=1)
                    train_loss = Loss(label_hats, labels)
                    train_r2 = 0
                elif options.task == 'reg':
                    required_time = graph.ndata['required_time'][target_list].squeeze()
                    arrival_time = graph.ndata['arrival_time'][target_list].squeeze()
                    train_loss = Loss(label_hats, arrival_time)
                    predict_labels = judge_critical(label_hats, required_time).to(device)
                    train_r2 = R2_score(label_hats, arrival_time).to(device)
                    total_r2 = train_r2.item() * len(labels)
                # calculate loss
                total_num = len(labels)
                total_loss = train_loss.item() * len(labels)
                # print('loss: {}, R2 score: {}'.format(train_loss.item(),train_r2))

                # calculate accuracy
                correct = (
                        predict_labels == labels
                ).sum().item()

                # calculate accuracy, recall, precision and F1-score
                acc = correct / total_num
                # calculate fake negative, true positive, fake negative, and true negative rate
                fn = ((predict_labels == 0) & (labels != 0)).sum().item()
                tp = ((predict_labels != 0) & (labels != 0)).sum().item()
                tn = ((predict_labels == 0) & (labels == 0)).sum().item()
                fp = ((predict_labels != 0) & (labels == 0)).sum().item()
                recall = 0
                precision = 0
                if tp != 0:
                    recall = tp / (tp + fn)
                    precision = tp / (tp + fp)
                F1_score = 0
                if precision != 0 or recall != 0:
                    F1_score = 2 * recall * precision / (recall + precision)

                # back-propagate
                optim.zero_grad()
                train_loss.backward(retain_graph=True)
                # print(model.GCN1.layers[0].attn_n.grad)
                optim.step()

                target_list = []
                label_hats = None
                graph.ndata['h'] = th.zeros((graph.number_of_nodes(), options.out_dim), dtype=th.float).to(device)
                graph.edges['cell'].data['a'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(
                    device)
                graph.edges['cell'].data['e'] = th.zeros((graph.number_of_edges(etype='cell'), 1), dtype=th.float).to(
                    device)
                feat_map = cnn(cnn_inputs.to(device)).reshape((1, -1)) if cnn is not None else None
                # if feat_map is not None: feat_map = feat_map.reshape((1,feat_map[0]*feat_map[1]))
                end = time()
                print("e{},{},b{}/{}, l:{:.3f}, r2:{:.3f}, r:{:.3f}, F1:{:.3f}".format(epoch, design, bidx, num_batch,
                                                                                       train_loss.item(),
                                                                                       train_r2.item(), recall,
                                                                                       F1_score))
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
                if flag or bidx == num_batch - 1:
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

        # end = time()
        # runtime += end-start
        # Train_loss = total_loss / total_num
        # Train_r2 = total_r2 / total_num
        # # calculate accuracy, recall, precision and F1-score
        # Train_acc = correct / total_num
        # Train_recall = 0
        # Train_precision = 0
        # if tp != 0:
        #     Train_recall = tp / (tp + fn)
        #     Train_precision = tp / (tp + fp)
        # Train_F1_score = 0
        # if Train_precision != 0 or Train_recall != 0:
        #     Train_F1_score = 2 * Train_recall * Train_precision / (Train_recall + Train_precision)

        # print("Task: {}, epoch[{:d}]".format('classification' if options.task=='cls' else 'regression', epoch))
        # print("training runtime: ",runtime)
        # print("  train:")
        # # print("\ttp:", tp, " fp:", fp, " fn:", fn, " tn:", tn, " precision:", round(Train_precision,3))
        # print("loss:{:.8f}, r2:{:.3f}, acc:{:.3f}, recall:{:.3f}, F1 score:{:.3f}".format(Train_loss,Train_r2,Train_acc,Train_recall,Train_F1_score))

        # validate
        # print("  validate:")
        # val_res,val_F1_score,val_r2 = validate(val_dataset, device, model,cnn,beta,options)
        # # print("  test:")
        # # validate(testdataloader, label_name, device, model,
        # #          Loss, beta, options)

        # # save the result of current epoch
        # with open(os.path.join(options.model_saving_dir, 'res.txt'), 'a') as f:
        #     f.write(str(round(Train_loss, 8)) + " " + str(round(Train_r2, 3)) + " " + str(round(Train_acc, 3)) + " " + str(
        #         round(Train_recall, 3)) + " " + str(round(Train_precision,3))+" " + str(round(Train_F1_score, 3)) + "\n")
        #     for res in val_res:
        #         f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(res[0], res[1],res[2], res[3], res[4], res[5]) + "\n")
        #     f.write('\n')

        # if options.task == 'cls':
        #     judgement = val_F1_score > max_F1_score 
        # elif options.task == 'reg':
        #     judgement = val_r2 > max_r2
        # else:
        #     assert False
        # #judgement = True
        # if judgement:
        #    stop_score = 0
        #    max_F1_score = val_F1_score
        #    max_r2 = val_r2
        #    print("Saving model.... ", os.path.join(options.model_saving_dir))
        #    if os.path.exists(options.model_saving_dir) is False:
        #       os.makedirs(options.model_saving_dir)
        #    with open(os.path.join(options.model_saving_dir, 'model.pkl'), 'wb') as f:
        #       parameters = options
        #       pickle.dump((parameters, model,cnn), f)
        #    print("Model successfully saved")
        # else:
        #     stop_score += 1
        #     if stop_score >= 5:
        #         print('Early Stop!')
        #         exit(0)


def _seed_everything(seed: int):
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


NUMERIC_COLS = [
    "slew", "cap", "voltage", "temp",
    "wp_over_wn", "wp_sum", "wn_sum",
    "is_inv",
    "log_slew", "log_cap",
    "req_p", "req_n",
    "rc_p", "rc_n",
    "rc_eff", "req_eff",
    "inv_v", "inv_temp",
    "pn_balance",
    "pol_bit",
]
TARGET_COL = "delay"


def _ensure_pol_bit(df):
    if "pol_bit" not in df.columns:
        if "pol" in df.columns:
            df["pol_bit"] = (df["pol"].astype(str) == "rise").astype(np.float32)
        else:
            df["pol_bit"] = 0.0
    return df


def _ensure_numeric_cols(df):
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def _norm_xy(df, x_mean, x_std, y_mean, y_std):
    df = _ensure_pol_bit(df.copy())
    df = _ensure_numeric_cols(df)
    x_raw = df[NUMERIC_COLS].fillna(0.0).astype(np.float32).values
    x = (x_raw - x_mean) / x_std
    y_raw = df[TARGET_COL].astype(np.float32).values
    y = (y_raw - y_mean) / y_std
    cts = df["cell_type"].astype(str).values
    return x, y, cts


class PklDataset(Dataset):
    def __init__(self, df, x_mean, x_std, y_mean, y_std):
        self.df = df.reset_index(drop=True)
        self.x, self.y, self.cell_types = _norm_xy(self.df, x_mean, x_std, y_mean, y_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return th.from_numpy(self.x[i]), th.tensor(self.y[i]), self.cell_types[i]


def _load_dataset_pkl(pkl_path: str) -> dict:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing dataset.pkl: {pkl_path}")
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def _load_scalers_from_pkl(dataset: dict):
    stats = dataset.get("scaler_stats") or {}
    yinfo = dataset.get("y_scaler") or {}
    x_mean = np.array([stats.get("mean", {}).get(c, 0.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.array([stats.get("std", {}).get(c, 1.0) for c in NUMERIC_COLS], dtype=np.float32)
    x_std = np.where(x_std < 1e-12, 1.0, x_std).astype(np.float32)
    y_mean = float(yinfo.get("mean", 0.0))
    y_std = float(yinfo.get("std", 1.0))
    if abs(y_std) < 1e-12:
        y_std = 1.0
    return x_mean, x_std, y_mean, y_std, stats, yinfo


def _infer_hgat_hid(sd: dict) -> int:
    if sd is None:
        return 64
    for k in ("embed.NET.weight", "embed.PMOS.weight", "embed.NMOS.weight"):
        if k in sd and sd[k].dim() == 2:
            return sd[k].shape[0]
    return 64


def extract_subckt_text(sp_text: str, subckt_name: str) -> str:
    lines = sp_text.splitlines(keepends=True)
    collecting = False
    buf = []
    patt_begin = re.compile(r"\s*\.subckt\s+%s\b" % re.escape(subckt_name), re.IGNORECASE)
    patt_end = re.compile(r"\s*\.ends\b", re.IGNORECASE)
    for line in lines:
        if not collecting:
            if patt_begin.search(line):
                collecting = True
                buf.append(line)
        else:
            buf.append(line)
            if patt_end.match(line):
                break
    return "".join(buf) if buf else ""


def _precompute_z_from_src_map(src_map, enc, device, base_dir=None):
    graph_cache = {}
    if not src_map:
        return graph_cache
    for ct, path in src_map.items():
        full_path = path
        if base_dir and not os.path.exists(full_path):
            cand = os.path.join(base_dir, path)
            if os.path.exists(cand):
                full_path = cand
        if not os.path.exists(full_path):
            continue
        txt = open(full_path, 'r', encoding="utf-8", errors="ignore").read()
        devs = parse_transistors_spice(txt)
        _, pins = parse_top_subckt_pins(txt)
        if not devs:
            continue
        g, feats, _ = build_dgl_graph_from_devs(devs, pins)
        graph_cache[ct] = (g.to(device), {k: v.to(device) for k, v in feats.items()})
    return graph_cache


def _precompute_z_from_tgt_spice(tgt_spice, subckt_map, enc, device, base_dir=None):
    if not tgt_spice or not subckt_map:
        return {}
    if not os.path.exists(tgt_spice) and base_dir:
        cand = os.path.join(base_dir, tgt_spice)
        if os.path.exists(cand):
            tgt_spice = cand
    if not os.path.exists(tgt_spice):
        raise FileNotFoundError(f"Target SPICE not found: {tgt_spice}")

    sp_text = open(tgt_spice, "r", encoding="utf-8", errors="ignore").read()
    z_dict = {}
    enc.eval()
    with th.no_grad():
        for ctype, sub_name in subckt_map.items():
            sub_txt = extract_subckt_text(sp_text, sub_name)
            if not sub_txt:
                continue
            devs = parse_transistors_spice(sub_txt)
            _, pins = parse_top_subckt_pins(sub_txt)
            if not devs:
                continue
            g, feats, _ = build_dgl_graph_from_devs(devs, pins)
            g = g.to(device)
            feats = {k: v.to(device) for k, v in feats.items()}
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_dict[ctype] = z
    return z_dict


@th.no_grad()
def _eval_mae_on_dataset(dl: DataLoader, enc, model, z_provider, y_mean, y_std, device, design_dim):
    enc.eval()
    model.eval()
    err_sum = 0.0
    n = 0
    for xb, yb, cts in dl:
        xb = xb.to(device)
        yb = yb.to(device)

        z_list = []
        for ct in cts:
            z = z_provider(ct)
            if z is None:
                z_list.append(th.zeros(1, design_dim, device=device))
            else:
                z_list.append(z)
        zb = th.cat(z_list, dim=0)

        mu, _, _, _ = model(xb, zb)
        max_abs = 10.0
        mu_t = max_abs * th.tanh(mu / max_abs)

        pred_ps = mu_t * y_std + y_mean
        true_ps = yb * y_std + y_mean
        err_sum += th.abs(pred_ps - true_ps).sum().item()
        n += len(xb)
    return err_sum / max(1, n)


def _split_src_by_celltype(df, val_ratio=0.1, seed=42):
    if val_ratio <= 0:
        return df, None
    rng = np.random.RandomState(seed)
    cell_types = df["cell_type"].astype(str).unique().tolist()
    rng.shuffle(cell_types)
    n_val = max(1, int(len(cell_types) * val_ratio))
    val_ct = set(cell_types[:n_val])

    is_val = df["cell_type"].astype(str).isin(val_ct)
    df_val = df[is_val].reset_index(drop=True)
    df_tr = df[~is_val].reset_index(drop=True)
    return df_tr, df_val


def run_stage1_pretraining_pkl(df_src, src_map, data_dir, save_dir, device,
                               x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler,
                               hid, design_dim, s1_epochs, lr, auto_lr, lr_patience, lr_factor,
                               min_lr, early_patience, grad_clip, s1_kl, src_val_ratio, seed):
    print("\n" + "=" * 60)
    print(" >>> STAGE 1: Source Domain Pre-training (PKL) <<<")
    print("=" * 60)

    df_tr, df_val = _split_src_by_celltype(df_src, val_ratio=src_val_ratio, seed=seed)
    if df_val is not None:
        print(f"[S1] Split src by cell_type: train_rows={len(df_tr)} val_rows={len(df_val)} (val_ratio={src_val_ratio})")
    else:
        print(f"[S1] No src val split (src_val_ratio=0). Use train loss as best criterion. train_rows={len(df_tr)}")

    train_ds = PklDataset(df_tr, x_mean, x_std, y_mean, y_std)
    val_ds = PklDataset(df_val, x_mean, x_std, y_mean, y_std) if df_val is not None else None

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=my_collate) if val_ds else None

    in_map = {"NET": 4, "PMOS": 2, "NMOS": 2}
    enc = HGATDesignEncoder(in_dim_map=in_map, hid=hid, out=design_dim).to(device)
    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=hid, design_dim_override=design_dim).to(device)

    optimizer = th.optim.Adam(list(enc.parameters()) + list(model.parameters()), lr=lr)

    scheduler = None
    if auto_lr:
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
        )

    print("[S1] Caching Source Graphs ...")
    graph_cache = _precompute_z_from_src_map(src_map, enc, device, base_dir=data_dir)

    def z_provider(ct):
        if ct in graph_cache:
            g, feats = graph_cache[ct]
            z = enc(g, feats)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return z
        return None

    best_metric = float("inf")
    bad_epochs = 0

    best_path = os.path.join(save_dir, "ckpt_stage1_best.pt")
    last_path = os.path.join(save_dir, "ckpt_stage1_last.pt")

    for epoch in range(s1_epochs):
        enc.train()
        model.train()
        epoch_loss_val = 0.0
        count = 0

        for xb, yb, cts in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            z_list = []
            valid = []
            for i, ct in enumerate(cts):
                if ct in graph_cache:
                    g, feats = graph_cache[ct]
                    z = enc(g, feats)
                    if z.dim() == 1:
                        z = z.unsqueeze(0)
                    z_list.append(z)
                    valid.append(i)
            if not z_list:
                continue

            zb = th.cat(z_list, dim=0)
            xb = xb[valid]
            yb = yb[valid]

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=s1_kl)

            loss.backward()
            if grad_clip and grad_clip > 0:
                th.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(model.parameters()), grad_clip)
            optimizer.step()

            epoch_loss_val += loss.item() * len(yb)
            count += len(yb)

        avg_loss = epoch_loss_val / (count + 1e-6)

        if val_dl is not None:
            src_val_mae = _eval_mae_on_dataset(val_dl, enc, model, z_provider, y_mean, y_std, device, design_dim)
            metric = src_val_mae
        else:
            src_val_mae = None
            metric = avg_loss

        if scheduler is not None:
            scheduler.step(metric)

        cur_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0:
            if src_val_mae is None:
                print(f"  [S1] Ep {epoch+1:3d}/{s1_epochs} | TrainLoss {avg_loss:.4f} | LR {cur_lr:.2e}")
            else:
                print(f"  [S1] Ep {epoch+1:3d}/{s1_epochs} | TrainLoss {avg_loss:.4f} | SrcValMAE {src_val_mae:.4f} ps | LR {cur_lr:.2e}")

        th.save({
            "model": model.state_dict(),
            "enc": enc.state_dict(),
            "hgat_in_dim_map": in_map,
            "design_dim": design_dim,
            "scaler_stats": scaler_stats,
            "y_scaler": y_scaler,
            "stage": "stage1_last",
            "epoch": epoch + 1,
        }, last_path)

        if metric < best_metric:
            best_metric = metric
            bad_epochs = 0
            th.save({
                "model": model.state_dict(),
                "enc": enc.state_dict(),
                "hgat_in_dim_map": in_map,
                "design_dim": design_dim,
                "scaler_stats": scaler_stats,
                "y_scaler": y_scaler,
                "stage": "stage1_best",
                "epoch": epoch + 1,
                "best_metric": best_metric,
                "criterion": "src_val_mae" if val_dl is not None else "train_loss",
            }, best_path)
        else:
            bad_epochs += 1
            if early_patience and early_patience > 0 and bad_epochs >= early_patience:
                print(f"  [S1] Early stop at ep={epoch+1} (no improve for {bad_epochs} epochs).")
                break

    print(f"[S1] Best saved: {best_path}")
    print(f"[S1] Last saved: {last_path}")
    return best_path


def run_stage2_transfer_pkl(df_train, df_val, tgt_subckt_map, tgt_spice, data_dir, save_dir, device,
                             src_ckpt_path, x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler,
                             hid, s2_epochs, lr, auto_lr, lr_patience, lr_factor, min_lr, early_patience,
                             grad_clip, s2_kl):
    print("\n" + "=" * 60)
    print(" >>> STAGE 2: Target Fine-tuning (PKL) <<<")
    print("=" * 60)

    if not os.path.exists(src_ckpt_path):
        raise FileNotFoundError(f"Missing src ckpt: {src_ckpt_path}")

    state = th.load(src_ckpt_path, map_location=device)
    design_dim = int(state.get("design_dim", 64))
    enc_hid = _infer_hgat_hid(state.get("enc"))
    in_map = state.get("hgat_in_dim_map", {"NET": 4, "PMOS": 2, "NMOS": 2})

    enc = HGATDesignEncoder(in_dim_map=in_map, hid=enc_hid, out=design_dim).to(device)
    enc.load_state_dict(state["enc"], strict=True)

    model = DisentangledRegressor(in_dim=len(NUMERIC_COLS), hid=hid, design_dim_override=design_dim).to(device)
    model.load_state_dict(state["model"], strict=False)

    print("[S2] Fine-tuning HGAT encoder (Unfrozen).")
    optimizer = th.optim.Adam([
        {'params': model.parameters(), 'lr': lr * 0.5},
        {'params': enc.parameters(), 'lr': lr * 0.01}
    ])
    scheduler = None
    if auto_lr:
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
        )

    z_map = _precompute_z_from_tgt_spice(tgt_spice, tgt_subckt_map, enc, device, base_dir=data_dir)

    train_ds = PklDataset(df_train, x_mean, x_std, y_mean, y_std)
    val_ds = PklDataset(df_val, x_mean, x_std, y_mean, y_std) if df_val is not None and len(df_val) > 0 else None

    def my_collate(batch):
        xs, ys, cts = zip(*batch)
        return th.stack(xs), th.stack(ys), cts

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=my_collate)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=my_collate) if val_ds else None

    print(f"[S2] Train rows={len(train_ds)} | Val rows={len(val_ds) if val_ds else 0}")

    def z_provider(ct):
        return z_map.get(ct, None)

    best_mae = float("inf")
    bad_epochs = 0

    best_path = os.path.join(save_dir, "ckpt_transfer_best.pt")
    last_path = os.path.join(save_dir, "ckpt_transfer_last.pt")

    for epoch in range(s2_epochs):
        model.train()
        enc.train()
        epoch_loss = 0.0

        for xb, yb, cts in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            z_list = []
            for ct in cts:
                z = z_map.get(ct, None)
                if z is None:
                    z_list.append(th.zeros(1, design_dim, device=device))
                else:
                    z_list.append(z)
            zb = th.cat(z_list, dim=0)

            optimizer.zero_grad()
            mu, logv, z_q, z_p = model(xb, zb)
            loss, _, _ = total_loss(yb, mu, logv, z_q, z_p, kl_weight=s2_kl)
            loss.backward()
            if grad_clip and grad_clip > 0:
                th.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        avg_train_loss = epoch_loss / max(1, len(train_ds))

        val_mae = None
        if val_dl is not None:
            val_mae = _eval_mae_on_dataset(val_dl, enc, model, z_provider, y_mean, y_std, device, design_dim)
            metric = val_mae
        else:
            metric = avg_train_loss

        if scheduler is not None:
            scheduler.step(metric)

        cur_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 10 == 0:
            if val_mae is None:
                print(f"  [S2] Ep {epoch+1:3d}/{s2_epochs} | TrainLoss {avg_train_loss:.4f} | LR {cur_lr:.2e}")
            else:
                print(f"  [S2] Ep {epoch+1:3d}/{s2_epochs} | TrainLoss {avg_train_loss:.4f} | ValMAE {val_mae:.4f} ps | LR {cur_lr:.2e}")

        th.save({
            "model": model.state_dict(),
            "enc": enc.state_dict(),
            "hgat_in_dim_map": in_map,
            "design_dim": design_dim,
            "scaler_stats": scaler_stats,
            "y_scaler": y_scaler,
            "stage": "stage2_last",
            "epoch": epoch + 1,
        }, last_path)

        if val_mae is not None and val_mae < best_mae:
            best_mae = val_mae
            bad_epochs = 0
            th.save({
                "model": model.state_dict(),
                "enc": enc.state_dict(),
                "hgat_in_dim_map": in_map,
                "design_dim": design_dim,
                "scaler_stats": scaler_stats,
                "y_scaler": y_scaler,
                "stage": "stage2_best",
                "epoch": epoch + 1,
                "best_metric": best_mae,
                "criterion": "val_mae",
            }, best_path)
        else:
            bad_epochs += 1
            if early_patience and early_patience > 0 and bad_epochs >= early_patience:
                print(f"  [S2] Early stop at ep={epoch+1} (no improve for {bad_epochs} epochs).")
                break

    print(f"[S2] Best saved: {best_path}")
    print(f"[S2] Last saved: {last_path}")
    return best_path


def train_cell_delay(options):
    """Train delay predictor using lib+sp features + HGAT (see train_hgat.py)."""
    data_dir = options.data_save_path
    save_dir = options.model_saving_dir
    os.makedirs(save_dir, exist_ok=True)

    device = th.device(f"cuda:{options.gpu}" if th.cuda.is_available() else "cpu")

    dataset_pkl = getattr(options, "dataset_pkl", None) or os.path.join(data_dir, "dataset.pkl")
    dataset = _load_dataset_pkl(dataset_pkl)
    meta = dataset.get("meta", {})

    tgt_spice = getattr(options, "tgt_spice", None) or meta.get("tgt_sp_file") or getattr(options, "tgt_sp", None)
    if not tgt_spice:
        raise FileNotFoundError("Missing target SPICE. Set --tgt_spice or ensure meta.tgt_sp_file exists in dataset.pkl")

    print("[Info] Cell-delay mode (HGAT + PKL)")
    print(f"  data_dir: {data_dir}")
    print(f"  save_dir: {save_dir}")
    print(f"  dataset_pkl: {dataset_pkl}")
    print(f"  tgt_spice: {tgt_spice}")
    print(f"  device: {device}")

    print("[Info] Loading scalers from dataset.pkl...")
    x_mean, x_std, y_mean, y_std, scaler_stats, y_scaler = _load_scalers_from_pkl(dataset)

    mode = getattr(options, "cell_train_mode", "all")
    src_ckpt = getattr(options, "src_ckpt", None)
    hid = getattr(options, "hgat_hid", 64)
    design_dim = getattr(options, "hgat_design_dim", 64)
    s1_epochs = getattr(options, "s1_epochs", 200)
    s2_epochs = getattr(options, "s2_epochs", 200)
    auto_lr = getattr(options, "auto_lr", False)
    lr_patience = getattr(options, "lr_patience", 10)
    lr_factor = getattr(options, "lr_factor", 0.5)
    min_lr = getattr(options, "min_lr", 1e-6)
    early_patience = getattr(options, "early_patience", 0)
    grad_clip = getattr(options, "grad_clip", 0.0)
    s1_kl = getattr(options, "s1_kl", 0.0)
    s2_kl = getattr(options, "s2_kl", 0.0)
    src_val_ratio = getattr(options, "src_val_ratio", 0.1)

    df_src = dataset.get("src")
    df_tgt_train = dataset.get("tgt_train")
    df_tgt_val = dataset.get("tgt_val")

    src_map = meta.get("src_spi_by_cell", {})
    tgt_subckt_map = meta.get("tgt_subckt_by_cell", {})

    current_ckpt = src_ckpt
    if not current_ckpt:
        cand = os.path.join(save_dir, "ckpt_stage1_best.pt")
        if os.path.exists(cand):
            current_ckpt = cand

    if mode in ["pretrain", "all"]:
        if df_src is None or len(df_src) == 0:
            print("[Warn] No src data in dataset.pkl. Skip stage1 pretrain.")
        else:
            current_ckpt = run_stage1_pretraining_pkl(
                df_src=df_src,
                src_map=src_map,
                data_dir=data_dir,
                save_dir=save_dir,
                device=device,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                scaler_stats=scaler_stats,
                y_scaler=y_scaler,
                hid=hid,
                design_dim=design_dim,
                s1_epochs=s1_epochs,
                lr=options.learning_rate,
                auto_lr=auto_lr,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                min_lr=min_lr,
                early_patience=early_patience,
                grad_clip=grad_clip,
                s1_kl=s1_kl,
                src_val_ratio=src_val_ratio,
                seed=options.seed,
            )

    if mode in ["transfer", "all"]:
        if df_tgt_train is None or len(df_tgt_train) == 0:
            raise RuntimeError("No tgt_train data in dataset.pkl. Cannot run transfer.")
        if not current_ckpt or not os.path.exists(current_ckpt):
            raise FileNotFoundError(
                "Missing stage1 checkpoint for transfer. "
                "Run with --cell_train_mode pretrain/all or pass --src_ckpt."
            )
        run_stage2_transfer_pkl(
            df_train=df_tgt_train,
            df_val=df_tgt_val,
            tgt_subckt_map=tgt_subckt_map,
            tgt_spice=tgt_spice,
            data_dir=data_dir,
            save_dir=save_dir,
            device=device,
            src_ckpt_path=current_ckpt,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            scaler_stats=scaler_stats,
            y_scaler=y_scaler,
            hid=hid,
            s2_epochs=s2_epochs,
            lr=options.learning_rate,
            auto_lr=auto_lr,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            min_lr=min_lr,
            early_patience=early_patience,
            grad_clip=grad_clip,
            s2_kl=s2_kl,
        )


if __name__ == "__main__":
    options = get_options()
    _seed_everything(options.seed)
    stdout_f = os.path.join(options.model_saving_dir, "stdout.log")
    stderr_f = os.path.join(options.model_saving_dir, "stderr.log")
    os.makedirs(options.model_saving_dir, exist_ok=True)
    copilot_log_dir = os.path.join(os.getcwd(), "copilot_train_logs")
    os.makedirs(copilot_log_dir, exist_ok=True)
    script_stem = os.path.splitext(os.path.basename(__file__))[0]
    copilot_log_f = os.path.join(copilot_log_dir, f"{script_stem}.log")

    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f), tee.StdoutTee(copilot_log_f), tee.StderrTee(copilot_log_f):
        if options.mode == "cell_delay":
            train_cell_delay(options)
        else:
            raise SystemExit(
                "[error] Legacy '--mode path' training depends on verilog/layout/path data and was intentionally not wired for this workspace. "
                "Use --mode cell_delay for lib+sp cell delay modeling."
            )
