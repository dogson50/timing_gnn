r"""
this script is used to train and validate the models
"""
import os
import torch as th
import argparse
import json
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = {
    '7': 'datasets/asap7-designs',
    '130': 'datasets/130-designs'
}


def load_ctype(node):
    if node == '130':
        with open('./rawdata-130/cell_info_map.json', 'r') as f:
            cell_info_map = json.load(f)
        with open('./rawdata-130/ctype2id.json', 'r') as f:
            ctype2id = json.load(f)
    elif node == '7':
        with open('./rawdata/cell_info_map.json', 'r') as f:
            cell_info_map = json.load(f)
        with open('./rawdata/ctype2id.json', 'r') as f:
            ctype2id = json.load(f)
    else:
        raise ValueError
    ctype2id['SRAM'] = len(ctype2id)
    num_ctypes = len(ctype2id)
    # reverse ctype2id
    id2ctype = {}
    for c, i in ctype2id.items():
        assert i not in id2ctype.keys()
        id2ctype[i] = c
    print(f'id2ctype: {id2ctype}')
    return id2ctype


def extract_design_info(design, data_path):
    dataset_file = os.path.join(data_path, '{}.pkl'.format(design))
    graph, topo_levels, path_masks, path2level, path2endpoint, critical_paths, cnn_inputs = th.load(dataset_file)
    cell_feat = graph.nodes['pin'].data['cell_feat']
    cell_feat = cell_feat[:, :-8]
    n_nodes, feat_dim = cell_feat.shape
    #print(f'cell feat shape: {cell_feat.shape}')
    #print(f'sum dim=0')
    cell_type_dist = th.sum(cell_feat, dim=0)
    n_cell_nodes = th.sum(cell_type_dist)
    cell_type_dist = cell_type_dist / n_cell_nodes
    #print(f'normalized cell type dist: {cell_type_dist} {th.sum(cell_type_dist)}')
    #print(f'max cell: {th.argmax(cell_type_dist)}')
    return cell_type_dist


def cal_dist(sd, td, metric='kl'):
    if metric == 'kl':
        sd = F.softmax(sd, dim=1)
        td = F.softmax(td, dim=1)
        return F.kl_div(td, sd)
    elif metric == 'cos':
        return sd @ td.t()


def vis_dist_comp(sd, td, sd_name, td_name, node='130'):
    fig_name = f'./vis_comp_design/node-{node}-source-{sd_name}-target-td-{td_name}.png'
    _, len_x = sd.shape
    sd = sd.squeeze(0).numpy()
    td = td.squeeze(0).numpy()
    x = np.arange(len_x)
    # Create the plot
    plt.figure()
    # Plot a
    plt.plot(x, sd, label='source')
    # Plot b
    plt.plot(x, td, label='target')
    # Add a legend
    plt.legend()
    plt.savefig(fig_name)


def extract_design_output(design, data_path):
    dataset_file = os.path.join(data_path, '{}.pkl'.format(design))
    graph, topo_levels, path_masks, path2level, path2endpoint, critical_paths, cnn_inputs = th.load(dataset_file)
    arrival_time = graph.nodes['pin'].data['arrival_time']
    end = graph.nodes['pin'].data['end']
    print(f'arrival time shape: {arrival_time.shape}')
    print(f'end shape: {end.shape}')
    indices = th.where(end == 1)[0]
    end_arrival_time = arrival_time[indices]
    max_time = th.max(end_arrival_time)
    min_time = th.min(end_arrival_time)
    mean_time = th.mean(end_arrival_time)
    var_time = th.var(end_arrival_time)
    return max_time, min_time, mean_time, var_time


def vis_design_output(args):
    source_designs = args.source_designs
    target_designs = args.target_designs
    node = args.node
    data_path = DATA_PATH[args.node]
    source_designs = source_designs.split(',')
    target_designs = target_designs.split(',')
    for sd in source_designs:
        max_time, min_time, mean_time, var_time = extract_design_output(sd, data_path)
        print(f'For source design {sd}, the time info: ')
        print(f'max min mean var time: {max_time} {min_time} {mean_time} {var_time}')

def main(args):
    id2ctype = load_ctype(args.node)
    data_path = DATA_PATH[args.node]
    source_designs = args.source_designs.split(',')
    target_designs = args.target_designs.split(',')
    print(f'source designs: {source_designs} target design: {target_designs}')

    for sd in source_designs:
        res = {}
        sd_dist = extract_design_info(sd, data_path)
        sd_dist = sd_dist.unsqueeze(0)
        for i, td in enumerate(target_designs):
            td_dist = extract_design_info(td, data_path)
            td_dist = td_dist.unsqueeze(0)
            # Assuming a and b are your tensors
            out = cal_dist(sd_dist, td_dist, metric='cos')
            res[target_designs[i]] = out
            vis_dist_comp(sd_dist, td_dist, sd, td, node=args.node)
        print(f'For source design: {sd} the distances are: {res}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=str, default='130')
    parser.add_argument('--source_designs', type=str, default=None)
    parser.add_argument('--target_designs', type=str, default=None)
    args = parser.parse_args()
    # main(args)
    vis_design_output(args)