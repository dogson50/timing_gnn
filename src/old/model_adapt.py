import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from time import time


class MLP(nn.Module):
    r"""
                Description
                -----------
                a simple multilayer perceptron
    """

    def __init__(self, in_dim, out_dim, nlayers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.Sequential()
        dim1 = in_dim
        for i in range(nlayers - 1):
            self.layers.add_module('dropout_{}'.format(i + 1), self.dropout)
            self.layers.add_module('activation_{}'.format(i + 1), self.activation)
            self.layers.add_module('linear_{}'.format(i + 1), nn.Linear(dim1, int(dim1 / 2)))
            dim1 = int(dim1 / 2)
        self.layers.add_module('linear_{}'.format(nlayers), nn.Linear(dim1, out_dim))

    def forward(self, embedding):
        return self.layers(embedding).squeeze(-1)


def cal_KL_gaussians(mean1, var1, mean2, var2):
    kl_div = 0.5 * torch.mean(torch.log(var2) - torch.log(var1) - 1 + (var1 + (mean2 - mean1).pow(2)) / var2, 1)
    return kl_div


class MlpBayesRes(nn.Module):

    def __init__(self, in_dim, out_dim, nlayers, activation=nn.ReLU(), dropout=0.5,
                 momentum=0.01, K=10, fix_alpha=False, alpha=0.):
        super(MlpBayesRes, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation1 = activation
        self.sampling_num = K
        self.dropout = nn.Dropout(p=dropout)
        self.w0 = nn.Linear(2 * in_dim, 1)
        self.linear1 = nn.Linear(in_dim, 2 * in_dim)
        self.shared = nn.Linear(2 * in_dim, 2 * in_dim)
        self.activation2 = nn.ReLU()
        self.mean = nn.Linear(2 * in_dim, 2 * in_dim)
        self.sigma = nn.Linear(2 * in_dim, 2 * in_dim)
        if fix_alpha:
            self.register_buffer('alpha', torch.Tensor([alpha]))
        else:
            self.register_parameter('alpha', nn.Parameter(torch.Tensor([0.])))
        self.linear2 = nn.Linear(2 * in_dim, out_dim)
        self.register_buffer('global_feat', torch.zeros((2 * in_dim)))
        self.momentum = momentum

    def global_ema(self, embedding):
        mean = th.mean(embedding, dim=0)
        self.global_feat = self.momentum * mean.clone().detach() + (1 - self.momentum) * self.global_feat

    def get_prior(self, gaussian_prior=False):
        if gaussian_prior:
            mean_global = th.zeros(2 * self.in_dim).to(self.global_feat.device)
            sigma_global = th.ones(2 * self.in_dim).to(self.global_feat.device)
        else:
            global_hidden_bayes = self.activation2(self.shared(self.global_feat))
            mean_global = self.mean(global_hidden_bayes)  # (BS, 256)
            sigma_global = F.softplus(self.sigma(global_hidden_bayes))  # (BS, 256)
        return mean_global, sigma_global

    def forward(self, embedding, training=False, kl=False, sampling=False, gaussian_prior=False):
        hidden_embed = self.activation1(self.linear1(embedding))
        # print(f'hidden_embed shape: {hidden_embed.shape}')
        # update the global feature
        self.global_ema(hidden_embed)
        hidden_bayes = self.activation2(self.shared(hidden_embed))
        # print(f'hidden bayes shape: {hidden_bayes.shape}')
        mean_ = self.mean(hidden_bayes) # (BS, 256)
        sigma_ = F.softplus(self.sigma(hidden_bayes)) # (BS, 256)
        kl_div = None
        if training and kl:
            mean_global_, sigma_global_ = self.get_prior(gaussian_prior=gaussian_prior)
            # calculate the KL divergence
            mean_global = mean_global_.unsqueeze(0).repeat(mean_.shape[0], 1) # (BS, 256)
            sigma_global = sigma_global_.unsqueeze(0).repeat(mean_.shape[0], 1) # (BS, 256)
            # print(f'mean global shape: {mean_global.shape}')
            kl_div = cal_KL_gaussians(mean_, sigma_**2+1e-6, mean_global, sigma_global**2+1e-6)
            # print(f'kl div: {kl_div}')
        # sampling
        if sampling:
            mean = mean_.unsqueeze(1).repeat(1, self.sampling_num, 1)
            sigma = sigma_.unsqueeze(1).repeat(1, self.sampling_num, 1)
            sampling_gaussian = mean.new(mean.size()).normal_()
            w_p = mean + sigma * sampling_gaussian # (BS, n_mc, 256)
        else:
            w_p = mean_.unsqueeze(1) # (BS, 1, 256)
        # print(f'wp shape: {w_p.shape}')
        pred_0 = self.w0(hidden_embed).squeeze(1)
        # print(f'pred 0 shape: {pred_0.shape}')
        # residual prediction
        hidden_embed_ = hidden_embed.unsqueeze(1) # (BS, 1, 256)
        # print(f'hidden embed shape: {hidden_embed.shape}')
        pred_mc = th.matmul(w_p, hidden_embed_.transpose(-2, -1)) # (BS, n_mc, 1)
        # print(f'pred mc shape: {pred_mc.shape}')
        pred_res = torch.mean(pred_mc, dim=1)
        # print(f'pred shape: {pred.shape}')
        pred_res = pred_res.squeeze(1)
        # print(f'pred res shape: {pred_res.shape}')
        pred = pred_0 * (1 - self.alpha) + self.alpha * pred_res
        # exit()
        return pred, kl_div


class MLP2BAYES(nn.Module):

    def __init__(self, in_dim, out_dim, nlayers, activation=nn.ReLU(), dropout=0.5, momentum=0.01, K=10):
        super(MLP2BAYES, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation1 = activation
        self.sampling_num = K
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_dim, 2 * in_dim)
        self.shared = nn.Linear(2 * in_dim, 2 * in_dim)
        self.activation2 = nn.ReLU()
        self.mean = nn.Linear(2 * in_dim, 2 * in_dim)
        self.sigma = nn.Linear(2 * in_dim, 2 * in_dim)
        self.register_parameter('alpha', nn.Parameter(torch.Tensor([0.])))
        self.linear2 = nn.Linear(2 * in_dim, out_dim)
        self.register_buffer('global_feat', torch.zeros((2 * in_dim)))
        self.momentum = momentum

    def global_ema(self, embedding):
        mean = th.mean(embedding, dim=0)
        self.global_feat = self.momentum * mean.clone().detach() + (1 - self.momentum) * self.global_feat

    def get_prior(self, gaussian_prior=False):
        if gaussian_prior:
            mean_global = th.zeros(2 * self.in_dim).to(self.global_feat.device)
            sigma_global = th.ones(2 * self.in_dim).to(self.global_feat.device)
        else:
            global_hidden_bayes = self.activation2(self.shared(self.global_feat))
            mean_global = self.mean(global_hidden_bayes)  # (BS, 256)
            sigma_global = F.softplus(self.sigma(global_hidden_bayes))  # (BS, 256)
        return mean_global, sigma_global

    def forward(self, embedding, training=False, kl=False, sampling=False, gaussian_prior=False):
        hidden_embed = self.activation1(self.linear1(embedding))
        # print(f'hidden_embed shape: {hidden_embed.shape}')
        # update the global feature
        self.global_ema(hidden_embed)
        hidden_bayes = self.activation2(self.shared(hidden_embed))
        # print(f'hidden bayes shape: {hidden_bayes.shape}')
        mean_ = self.mean(hidden_bayes) # (BS, 256)
        sigma_ = F.softplus(self.sigma(hidden_bayes)) # (BS, 256)
        kl_div = None
        if training and kl:
            mean_global_, sigma_global_ = self.get_prior(gaussian_prior=gaussian_prior)
            # calculate the KL divergence
            mean_global = mean_global_.unsqueeze(0).repeat(mean_.shape[0], 1) # (BS, 256)
            sigma_global = sigma_global_.unsqueeze(0).repeat(mean_.shape[0], 1) # (BS, 256)
            # print(f'mean global shape: {mean_global.shape}')
            kl_div = cal_KL_gaussians(mean_, sigma_**2+1e-6, mean_global, sigma_global**2+1e-6)
            # print(f'kl div: {kl_div}')
        # sampling
        if sampling:
            mean = mean_.unsqueeze(1).repeat(1, self.sampling_num, 1)
            sigma = sigma_.unsqueeze(1).repeat(1, self.sampling_num, 1)
            sampling_gaussian = mean.new(mean.size()).normal_()
            w_p = mean + sigma * sampling_gaussian # (BS, n_mc, 256)
        else:
            w_p = mean_.unsqueeze(1) # (BS, 1, 256)
        # print(f'wp shape: {w_p.shape}')
        hidden_embed_ = hidden_embed.unsqueeze(1) # (BS, 1, 256)
        # print(f'hidden embed shape: {hidden_embed.shape}')
        pred_mc = th.matmul(w_p, hidden_embed_.transpose(-2, -1)) # (BS, n_mc, 1)
        # print(f'pred mc shape: {pred_mc.shape}')
        pred = torch.mean(pred_mc, dim=1)
        # print(f'pred shape: {pred.shape}')
        pred = pred.squeeze(1)
        return pred, kl_div


class MLP2(nn.Module):
    r"""
                Description
                -----------
                a simple multilayer perceptron
    """

    def __init__(self, in_dim, out_dim, nlayers, activation=nn.ReLU(), dropout=0.5):
        super(MLP2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 2 * in_dim),
            self.activation,
            nn.Linear(2 * in_dim, out_dim),
        )
        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.layers[0].weight, gain=gain)
        # nn.init.xavier_uniform_(self.layers[2].weight, gain=gain)

    def forward(self, embedding):
        return self.layers(embedding).squeeze(-1)


class MLP3(nn.Module):
    r"""
                Description
                -----------
                MLP with tanh activation at the output
    """

    def __init__(self, in_dim, out_dim, nlayers, activation1=nn.ReLU(), activation2=nn.Tanh(), dropout=0.5):
        super(MLP3, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation1 = activation1
        self.activation2 = activation2
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 2 * in_dim),
            self.activation1,
            nn.Linear(2 * in_dim, out_dim),
            self.activation2
        )
        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.layers[0].weight, gain=gain)
        # nn.init.xavier_uniform_(self.layers[2].weight, gain=gain)

    def forward(self, embedding):
        return self.layers(embedding).squeeze(-1)


class LayoutNet3(nn.Module):
    def __init__(self, pooling):
        super(LayoutNet3, self).__init__()
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2, 2, 0, 1)
            pooling_layer2 = nn.MaxPool2d(2, 2, 0, 1)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2, 2, 0)
            pooling_layer2 = nn.AvgPool2d(2, 2, 0)
        else:
            assert False, 'wrong pooling type for layoutnet!'

        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(64, 128, 9, 1, 4),
            nn.ReLU(),
            # nn.Conv2d(32, 1, 7, 1, 3),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(128, 256, 7, 1, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 9, 2, 4, 1),
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, 2, 1),
            nn.Conv2d(32, 1, 3, 1, 1),
            pooling_layer2,
            nn.ReLU()
        )

    def forward(self, x):
        encode_out = self.encode(x)
        out = self.decode(encode_out)
        return out


class LayoutNet2(nn.Module):
    def __init__(self, pooling):
        super(LayoutNet2, self).__init__()
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2, 2, 0, 1)
            pooling_layer2 = nn.MaxPool2d(4, 4, 0, 1)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2, 2, 0)
            pooling_layer2 = nn.AvgPool2d(4, 4, 0)
        else:
            assert False, 'wrong pooling type for layoutnet!'

        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(),
            pooling_layer,
            nn.Conv2d(64, 32, 9, 1, 4),
            nn.ReLU(),
            # nn.Conv2d(32, 1, 7, 1, 3),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(32, 32, 7, 1, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 9, 2, 4, 1),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, 5, 2, 2, 1),
            nn.Conv2d(4, 1, 3, 1, 1),
            pooling_layer2,
            nn.ReLU()
        )

    def forward(self, x):
        encode_out = self.encode(x)
        out = self.decode(encode_out)
        return out


class LayoutNet(nn.Module):

    def __init__(self, pooling):
        super(LayoutNet, self).__init__()
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2, 2, 0, 1)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2, 2, 0)
        else:
            assert False, 'wrong pooling type for layoutnet!'
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            # nn.ReLU(),
            pooling_layer,
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, 1, 3),
            # nn.ReLU(),
            pooling_layer,
            nn.ReLU(),
            nn.Conv2d(64, 32, 9, 1, 4),
            nn.ReLU(),
            nn.Conv2d(32, 1, 7, 1, 3),
            nn.ReLU())

        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_uniform_(self.encode[0].weight, gain=gain)
        # nn.init.xavier_uniform_(self.encode[3].weight, gain=gain)
        # nn.init.xavier_uniform_(self.encode[6].weight, gain=gain)
        # nn.init.xavier_uniform_(self.encode[8].weight, gain=gain)

    def forward(self, x):
        out = self.encode(x)
        return out


class SepBayesDisentangleModel(nn.Module):
    r"""
                    Description
                    -----------
                    The model used to predict the arrival time
                    consisting of one GNN model and one MLP model.

        """

    def __init__(
            self, gnn, fcn, mlp_node, mlp_others, mlp_7, mlp_130
    ):
        super(SepBayesDisentangleModel, self).__init__()

        self.gnn = nn.Sequential(gnn)
        self.fcn = nn.Sequential(fcn)
        self.mlp_7 = nn.Sequential(mlp_7)
        self.mlp_130 = nn.Sequential(mlp_130)
        self.mlp_node = nn.Sequential(mlp_node)
        self.mlp_others = nn.Sequential(mlp_others)

    def forward(self, graph, nodes, eids, target_list, level_id, path_map, node='7',
                training=False, kl=False, sampling=False, gaussian_prior=False):

        h_cnn = self.fcn[0](path_map) \
            if self.fcn[0] is not None and len(target_list) != 0 \
            else None
        h_gnn = self.gnn[0](graph, nodes, eids, target_list, level_id) \
            if self.gnn[0] is not None \
            else None

        if len(target_list) == 0:
            return None

        if h_cnn is None:
            h = h_gnn
        elif h_gnn is None:
            h = h_cnn
        else:
            h = th.cat((h_gnn, h_cnn), 1)
        h_node = self.mlp_node[0](h)
        h_others = self.mlp_others[0](h)
        h = th.cat((h_node, h_others), 1)
        if node == '7':
            # print(f'forward with 7nm mlp')
            return self.mlp_7[0](h, training=training, kl=kl, sampling=sampling,
                                 gaussian_prior=gaussian_prior), h_node, h_others
        elif node == '130':
            # print(f'forward with 130nm mlp')
            return self.mlp_130[0](h, training=training, kl=kl, sampling=sampling,
                                   gaussian_prior=gaussian_prior), h_node, h_others
        else:
            raise ValueError


class DisSepPathModel(nn.Module):
    r"""
                    Description
                    -----------
                    The model used to predict the arrival time
                    consisting of one GNN model and one MLP model.

        """

    def __init__(
            self, gnn, fcn, mlp_7, mlp_130, mlp_node, mlp_others
    ):
        super(DisSepPathModel, self).__init__()

        self.gnn = nn.Sequential(gnn)
        self.fcn = nn.Sequential(fcn)
        self.mlp_7 = nn.Sequential(mlp_7)
        self.mlp_130 = nn.Sequential(mlp_130)
        self.mlp_node = nn.Sequential(mlp_node)
        self.mlp_others = nn.Sequential(mlp_others)

    def forward(self, graph, nodes, eids, target_list, level_id, path_map, node='7'):

        h_cnn = self.fcn[0](path_map) \
            if self.fcn[0] is not None and len(target_list) != 0 \
            else None
        h_gnn = self.gnn[0](graph, nodes, eids, target_list, level_id) \
            if self.gnn[0] is not None \
            else None

        if len(target_list) == 0:
            return None

        if h_cnn is None:
            h = h_gnn
        elif h_gnn is None:
            h = h_cnn
        else:
            h = th.cat((h_gnn, h_cnn), 1)
        h_node = self.mlp_node[0](h)
        h_others = self.mlp_others[0](h)
        h = th.cat((h_node, h_others), 1)
        if node == '7':
            return self.mlp_7[0](h), h_node, h_others
        elif node == '130':
            return self.mlp_130[0](h), h_node, h_others
        else:
            raise ValueError


class SepBayesModel(nn.Module):
    r"""
                    Description
                    -----------
                    The model used to predict the arrival time
                    consisting of one GNN model and one MLP model.

        """

    def __init__(
            self, gnn, fcn, mlp_7, mlp_130
    ):
        super(SepBayesModel, self).__init__()

        self.gnn = nn.Sequential(gnn)
        self.fcn = nn.Sequential(fcn)
        self.mlp_7 = nn.Sequential(mlp_7)
        self.mlp_130 = nn.Sequential(mlp_130)

    def forward(self, graph, nodes, eids, target_list, level_id, path_map, node='7',
                training=False, kl=False, sampling=False, gaussian_prior=False):

        h_cnn = self.fcn[0](path_map) \
            if self.fcn[0] is not None and len(target_list) != 0 \
            else None
        h_gnn = self.gnn[0](graph, nodes, eids, target_list, level_id) \
            if self.gnn[0] is not None \
            else None

        if len(target_list) == 0:
            return None

        if h_cnn is None:
            h = h_gnn
        elif h_gnn is None:
            h = h_cnn
        else:
            h = th.cat((h_gnn, h_cnn), 1)
        if node == '7':
            # print(f'forward with 7nm mlp')
            return self.mlp_7[0](h, training=training, kl=kl, sampling=sampling, gaussian_prior=gaussian_prior)
        elif node == '130':
            # print(f'forward with 130nm mlp')
            return self.mlp_130[0](h, training=training, kl=kl, sampling=sampling, gaussian_prior=gaussian_prior)
        else:
            raise ValueError


class SepPathModel(nn.Module):
    r"""
                    Description
                    -----------
                    The model used to predict the arrival time
                    consisting of one GNN model and one MLP model.

        """

    def __init__(
            self, gnn, fcn, mlp_7, mlp_130
    ):
        super(SepPathModel, self).__init__()

        self.gnn = nn.Sequential(gnn)
        self.fcn = nn.Sequential(fcn)
        self.mlp_7 = nn.Sequential(mlp_7)
        self.mlp_130 = nn.Sequential(mlp_130)

    def forward(self, graph, nodes, eids, target_list, level_id, path_map, node='7', return_feat=False):

        h_cnn = self.fcn[0](path_map) \
            if self.fcn[0] is not None and len(target_list) != 0 \
            else None
        h_gnn = self.gnn[0](graph, nodes, eids, target_list, level_id) \
            if self.gnn[0] is not None \
            else None

        if len(target_list) == 0:
            return None

        if h_cnn is None:
            h = h_gnn
        elif h_gnn is None:
            h = h_cnn
        else:
            h = th.cat((h_gnn, h_cnn), 1)
        if node == '7':
            # print(f'forward with 7nm mlp')
            if return_feat:
                return self.mlp_7[0](h), h
            else:
                return self.mlp_7[0](h)
        elif node == '130':
            # print(f'forward with 130nm mlp')
            if return_feat:
                return self.mlp_7[0](h), h
            else:
                return self.mlp_130[0](h)
        else:
            raise ValueError


class PathModel(nn.Module):
    r"""
                    Description
                    -----------
                    the model used to classify
                    consits of two GNN models and one MLP model
        """

    def __init__(
            self, gnn, fcn, mlp
    ):
        super(PathModel, self).__init__()

        self.gnn = nn.Sequential(gnn)
        self.fcn = nn.Sequential(fcn)
        self.mlp = nn.Sequential(mlp)

    def forward(self, graph, nodes, eids, target_list, level_id, path_map):

        h_cnn = self.fcn[0](path_map) \
            if self.fcn[0] is not None and len(target_list) != 0 \
            else None
        h_gnn = self.gnn[0](graph, nodes, eids, target_list, level_id) \
            if self.gnn[0] is not None \
            else None

        if len(target_list) == 0:
            return None

        if h_cnn is None:
            h = h_gnn
        elif h_gnn is None:
            h = h_cnn
        else:
            h = th.cat((h_gnn, h_cnn), 1)

        return self.mlp[0](h)

        if self.gnn[0] is None:
            path_map = path_map.view(-1, 128 * 128)
            h = self.fcn[0](path_map)
        elif self.fcn[0] is None:
            h = self.gnn[0](graph, nodes, target_list, level_id)
        else:
            h_gnn = self.gnn[0](graph, nodes, target_list, level_id)
            h_cnn = self.fcn[0](path_map)

            # combine the information from both direction
            h = th.cat((h_gnn, h_cnn), 1)

        h = self.mlp[0](h)

        return h
