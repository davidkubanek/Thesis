

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

'''
GCN and GCN_FP
- GCN: graph embedding followed by a final classification layer
- GCN_FP: graph + fingerprints embedding followed by a final classification layer
'''


class GCN(nn.Module):
    '''
    Define a Graph Convolutional Network (GCN) model architecture.
    Can include 'graph' only or 'graph + fingerprints' embedding before final classification layer.
    '''

    def __init__(self, args):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        num_node_features = args['num_node_features']
        hidden_channels = args['hidden_channels']
        num_classes = args['num_assays']
        self.dropout = args['dropout']

        if args['model'] == 'GCN_FP':
            fp_dim = args['fp_dim']
        else:
            fp_dim = 0

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels + fp_dim, num_classes)

    def forward(self, x, edge_index, batch, fp=None):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # if also using fingerprints
        if fp is not None:
            # reshape fp to batch_size x fp_dim
            fp = fp.reshape(x.shape[0], -1)
            # concatenate graph node embeddings with fingerprint
            # print('BEFORE CONCAT x:',x.shape, 'fp:', fp.shape)
            x = torch.cat([x, fp], dim=1)
            # print('AFTER CONCAT x:',x.shape)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x


'''
FP, GROVER and GROVER_FP
- FP: fingerprints embedding followed by a final classification layer
- GROVER: graph transformer embedding followed by a final classification layer
- GROVER_FP: graph transformer + fingerprints embedding followed by a final classification layer
'''


class LinearBlock(nn.Module):
    """ basic block in an MLP, with dropout and batch norm """

    def __init__(self, in_feats, out_feats, dropout=0.1):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ReLU activation, batch norm, dropout on layer
        return self.bn(self.dropout(F.relu(self.linear(x))))


def construct_mlp(in_dim, out_dim, hidden_dim, hidden_layers, dropout=0.1):
    """
    Constructs an MLP with specified dimensions.
            - total number of layers = hidden_layers + 1 (the + 1 is for the output linear)
            - no activation/batch norm/dropout on output layer
    """

    assert hidden_layers >= 1, hidden_layers
    mlp_list = []
    mlp_list.append(LinearBlock(in_dim, hidden_dim, dropout=dropout))
    for i in range(hidden_layers-1):
        mlp_list.append(LinearBlock(hidden_dim, hidden_dim, dropout=dropout))

    # no activation/batch norm/dropout on output layer
    mlp_list.append(nn.Linear(hidden_dim, out_dim))
    mlp = nn.Sequential(*mlp_list)
    return mlp


class MLP(nn.Module):
    '''
    MLP with optional Grover fingerprints.
    Customizable number of layers, hidden dimensions, and dropout.
    '''

    def __init__(self, args):

        super(MLP, self).__init__()

        self.model_type = args['model']
        self.fp_dim = args['fp_dim']  # can be 0
        self.grover_fp_dim = args['grover_fp_dim']  # can be 0
        self.hidden_dim = args['hidden_channels']
        self.output_dim = args['num_assays']
        self.num_layers = args['num_layers']
        self.dropout = args['dropout']

        assert self.model_type in [
            'FP', 'GROVER', 'GROVER_FP'], f'model type not supported: {self.model_type}'

        if self.model_type == 'FP':
            self.grover_fp_dim = 0
        elif self.model_type == 'GROVER':
            self.fp_dim = 0

        self.ff_layers = construct_mlp(
            self.fp_dim + self.grover_fp_dim,
            self.output_dim,
            self.hidden_dim,
            self.num_layers,
            self.dropout
        )

    def forward(self, data):

        if self.model_type == 'FP':  # only fp is used
            fingerprints = data.fp
            # reshape fp to batch_size x fp_dim
            fingerprints = fingerprints.reshape(
                int(fingerprints.shape[0]/self.fp_dim), -1)

            output = self.ff_layers(fingerprints)

        elif self.model_type == 'GROVER':  # only grover is used
            # reshape grover_fp to batch_size x grover_fp_dim
            grover_fp = data.grover_fp
            grover_fp = grover_fp.reshape(
                int(grover_fp.shape[0]/self.grover_fp_dim), -1)

            output = self.ff_layers(grover_fp)

        elif self.model_type == 'GROVER_FP':  # grover and fp are concatenated
            fingerprints = data.fp
            # reshape fp to batch_size x fp_dim
            fingerprints = fingerprints.reshape(
                int(fingerprints.shape[0]/self.fp_dim), -1)
            # reshape grover_fp to batch_size x grover_fp_dim
            grover_fp = data.grover_fp
            grover_fp = grover_fp.reshape(
                int(grover_fp.shape[0]/self.grover_fp_dim), -1)

            output = self.ff_layers(
                torch.cat([fingerprints, grover_fp], dim=1))

        return output

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # reshape fp to batch_size x fp_dim
        x = x.reshape(
            int(x.shape[0]/self.input_dim), -1)
            
        outputs = torch.sigmoid(self.linear(x))
        return outputs