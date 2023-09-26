# %%
from sweep import *
from load_data import *
import torch


'''
Config
'''

args = {}
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args['directory'] = 'data/'

# data parameters
args['num_data_points'] = 324191  # all=324191 , number of data points to use


# training parameters
args['num_epochs'] = 10 # how many epochs to train for in sweep
args['num_layers'] = 3  # number of layers in MLP
args['hidden_channels_conv'] = 64 # channels in convolutional block in GCN_base
args['lr'] = 0.01
args['lr_decay_factor'] = 0.5

# hyperparameter search parameters
args['num_folds'] = 2 # number of folds for cross-validation
args['samples'] = 2 # number of hyperparameter combinations to try
# which hyperparameters to search over by random sampling
hyperparams_dict = {
            'batch_size': {
                'values': [128, 256]
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            },
            'hidden_channels': {
                'values': [64, 128, 256]
            },
        }

'''
Provide a list of assays to run hyperparameter sweep for.

Example: [['2797'], ['2796','1979']] will train a uni-assay model for assay 2797 and a multi-assay model for assays 2796 and 1979

The uni-assay models I ran were: [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]
The multi-assay models I ran were: [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]
'''

assay_groups = load_assay_groups(args['directory'])


args['assay_list'] = [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]

'''
Provide a list of models to run training for (for the assays chosen above).
'''

args['models_list'] = ['GCN_base', 'FP', 'GROVER_FP', 'GCN', 'GCN_FP', 'GCN_FP_GROVER']

args['assays_list'] = [['2797'], ['2796', '1259313']]



# run sweep
sweep(hyperparams_dict, args)