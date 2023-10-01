# %%
from sweep import *
from load_data import *
import torch


'''
Run hyperparameter sweep for a given model and list of assays

Provide a list of assays to run training for in args['assays_list'].

Example: [['2797'], ['2796','1979']] will train a uni-assay model for assay 2797 and a multi-assay model for assays 2796 and 1979

The uni-assay models I ran were: [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]
The multi-assay models I ran were: [['2796','2797'], ['2796','1979'], ['1910', '1979'], ['720582', '652039'], ['720582', '602274'], ['1259313', '602274'], ['2796', '2797', '602248'], ['2796', '2797', '602248', '1910'], ['2796', '2797', '602248', '1910', '1979'], ['720582', '624204', '652039'], ['720582', '624204', '652039', '1259313'], ['720582', '624204', '652039', '1259313', '602274']]


Can also use 'assay_groups' dictionary to access lists of assays in different hit rates categories: 'cell_based_high_hr', 'cell_based_med_hr', 'cell_based_low_hr', 'cell_based', 'biochemical_high_hr', 'biochemical_med_hr', 'biochemical_low_hr', 'biochemical'

Provide a list of models to run training for (for the assays chosen above) in args['models_list'].

Hyperparameter sweeps are only possible for models: GCN_base, FP, GROVER_FP, GCN, GCN_FP, GCN_FP_GROVER
'''

args = {}
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args['directory'] = 'data/'

# data parameters
args['num_data_points'] = 320016  # all=320016 , number of data points to use


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

# print out more information during training
args['verbose'] = False

assay_groups = load_assay_groups(args['directory'])


args['assay_list'] = [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]


args['models_list'] = ['GCN_base', 'FP', 'GROVER_FP', 'GCN', 'GCN_FP', 'GCN_FP_GROVER']

args['assays_list'] = [['2797'], ['2796', '1259313']]


# run sweep
print('\n\n==================================================================================')
print('Running hyperparameter sweeps for selected assays and models...')
print('Samples: ', args['samples'], ' | CV folds: ', args['num_folds'])
print('==================================================================================\n')
sweep(hyperparams_dict, args)