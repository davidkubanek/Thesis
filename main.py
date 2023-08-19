# %%
import torch
import wandb
import pickle

# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import train
import load_data
import support_funcs
import sweep_run


import importlib
# this method of import ensures that when support scripts are updated, the changes are imported in this script
importlib.reload(support_funcs)
importlib.reload(load_data)
# importlib.reload(train)
importlib.reload(sweep_run)

# from train import *
from load_data import *
from support_funcs import *
from sweep_run import *

# %%
'''
Load data
'''
directory = 'data/'
# directory = '/content/drive/MyDrive/Thesis/Data/'
# directory = '/Volumes/Kubánek UCL/Data/Thesis MSc/PubChem Data/'

# Specify the path where you saved the dictionary
load_path = directory + 'final/datalist_small.pkl' #no_out.pkl'

print('\nLoading data...')
data_list, assay_groups, assay_order = load_datalist(directory, load_path)
print('SUCCESS: Data loaded.')

# %%
'''
Config
'''

args = {}
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args['directory'] = directory

# data parameters
args['num_data_points'] = 324191  # all=324191 # number of data points to use
args['num_assays'] = 5  # number of assays to use (i.e., no. of output classes)
args['assay_start'] = 0  # which assay to start from
args['assay_order'] = assay_order
# number of node features in graph representation
args['num_node_features'] = 79
# grover_fp['fps'][0].shape[0] # None  # dim of grover fingerprints
args['grover_fp_dim'] = 5000
args['fp_dim'] = 2215  # dim of fingerprints


# training parameters
args['model'] = 'GROVER_FP'  # 'GCN', 'GCN_FP', 'FP', 'GROVER', 'GROVER_FP'
args['num_layers'] = 5  # number of layers in MLP
args['hidden_channels'] = 64  # 64
args['dropout'] = 0.2
args['batch_size'] = 256
args['num_epochs'] = 5
args['lr'] = 0.01
# args['gradient_clip_norm'] = 1.0
# args['network_weight_decay'] = 0.0001
args['lr_decay_factor'] = 0.5

# assay parameters
args['assay_list'] = [assay_groups['non_cell_based_high_hr'][0]] #['2797']
args['num_assays'] = 1
args['assays_idx'] = find_assay_indeces(args['assay_list'], assay_order)
print('Assays used:', args['assay_list'], 'Assay indeces:', args['assays_idx'])

# create dataset splits (train, val, test) on device given args
data_splits = prepare_splits_forCV(data_list, args)

args['best_auc'] = 0

#%%
'''
Sweeps
'''
wandb.login(key='69f641df6e6f0934ab302070cf0b3bcd5399ddd3')

sweep_config = {
    'program': 'sweep_run.py',
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'AUC test'},
    }
parameters_dict = {
    'batch_size': {
        'values': [128, 256, 512, 1014]
        },
    'dropout': {
        'values': [0.3, 0.5]
        },
    'hidden_channels': {
        'values': [64, 128, 256]
        },
    }

parameters_dict.update({
    'assays': {
        'value': args['assay_list']},
    'num_data_points': {
        'value': args['num_data_points']},
    'num_layers': {
        'value': args['num_layers']},
    'lr': {
        'value': args['lr']}
    })

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="GDL_molecular_activity_prediction_SWEEPS")

# %%

# save args dictionary with pickle
with open('wandb/args.pkl', 'wb') as f:
    pickle.dump(args, f)
# run the sweep
wandb.agent(sweep_id, run_sweep(data_splits, args), count=3)
