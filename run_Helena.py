# %%
from sweep_run import *
from support_funcs import *
from load_data import *
import torch
import wandb
import pickle

# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
'''
Load assay info
'''
directory = 'data/'

# load the assay groups
assay_groups = {}
with open(directory + 'info/cell_based_high_hr.txt', 'r') as file:
    lines = file.read().splitlines()
assay_groups['cell_based_high_hr'] = list(map(str, lines))
with open(directory + 'info/cell_based_med_hr.txt', 'r') as file:
    lines = file.read().splitlines()
assay_groups['cell_based_med_hr'] = list(map(str, lines))
with open(directory + 'info/cell_based_low_hr.txt', 'r') as file:
    lines = file.read().splitlines()
assay_groups['cell_based_low_hr'] = list(map(str, lines))
with open(directory + 'info/non_cell_based_high_hr.txt', 'r') as file:
    lines = file.read().splitlines()
assay_groups['non_cell_based_high_hr'] = list(map(str, lines))
with open(directory + 'info/non_cell_based_med_hr.txt', 'r') as file:
    lines = file.read().splitlines()
assay_groups['non_cell_based_med_hr'] = list(map(str, lines))
with open(directory + 'info/non_cell_based_low_hr.txt', 'r') as file:
    lines = file.read().splitlines()
assay_groups['non_cell_based_low_hr'] = list(map(str, lines))
# load assay order
with open(directory + 'info/assay_order.txt', 'r') as f:
    assay_order = [line.strip() for line in f.readlines()]

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
args['grover_fp_dim'] = 5000
args['fp_dim'] = 2215  # dim of fingerprints


# training parameters
# 'LR, 'GCN', 'GCN_FP', 'GCN_MLP', 'GCN_MLP_FP', 'FP', 'GROVER', 'GROVER_FP'
args['model'] = 'GCN_MLP'
args['num_layers'] = 3  # number of layers in MLP
args['hidden_channels'] = 64  # 64
args['hidden_channels_conv'] = 64
args['dropout'] = 0.2
args['batch_size'] = 256
args['num_epochs'] = 70
args['lr'] = 0.01

args['best_auc'] = 0

# %%
'''
Sweeps
'''
wandb.login(key='69f641df6e6f0934ab302070cf0b3bcd5399ddd3')

for assay in ['2797', '2796', '1979', '602248', '1910',  '602274', '720582', '1259313', '624204', '652039']:
    for model in ['GCN_MLP', 'GCN_MLP_FP']:
        args['assay_list'] = [assay]
        args['num_assays'] = len(args['assay_list'])
        args['assays_idx'] = find_assay_indeces(
            args['assay_list'], assay_order)
        args['model'] = model
        args['best_auc'] = 0

        print('\n\n====================================================')
        print('Assays:', args['assay_list'], '| Model:', args['model'])
        print('====================================================\n')

        name = 'ass' + args['assay_list'][0] + '_' + args['model']
        sweep_config = {
            'name': name,
            'program': 'sweep_run.py',
            'method': 'bayes',
            'metric': {'goal': 'maximize', 'name': 'AUC val'},
        }
        parameters_dict = {
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

        parameters_dict.update({
            'assays': {
                'value': args['assay_list']},
            'num_data_points': {
                'value': args['num_data_points']},
            'num_layers': {
                'value': args['num_layers']},
            'hidden_channels_conv': {
                'value': args['hidden_channels_conv']
            },
            'lr': {
                'value': args['lr']}
        })

        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(
            sweep_config, project="GDL_molecular_activity_prediction_SWEEPS")

        # %%

        # save args dictionary with pickle
        with open('wandb/args.pkl', 'wb') as f:
            pickle.dump(args, f)
        # run the sweep
        wandb.agent(sweep_id, count=6)
