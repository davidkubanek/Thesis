# %%
from load_data import *
from support_funcs import *
from sweep_run import *
from train import *
from models import *

import torch
import wandb
# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
'''
Load data
'''
# directory = 'data/'
# directory = '/content/drive/MyDrive/Thesis/Data/'
directory = '/Volumes/Kubánek UCL/Data/Thesis MSc/PubChem Data/'
# directory = 'Data/PubChem Data/'

# Specify the path where you saved the dictionary
load_path = directory + 'final/datalist_small.pkl'  # no_out.pkl'

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
args['model'] = 'GCN_MLP'  # 'GCN', 'GCN_FP', 'FP', 'GROVER', 'GROVER_FP'
args['num_layers'] = 3  # number of layers in MLP
args['hidden_channels'] = 64  # 64
args['hidden_channels_conv'] = 64
args['dropout'] = 0.2
args['batch_size'] = 256
args['num_epochs'] = 5
args['lr'] = 0.01

# create dataset splits (train, val, test) on device given args
data_splits = prepare_splits(data_list, args)

args['best_auc'] = 0


# %%

# create dataset from data_list
dataloader = prepare_dataloader(data_splits, args)

args['assay_list'] = ['2797']
args['num_assays'] = 1
args['assays_idx'] = find_assay_indeces(args['assay_list'], assay_order)

print('TESTING')
for model_type in ['GCN_MLP','GCN_MLP_FP']:
  args['model'] = model_type
  model = GCN_MLP(args)
  for data in dataloader['train']:  # Iterate in batches over the training dataset
          print(f'------{model_type}------')
          print('inputs:')
          print(' x:', data.x.shape, '| y:',data.y.shape, '| fp:',data.fp.shape, '| grover:', data.grover_fp.shape)
          # print num of params in model
          print('num of params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
          if args['model'] == 'GCN_MLP': # Perform a single forward pass
              out = model(data.x, data.edge_index, data.batch)
          elif args['model'] == 'GCN_MLP_FP':
              out = model(data.x, data.edge_index, data.batch, fp=data.fp)
          print('out:',out.shape)
          print('gt:', data.y[:,args['assays_idx']].shape)
          break
  

# %%
'''
Run
'''
# %%
# wandb.login(key='69f641df6e6f0934ab302070cf0b3bcd5399ddd3')
# # API KEY: 69f641df6e6f0934ab302070cf0b3bcd5399ddd3

# for assay in ['2797']: #, '2796', '1979', '602248', '1910', '602274', '720582', '1259313', '624204', '652039']:
#     # assay parameters
#     args['assay_list'] = [assay]
#     args['num_assays'] = 1
#     args['assays_idx'] = find_assay_indeces(args['assay_list'], assay_order)

#     args['model'] = 'GCN_MLP'
#     args['dropout'] = 0.1
#     args['batch_size'] = 256
#     args['hidden_channels'] = 256
#     args['hidden_channels_conv'] = 64
#     args['num_epochs'] = 100
#     args['num_layers'] = 3
#     args['lr'] = 0.01

#     # Create a custom run name dynamically
#     run_name = f"ass{args['assay_list'][0]}_{args['model']}"
#     run = wandb.init(
#         name=run_name,
#         # Set the project where this run will be logged
#         project="GDL_molecular_activity_prediction_BASE",
#         # Track hyperparameters and run metadata
#         config={
#             'num_data_points': args['num_data_points'],
#             'assays': args['assay_list'],
#             'num_assays': args['num_assays'],

#             'model': args['model'],
#             'dropout': args['dropout'],
#             'batch_size': args['batch_size'],
#             'num_epochs': args['num_epochs'],
#             'lr': args['lr'],
#         })

#     # create dataset from data_list
#     dataloader = prepare_dataloader(data_splits, args)

#     # train model
#     exp = TrainManager(dataloader, args)
#     exp.train(epochs=100, log=True, wb_log=True, early_stop=True)