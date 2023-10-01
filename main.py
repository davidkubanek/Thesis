
from sweep import *
from finish_train import *
from load_data import *
import torch


args = {}
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args['directory'] = 'data/'

'''
Run hyperparameter sweep for a given model and list of assays

Provide a list of assays to run training for in args['assays_list'].

Example: [['2797'], ['2796','1979']] will train a uni-assay model for assay 2797 and a multi-assay model for assays 2796 and 1979

The uni-assay models I ran were: [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]
The multi-assay models I ran were: [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]

Can also use 'assay_groups' dictionary to access lists of assays in different hit rates categories: 'cell_based_high_hr', 'cell_based_med_hr', 'cell_based_low_hr', 'cell_based', 'biochemical_high_hr', 'biochemical_med_hr', 'biochemical_low_hr', 'biochemical'

Provide a list of models to run training for (for the assays chosen above) in args['models_list'].

Hyperparameter sweeps are only possible for models: GCN_base, FP, GROVER_FP, GCN, GCN_FP, GCN_FP_GROVER
'''

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

# dictionary of assay groups for different hit rates and assay types
assay_groups = load_assay_groups(args['directory'])
print(assay_groups['cell_based_high_hr'])


args['assays_list'] = [['1910'], ['2796', '1979']]

args['models_list'] = ['GCN_base', 'FP', 'GROVER_FP', 'GCN', 'GCN_FP', 'GCN_FP_GROVER']


# run sweeps
print('\n\n==================================================================================')
print('Running hyperparameter sweeps for selected assays and models...')
print('Samples: ', args['samples'], ' | CV folds: ', args['num_folds'])
print('==================================================================================\n')
sweep(hyperparams_dict, args)


'''
Finish training for best hyperparameters

- if running pre-trained models, batch_size, dropout and hidden_channels will be rewritten with best hyperparams
- if using a pre-trained model, set args['pre_trained_epochs'] to the number of epochs it was trained for, if set to zero it will train model from scratch
- if training from scratch, set args['use_best_hyperparams'] to False, but it will also reset automatically
- if want to use best number of epochs found for a given model, set args['use_best_no_epochs'] to True, and args['num_epochs'] will be overwritten with best number of epochs
'''

# training parameters 
args['hidden_channels'] = 64  # channels in MLP
args['hidden_channels_conv'] = 64 # channels in convolutional block in GCN_base
args['dropout'] = 0.2
args['batch_size'] = 256


args['num_epochs'] = 15 # how many more epochs to train for

args['pre_trained_epochs'] = 10 # if using a pre-trained model, set this to the number of epochs it was trained for, if set to zero it will train model from scratch
args['use_best_hyperparams'] = True # training will use best hyperparams from best run. Can be used with pre-trained model or without but hyperparameter optimization must have been run before
args['use_best_no_epochs'] = False # if want to use best number of epochs found for a given model, set this to True

'''
Provide a list of models to run training for (for the same assays as in the sweeps above).

Models: 'LogReg', 'RF', 'GCN_base', 'FP', 'GROVER_FP', 'GCN', 'GCN_FP', 'GCN_FP_GROVER'

Note that each model achieved the best performance after a different number
of total training epochs:
    - LogReg: 100 epochs
    - GCN_base: 120 epochs
    - FP: 120 epochs
    - GROVER: 190 epochs
    - GROVER_FP: 120 epochs
    - GCN: 190 epochs
    - GCN_FP: 100 epochs
    - GCN_FP_GROVER: 190 epochs
'''

args['models_list'] = ['LogReg', 'RF', 'GCN_base', 'FP', 'GROVER_FP', 'GCN', 'GCN_FP', 'GCN_FP_GROVER']

# run function
print('\n\n==================================================================================')
print('Finishing training for selected assays and models...')
print('Pre-trained epochs:', args['pre_trained_epochs'], '| Use best hyperparams:', args['use_best_hyperparams'], '| Use best number of epochs:', args['use_best_no_epochs'])
print('==================================================================================\n')
finish_train(args)

print('\n\nAll training finished.')