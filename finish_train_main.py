from finish_train import *
from load_data import *
import torch

'''
Config
- if running pre-trained models, batch_size, dropout and hidden_channels will be rewritten with best hyperparams
'''

args = {}
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args['directory'] = 'data/'

# data parameters
args['num_data_points'] = 324191  # all=324191 , number of data points to use

# training parameters 
args['num_layers'] = 3  # number of layers in MLP
args['hidden_channels'] = 64  # channels in MLP
args['hidden_channels_conv'] = 64 # channels in convolutional block in GCN_base
args['dropout'] = 0.2
args['batch_size'] = 256
args['lr'] = 0.01
args['lr_decay_factor'] = 0.5

args['num_epochs'] = 10 # how many more epochs to train for
args['pre_trained_epochs'] = 10 # if using a pre-trained model, set this to the number of epochs it was trained for, if set to zero it will train model from scratch
args['use_best_hyperparams'] = True # training will use best hyperparams from best run. Can be used with pre-trained model or without but hyperparameter optimization must have been run before

'''
Provide a list of assays to run training for.

Example: [['2797'], ['2796','1979']] will train a uni-assay model for assay 2797 and a multi-assay model for assays 2796 and 1979

The uni-assay models I ran were: [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]
The multi-assay models I ran were: [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]
'''

assay_groups = load_assay_groups(args['directory'])


args['assay_list'] = [['2797'], ['2796'], ['1979'], ['602248'], ['1910'], ['602274'], ['720582'], ['1259313'], ['624204'], ['652039']]

'''
Provide a list of models to run training for (for the assays chosen above).
'''

args['models_list'] = ['FP', 'GROVER_FP', 'GCN_MLP_FP', 'GCN_MLP_FP_GROVER']



args['assay_list'] = [['2797'], ['2796', '1259313']]

args['models_list'] = ['FP', 'GROVER_FP']


# run function
finish_train(args)