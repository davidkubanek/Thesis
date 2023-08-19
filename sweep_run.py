
import cross_val
import load_data
import wandb

import importlib
# this method of import ensures that when support scripts are updated, the changes are imported in this script
importlib.reload(cross_val)
importlib.reload(load_data)
from cross_val import *
from load_data import *

def run_sweep(data_splits, args):
   
    with wandb.init(config=args):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller

        # with wandb.init(config=wandb.config) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        # config = wandb.config

        args['batch_size'] = wandb.config.batch_size
        args['dropout'] = wandb.config.dropout
        args['hidden_channels'] = wandb.config.hidden_channels

        # train model with cross-validation
        CV_results = cross_val(data_splits, args)

        # log mean metrics of all folds to wandb
        wandb.log({'loss': np.mean(CV_results['loss']),
                'AUC train': np.mean(CV_results['auc_train']),
                'AUC val': np.mean(CV_results['auc_test']),
                'F1 train': np.mean(CV_results['f1_train']),
                'F1 val': np.mean(CV_results['f1_test']),
                'Precision train': np.mean(CV_results['precision_train']),
                'Precision val': np.mean(CV_results['precision_test']),
                'Recall train': np.mean(CV_results['recall_train']),
                'Recall val': np.mean(CV_results['recall_test'])})
    

# if name is main
if __name__ == '__main__':
    # load args from file using pickle
    with open('wandb/args.pkl', 'rb') as f:
        args = pickle.load(f)

    # load data
    load_path = args['directory'] + 'final/datalist_no_out.pkl' #no_out.pkl'

    print('\nLoading data...')
    data_list, assay_groups, assay_order = load_datalist(args['directory'], load_path)
    print('SUCCESS: Data loaded.')

    # create dataset splits (train, val, test) on device given args
    data_splits = prepare_splits_forCV(data_list, args)

    # run the sweep
    run_sweep(data_splits, args)