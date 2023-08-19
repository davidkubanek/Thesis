
import cross_val
import wandb

import importlib
# this method of import ensures that when support scripts are updated, the changes are imported in this script
importlib.reload(cross_val)
from cross_val import *

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
                'AUC test': np.mean(CV_results['auc_test']),
                'F1 train': np.mean(CV_results['f1_train']),
                'F1 test': np.mean(CV_results['f1_test']),
                'Precision train': np.mean(CV_results['precision_train']),
                'Precision test': np.mean(CV_results['precision_test']),
                'Recall train': np.mean(CV_results['recall_train']),
                'Recall test': np.mean(CV_results['recall_test'])})
        