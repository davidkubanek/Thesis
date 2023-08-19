from sklearn.model_selection import ShuffleSplit

import train
import load_data

import importlib
# this method of import ensures that when support scripts are updated, the changes are imported in this script
importlib.reload(train)
importlib.reload(load_data)

from train import *
from load_data import *


def cross_val(data_splits, args):
    
    args['best_auc'] = 0
    # 3-fold cross-validation
    ss = ShuffleSplit(n_splits=3, test_size=0.25)

    CV_results = {'loss': [], 'auc_train': [], 'auc_test': [], 'f1_train':[], 'f1_test': [], 'precision_train': [], 'precision_test': [], 'recall_train': [], 'recall_test': []}

    for fold, (train_index, val_index) in enumerate(ss.split(data_splits['train-val'])):
        # move data to cuda only now and over-write in each fold to fit into memory
        data_splits['train'] = [data_splits['train-val'][i].to(args['device']) for i in train_index]
        data_splits['val'] = [data_splits['train-val'][i].to(args['device']) for i in val_index]

        args['fold'] = fold
        print('\nFold:', fold)

        # create dataset from data_splits
        dataloader = prepare_dataloader(data_splits, args)

        # train model
        exp = TrainManager(dataloader, args)
        exp.train(epochs=20, log=True, wb_log=False, early_stop=False)

        # save metrics for fold
        CV_results['loss'].append(exp.eval_metrics['loss'][-1])
        CV_results['auc_train'].append(exp.eval_metrics['auc_train'][-1])
        CV_results['auc_test'].append(exp.eval_metrics['auc_test'][-1])
        CV_results['f1_train'].append(exp.eval_metrics['f1_train'][-1])
        CV_results['f1_test'].append(exp.eval_metrics['f1_test'][-1])
        CV_results['precision_train'].append(exp.eval_metrics['precision_train'][-1])
        CV_results['precision_test'].append(exp.eval_metrics['precision_test'][-1])
        CV_results['recall_train'].append(exp.eval_metrics['recall_train'][-1])
        CV_results['recall_test'].append(exp.eval_metrics['recall_test'][-1])
        # save fold metrics into file
        folder = args['directory'] + 'CV_results/'
        exp.save_results(folder, save_logs=True)

        if exp.eval_metrics['auc_test'][-1] > args['best_auc']:
            args['best_auc'] = exp.eval_metrics['auc_test'][-1]
        
    return CV_results
