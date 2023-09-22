from load_data import *
from train import *
from sklearn.model_selection import ShuffleSplit
import numpy as np


def cross_val(data_splits, args):

    # 3-fold cross-validation
    ss = ShuffleSplit(n_splits=args['num_folds'], test_size=0.25)

    CV_results = {'loss': [], 'auc_train': [], 'auc_test': [], 'f1_train': [], 'f1_test': [
    ], 'precision_train': [], 'precision_test': [], 'recall_train': [], 'recall_test': []}

    for fold, (train_index, val_index) in enumerate(ss.split(data_splits['train-val'])):
        # move data to cuda only now and over-write in each fold to fit into memory
        data_splits['train'] = [data_splits['train-val']
                                [i].to(args['device']) for i in train_index]
        data_splits['val'] = [data_splits['train-val']
                              [i].to(args['device']) for i in val_index]

        args['fold'] = fold
        print('\nFold:', fold+1)

        # create dataset from data_splits
        dataloader = prepare_dataloader(data_splits, args)

        # train model
        exp = TrainManager(dataloader, args)
        exp.train(epochs=args['num_epochs'], log=True,
                  wb_log=False, early_stop=True)

        # save metrics for fold
        # take the mean of the last 3 epochs
        CV_results['loss'].append(np.mean(exp.eval_metrics['loss'][-3]))
        CV_results['auc_train'].append(np.mean(exp.eval_metrics['auc_train'][-3]))
        CV_results['auc_test'].append(
            np.mean(exp.eval_metrics['auc_test'][-3:]))
        CV_results['f1_train'].append(np.mean(exp.eval_metrics['f1_train'][-3]))
        CV_results['f1_test'].append(np.mean(exp.eval_metrics['f1_test'][-3]))
        CV_results['precision_train'].append(np.mean(
            exp.eval_metrics['precision_train'][-3]))
        CV_results['precision_test'].append(np.mean(
            exp.eval_metrics['precision_test'][-3]))
        CV_results['recall_train'].append(np.mean(exp.eval_metrics['recall_train'][-3]))
        CV_results['recall_test'].append(np.mean(exp.eval_metrics['recall_test'][-3]))
        if args['num_assays'] > 1:
            CV_results['auc_train_each'] = exp.eval_metrics['auc_train_each'][-1]
            CV_results['auc_test_each'] = exp.eval_metrics['auc_test_each'][-1]
            CV_results['f1_train_each'] = exp.eval_metrics['f1_train_each'][-1]
            CV_results['f1_test_each'] = exp.eval_metrics['f1_test_each'][-1]
            CV_results['precision_train_each'] = exp.eval_metrics['precision_train_each'][-1]
            CV_results['precision_test_each'] = exp.eval_metrics['precision_test_each'][-1]
            CV_results['recall_train_each'] = exp.eval_metrics['recall_train_each'][-1]
            CV_results['recall_test_each'] = exp.eval_metrics['recall_test_each'][-1]


        # save fold metrics into file
        folder = args['directory'] + 'CV_results/'
        # Check if directory exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        exp.save_results(folder, save_logs=True)

        del data_splits['train']
        del data_splits['val']

    return CV_results, exp.model.state_dict()
