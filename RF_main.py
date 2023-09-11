# %%
import time
import numpy as np
from load_data import *
from support_funcs import *
from sweep_run import *
from train import *
from random_forest import *

import torch
import wandb
# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
'''
Load data
'''
directory = 'data/'
# directory = '/content/drive/MyDrive/Thesis/Data/'
# directory = '/Volumes/Kubánek UCL/Data/Thesis MSc/PubChem Data/'
# directory = 'Data/PubChem Data/'

# Specify the path where you saved the dictionary
load_path = directory + 'final/datalist_no_out.pkl'  # no_out.pkl'

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

# training parameters
args['model'] = 'RF'  # 'GCN', 'GCN_FP', 'FP', 'GROVER', 'GROVER_FP'

if args['model'] == 'RF':
    args['device'] = 'cpu'
# create dataset splits (train, val, test) on device given args
data_splits = prepare_splits(data_list, args)

args['best_auc'] = 0

# %%
'''
Run
'''
# %%
wandb.login(key='69f641df6e6f0934ab302070cf0b3bcd5399ddd3')
# API KEY: 69f641df6e6f0934ab302070cf0b3bcd5399ddd3
for assay in ['720582']:

    now = time.time()

    # assay parameters
    args['assay_list'] = [assay]
    args['num_assays'] = 1
    args['assays_idx'] = find_assay_indeces(args['assay_list'], assay_order)

    args['model'] = 'RF'

    # Create a custom run name dynamically
    run_name = f"ass{assay}_{args['model']}"
    run = wandb.init(
        name=run_name,
        # Set the project where this run will be logged
        project="GDL_molecular_activity_prediction_BASE",
        # Track hyperparameters and run metadata
        config={
            'num_data_points': args['num_data_points'],
            'assays': 'cell_based_high_hr',
            'num_assays': args['num_assays'],

            'model': args['model'],
        })

    print('\n\n====================================================')
    print('Assays:', args['assay_list'], '| Model:', args['model'])
    print('====================================================\n')

    X_train, y_train = prep_data_matrix(data_splits['train'], args)

    print('train shape:', X_train.shape, y_train.shape)

    clf = RandomForestClassifier(n_estimators=100)
    print('Fitting RF...')
    clf.fit(X_train, y_train.flatten())

    # print time taken in minutes
    time_elapsed = time.time() - now

    print('Predicting...')
    X_test, y_test = prep_data_matrix(data_splits['test'], args)

    print('test shape:', X_test.shape, y_test.shape)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print('Evaluating...')
    accuracy, auc, precision, recall, f1 = eval_RF(y_test, y_pred)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"AUC: {auc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    wandb.log({'AUC test': auc, 'F1 test': f1,
               'Precision test': precision, 'Recall test': recall})

    # eval train performance
    # y_pred = clf.predict(X_train)

    # accuracy, auc, precision, recall, f1 = eval_RF(y_train, y_pred)

    # print(f"\nAccuracy Train: {accuracy * 100:.2f}%")
    # print(f"AUC Train: {auc * 100:.2f}%")
    # print(f"Precision Train: {precision * 100:.2f}%")
    # print(f"Recall Train: {recall * 100:.2f}%")
    # print(f"F1 Score Train: {f1 * 100:.2f}%")

    # wandb.log({'AUC train': auc, 'F1 train': f1,
    #            'Precision train': precision, 'Recall train': recall})
