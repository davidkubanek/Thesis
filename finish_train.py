# %%
import json
from load_data import *
from support_funcs import *
from sweep_run import *
from train import *

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
args['assay_start'] = 0  # which assay to start from
# number of node features in graph representation
args['num_node_features'] = 79
# grover_fp['fps'][0].shape[0] # None  # dim of grover fingerprints
args['grover_fp_dim'] = 5000
args['fp_dim'] = 2215  # dim of fingerprints


# training parameters
args['num_layers'] = 3  # number of layers in MLP
args['num_epochs'] = 120
args['lr'] = 0.01


# create dataset splits (train, val, test) on device given args
data_splits = prepare_splits(data_list, args)

args['best_auc'] = 0

# %%
'''
Run
'''
# %%

with open(directory + 'trained_models/best_run_names.json', 'r') as file:
    best_run_names = json.load(file)


wandb.login(key='69f641df6e6f0934ab302070cf0b3bcd5399ddd3')
# API KEY: 69f641df6e6f0934ab302070cf0b3bcd5399ddd3

# '2797', '2796', '1979', '602248', '1910', '602274', '720582', '1259313', '624204', '652039']
# for assay in ['2796', '1979', '602248', '1910', '602274', '720582', '1259313', '624204', '652039']:  # '602274'
# , '2796', '1979', '602248', '1910', '602274', '720582', '1259313', '624204', '652039']:
for assay in ['2796', '1979', '602248', '1910', '602274', '720582', '1259313', '624204', '652039']:
    for model in ['GCN_MLP_FP_GROVER']:

        # assay parameters
        args['assay_list'] = [assay]
        args['num_assays'] = len(args['assay_list'])
        args['assays_idx'] = find_assay_indeces(
            args['assay_list'], assay_order)
        args['model'] = model

        print('\n\n====================================================')
        print('Assays:', args['assay_list'], '| Model:', args['model'])
        print('====================================================\n')

        # load in best hyperparameters from best run
        api = wandb.Api()
        if len(args['assay_list']) > 1:
            run_id = best_run_names['+'.join(
                [assay for assay in args['assay_list']])][model]

        else:
            run_id = best_run_names[assay][model]
        run = api.run(f"GDL_molecular_activity_prediction_SWEEPS/{run_id}")
        args['dropout'] = run.config['dropout']
        args['batch_size'] = run.config['batch_size']
        args['hidden_channels'] = run.config['hidden_channels']

        print('best hyperparams:', run.config['dropout'], run.config['batch_size'],
              run.config['hidden_channels'])

        args['num_epochs'] = 50
        pre_trained_epochs = 70
        args['num_layers'] = 3
        args['hidden_channels_conv'] = 64
        args['lr'] = 0.01

        # Create a custom run name dynamically
        # run_name = f"ass{args['assay_list'][0]}_{args['model']}_best"
        run = wandb.init(
            # name=run_name,
            # Set the project where this run will be logged
            project="GDL_molecular_activity_prediction_SWEEPS",
            # Track hyperparameters and run metadata
            config={
                'num_data_points': args['num_data_points'],
                'assays': 'cell_based_high_hr',
                'num_assays': args['num_assays'],

                'model': args['model'],
                'num_layers': args['num_layers'],
                'hidden_channels': args['hidden_channels'],
                'dropout': args['dropout'],
                'batch_size': args['batch_size'],
                'num_epochs': args['num_epochs'],
                'lr': args['lr'],
            })

        # create dataset from data_list
        dataloader = prepare_dataloader(data_splits, args)

        # train model
        exp = TrainManager(dataloader, args)
        # load saved model
        if pre_trained_epochs > 0:
            folder = args['directory'] + 'trained_models/' + \
                f'{pre_trained_epochs}epochs/'
            exp.load_model(folder)
            print(f'Pre-trained model on {pre_trained_epochs} epochs loaded.')
        else:
            print('No pre-trained model loaded.')

        # finish training
        exp.train(epochs=args['num_epochs'], log=True,
                  wb_log=True, early_stop=True)
        # save model
        folder = args['directory'] + \
            f'trained_models/{pre_trained_epochs + exp.curr_epoch}epochs/'
        # Check if directory exists, if not, create it
        if not os.path.exists(folder):
            os.makedirs(folder)
        exp.save_model(folder)

        print('Evaluating on test set...')
        # eval on test set
        _, auc_test, precision_test, recall_test, f1_test = exp.eval(
            dataloader['test'])

        print('AUC test:', auc_test, '\nPrecision test:', precision_test,
              '\nRecall test:', recall_test, '\nF1 test:', f1_test)

        if args['num_assays'] > 1:
            auc_test = np.mean(auc_test)
            precision_test = np.mean(precision_test)
            recall_test = np.mean(recall_test)
            f1_test = np.mean(f1_test)

        wandb.log({'AUC Test': auc_test,
                   'Precision Test': precision_test,
                   'Recall Test': recall_test,
                   'F1 Test': f1_test})

        print('Saving results...\n\n')
        # save results
        filename = 'all_best_results.csv'
        folder = args['directory'] + 'trained_models/'

        # if file exists
        if os.path.exists(os.path.join(
                folder, filename)):
            results_df = pd.read_csv(os.path.join(
                folder, filename))
        else:
            # create new
            results_df = pd.DataFrame(columns=['assay', 'type', 'model_name', 'batch_size', 'dropout', 'hidden_dims', 'epochs', 'loss',
                                               'auc_train', 'auc_val', 'auc_test', 'f1_train', 'f1_val', 'f1_test', 'precision_train', 'precision_val', 'precision_test', 'recall_train', 'recall_val', 'recall_test'])

        #
        assay_name = assay[0]+'+'+assay[1] if len(
            args['assay_list']) > 1 else args['assay_list'][0]

        if len(args['assay_list']) == 1:
            assay_type = 'cell-based' if assay in assay_groups['cell_based_high_hr'] else 'biochemical'
        else:
            assay_type = 'multi'

        # Append results to DataFrame
        new_results_df = pd.DataFrame({
            'assay': [assay_name],
            'type': [assay_type],
            'model_name': [args['model']],
            'batch_size': [args['batch_size']],
            'dropout': [args['dropout']],
            'hidden_dims': [args['hidden_channels']],
            'epochs': [pre_trained_epochs+exp.curr_epoch],

            'loss': exp.eval_metrics['loss'][-1],

            'auc_train': exp.eval_metrics['auc_train'][-1],
            'auc_val': exp.eval_metrics['auc_test'][-1],
            'auc_test': [auc_test],

            'f1_train': exp.eval_metrics['f1_train'][-1],
            'f1_val': exp.eval_metrics['f1_test'][-1],
            'f1_test': [f1_test],

            'precision_train': exp.eval_metrics['precision_train'][-1],
            'precision_val': exp.eval_metrics['precision_test'][-1],
            'precision_test': [precision_test],

            'recall_train': exp.eval_metrics['recall_train'][-1],
            'recall_val': exp.eval_metrics['recall_test'][-1],
            'recall_test': [recall_test],
        })

        results_df = pd.concat(
            [results_df, new_results_df], ignore_index=True)

        # save updated results
        results_df.to_csv(os.path.join(folder, filename), index=False)
