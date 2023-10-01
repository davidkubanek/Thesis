# %%
import json
from load_data import *
from support_funcs import *
from sweep_run import *
from train import *
from random_forest import *

def finish_train(args):

    '''
    Load data
    '''

    # Specify the path where you saved the dictionary
    load_path = args['directory'] + 'final/datalist_no_out.pkl'  # no_out.pkl'

    print('\nLoading data...')
    data_list, assay_groups, assay_order = load_datalist(args['directory'], load_path)
    print('SUCCESS: Data loaded.')

    '''
    Config
    '''

    # data parameters
    args['num_node_features'] = 79 # number of node features in graph representation
    args['grover_fp_dim'] = 5000 # dim of grover fingerprints
    args['fp_dim'] = 2215  # dim of fingerprints


    # create dataset splits (train, val, test) on device given args
    data_splits = prepare_splits(data_list, args)

    args['best_auc'] = 0

    '''
    Run
    '''

    for assay in args['assays_list']:
        for model in args['models_list']:

            # assay parameters
            args['assay_list'] = assay
            args['num_assays'] = len(args['assay_list'])
            args['assays_idx'] = find_assay_indeces(
                args['assay_list'], assay_order)
            args['model'] = model


            if args['use_best_no_epochs'] is True:
                if args['model'] in ['LogReg', 'GCN_FP']:
                    args['num_epochs'] = 100

                if args['model'] in ['GCN_base', 'FP', 'GROVER_FP']:
                    args['num_epochs'] = 120

                if args['model'] in ['GCN', 'GROVER', 'GCN_FP_GROVER']:
                    args['num_epochs'] = 190
            

            if model == 'RF':
                run_RF(data_splits, assay_groups, args)
            
            else:

                print('\n\n====================================================')
                print('Finishing training for:')
                print('Assays:', args['assay_list'], '| Model:', args['model'])
                print('====================================================\n')

                pre_trained_epochs = args['pre_trained_epochs']
                # load in best hyperparameters from best run
                if len(args['assay_list']) > 1:
                    filename = 'ass' + \
                        '+'.join([assay for assay in args['assay_list']]) + \
                        '_' + args['model']
                else:
                    filename = 'ass' + args['assay_list'][0] + '_' + args['model']

                if model == 'LogReg':
                    args['use_best_hyperparams'] = False
                    if pre_trained_epochs > 0:
                        print('Hyperparameter sweep not defined for model \'LogReg\' so setting pre_trained_epochs to zero and use_best_hyperparams to False')
                        pre_trained_epochs = 0

                if pre_trained_epochs > 0:
                    # when running pre-trained model, optimal hyperparams are used
                    args['use_best_hyperparams'] = True 
                
                if args['use_best_hyperparams'] is True:
                    with open(args['directory']+'trained_models/best_hyperparams/'+filename, 'r') as file:
                            best_hyperparams = json.load(file)

                    # set with best hyperparams
                    args['batch_size'] = best_hyperparams['batch_size']
                    args['dropout'] = best_hyperparams['dropout']
                    args['hidden_channels'] = best_hyperparams['hidden_channels']

                    print('Best hyperparams used (batch_size, dropout, hidden_channels):', args['batch_size'],', ', args['dropout'],', ',
                        args['hidden_channels'])
                

                # create dataset from data_list
                dataloader = prepare_dataloader(data_splits, args)

                # train model
                exp = TrainManager(dataloader, args)
                # load saved model
                if pre_trained_epochs > 0:
                    folder = args['directory'] + 'trained_models/' + \
                        f'{pre_trained_epochs}epochs/'
                    exp.load_model(folder)
                    print(f'Model pre-trained on {pre_trained_epochs} epochs loaded. Continuing training...'+'\n')
                    
                else:
                    print('No pre-trained model loaded. Training from scratch...\n')

                # finish training
                exp.train(epochs=args['num_epochs'], log=args['verbose'],
                        wb_log=False, early_stop=True)
                # save model
                folder = args['directory'] + \
                    f'trained_models/{pre_trained_epochs + exp.curr_epoch}epochs/'
                # Check if directory exists, if not, create it
                if not os.path.exists(folder):
                    os.makedirs(folder)
                exp.save_model(folder)

                # eval on test set
                _, auc_test, precision_test, recall_test, f1_test = exp.eval(
                    dataloader['test'])

                # print('AUC test:', auc_test, '\nPrecision test:', precision_test,
                #       '\nRecall test:', recall_test, '\nF1 test:', f1_test)

                if args['num_assays'] > 1:
                    auc_test_macro = np.mean(auc_test)
                    precision_test_macro = np.mean(precision_test)
                    recall_test_macro = np.mean(recall_test)
                    f1_test_macro = np.mean(f1_test)
                else:
                    auc_test_macro = auc_test
                    precision_test_macro = precision_test
                    recall_test_macro = recall_test
                    f1_test_macro = f1_test

                if args['verbose'] is True:
                    print('saving results...\n\n')
                # save results
                filename = 'all_best_results.csv'
                folder = args['directory'] + 'results/'

                # if file exists
                if os.path.exists(os.path.join(
                        folder, filename)):
                    results_df = pd.read_csv(os.path.join(
                        folder, filename))
                else:
                    # create new
                    results_df = pd.DataFrame(columns=['assay', 'type', 'model_name', 'model_type', 'batch_size', 'dropout', 'hidden_dims', 'epochs', 'loss',
                                                    'auc_train', 'auc_val', 'auc_test', 'f1_train', 'f1_val', 'f1_test', 'precision_train', 'precision_val', 'precision_test', 'recall_train', 'recall_val', 'recall_test'])


                assay_name = '+'.join([assay for assay in args['assay_list']]) if len(
                    args['assay_list']) > 1 else args['assay_list'][0]

                model_type = 'multi-assay' if len(
                    args['assay_list']) > 1 else 'uni-assay'
                
            
                if set(assay).issubset(assay_groups['cell_based']):
                    assay_type = 'cell-based'
                elif set(assay).issubset(assay_groups['biochemical']):
                    assay_type = 'biochemical'
                else:
                    assay_type = 'mixed'


                # Append results to DataFrame
                new_results_df = pd.DataFrame({
                    'assay': [assay_name],
                    'type': [assay_type],
                    'model_name': [args['model']],
                    'model_type': [model_type],
                    'batch_size': [args['batch_size']],
                    'dropout': [args['dropout']],
                    'hidden_dims': [args['hidden_channels']],
                    'epochs': [pre_trained_epochs+exp.curr_epoch],

                    'loss': exp.eval_metrics['loss'][-1],

                    'auc_train': exp.eval_metrics['auc_train'][-1],
                    'auc_val': exp.eval_metrics['auc_test'][-1],
                    'auc_test': [auc_test_macro],

                    'f1_train': exp.eval_metrics['f1_train'][-1],
                    'f1_val': exp.eval_metrics['f1_test'][-1],
                    'f1_test': [f1_test_macro],

                    'precision_train': exp.eval_metrics['precision_train'][-1],
                    'precision_val': exp.eval_metrics['precision_test'][-1],
                    'precision_test': [precision_test_macro],

                    'recall_train': exp.eval_metrics['recall_train'][-1],
                    'recall_val': exp.eval_metrics['recall_test'][-1],
                    'recall_test': [recall_test_macro],
                })

                # add individual results for each assay
                if len(args['assay_list']) > 1:
                    for idx, auc_value in enumerate(auc_test):
                        new_results_df[f'auc_test_ass_{idx+1}'] = [auc_value]
                    for idx, f1_value in enumerate(f1_test):
                        new_results_df[f'f1_test_ass_{idx+1}'] = [f1_value]
                    for idx, precision_value in enumerate(precision_test):
                        new_results_df[f'precision_test_ass_{idx+1}'] = [precision_value]
                    for idx, recall_value in enumerate(recall_test):
                        new_results_df[f'recall_test_ass_{idx+1}'] = [recall_value]

                results_df = pd.concat(
                    [results_df, new_results_df], ignore_index=True)

                # save updated results
                results_df.to_csv(os.path.join(folder, filename), index=False)
