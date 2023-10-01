# %%
from sweep_run import *
from support_funcs import *
from load_data import *


def sweep(hyperparams_dict, args):
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

    # generate random combinations of hyperparameters from options above
    random_combinations_dicts = random_grid_search(hyperparams_dict, args['samples'])

    # create dataset splits (train, val, test) on device given args
    data_splits = prepare_splits_forCV(data_list, args)

    '''
    Sweeps
    '''

    for assay in args['assays_list']:
        for model in args['models_list']:
            # config
            args['assay_list'] = assay
            args['num_assays'] = len(args['assay_list'])
            args['assays_idx'] = find_assay_indeces(
                args['assay_list'], assay_order)
            args['model'] = model
            args['best_auc'] = 0

            if model not in ['GCN_base', 'FP', 'GROVER_FP', 'GCN', 'GCN_FP', 'GCN_FP_GROVER']:
                print('Sweep not implemented for this model.')
                print('Exiting...')
                return

            print('\n\n====================================================')
            print('Running hyperparameter sweep for:')
            print('Assays:', args['assay_list'], '| Model:', args['model'])
            print('====================================================\n')



            for s in range(args['samples']):

                args['batch_size'] = random_combinations_dicts[s]['batch_size']
                args['dropout'] = random_combinations_dicts[s]['dropout']
                args['hidden_channels'] = random_combinations_dicts[s]['hidden_channels']

                if args['verbose'] is True:
                    print('--------------------')
                    print('Sample no:', s+1)
                    print(' hyperparams (batch_size, dropout, hidden_channels):', args['batch_size'], ', ', args['dropout'], ', ', args['hidden_channels'])
                    print('\nCurrent best AUC:', args['best_auc'])
                args['best_auc'] = run_sweep(data_splits, args)

