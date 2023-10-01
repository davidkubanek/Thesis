
from load_data import *
from cross_val import *
import json


def run_sweep(data_splits, args):

    # train model with cross-validation
    CV_results, model_weights = cross_val(data_splits, args)

    # save the performance of the best model configuration
    if np.mean(CV_results['auc_test']) > args['best_auc']:
        args['best_auc'] = np.mean(CV_results['auc_test'])
        
        if args['verbose'] is True:
            print('New best AUC:', args['best_auc'], '\n')

        # save the best model
        folder = args['directory'] + 'trained_models/'+ str(args['num_epochs']) +'epochs/'

        filename = 'ass' + args['assay_list'][0] + '_' + args['model']
        if len(args['assay_list']) > 1:
            filename = 'ass' + \
                '+'.join([assay for assay in args['assay_list']]) + \
                '_' + args['model']

        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(model_weights,
                    os.path.join(folder, filename+'.pt'))
        
        # save hyperparameter configuration
        folder = args['directory'] + 'trained_models/best_hyperparams/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        best_config = {
            'batch_size': args['batch_size'],
            'dropout': args['dropout'],
            'hidden_channels': args['hidden_channels']
        }

        with open(folder+filename, 'w') as file:
            json.dump(best_config, file)
        
        if args['verbose'] is True:
            print('saving new best trained model...')

        
    return args['best_auc']

