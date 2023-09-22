
from load_data import *
from cross_val import *
import json


def run_sweep(data_splits, args):

    # train model with cross-validation
    CV_results, model_weights = cross_val(data_splits, args)

    # log mean metrics of all folds to wandb
    # wandb.log({'loss': np.mean(CV_results['loss']),
    #             'AUC train': np.mean(CV_results['auc_train']),
    #             'AUC val': np.mean(CV_results['auc_test']),
    #             'F1 train': np.mean(CV_results['f1_train']),
    #             'F1 val': np.mean(CV_results['f1_test']),
    #             'Precision train': np.mean(CV_results['precision_train']),
    #             'Precision val': np.mean(CV_results['precision_test']),
    #             'Recall train': np.mean(CV_results['recall_train']),
    #             'Recall val': np.mean(CV_results['recall_test'])})
    # if args['num_assays'] > 1:
    #     wandb.log({'AUC_1 train': CV_results['auc_train_each'][0],
    #                 'AUC_2 train': CV_results['auc_train_each'][1],
    #                 'AUC_1 val': CV_results['auc_test_each'][0],
    #                 'AUC_2 val': CV_results['auc_test_each'][1],
    #                 'F1_1 train': CV_results['f1_train_each'][0],
    #                 'F1_2 train': CV_results['f1_train_each'][1],
    #                 'F1_1 val': CV_results['f1_test_each'][0],
    #                 'F1_2 val': CV_results['f1_test_each'][1],
    #                 'Precision_1 train': CV_results['precision_train_each'][0],
    #                 'Precision_2 train': CV_results['precision_train_each'][1],
    #                 'Precision_1 val': CV_results['precision_test_each'][0],
    #                 'Precision_2 val': CV_results['precision_test_each'][1],
    #                 'Recall_1 train': CV_results['recall_train_each'][0],
    #                 'Recall_2 train': CV_results['recall_train_each'][1],
    #                 'Recall_1 val': CV_results['recall_test_each'][0],
    #                 'Recall_2 val': CV_results['recall_test_each'][1]})

    # save the performance of the best model configuration
    if np.mean(CV_results['auc_test']) > args['best_auc']:
        args['best_auc'] = np.mean(CV_results['auc_test'])
        print('New best AUC:', args['best_auc'], '\n')

        # save the best model
        folder = args['directory'] + 'trained_models/70epochs/'

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
        
        print('saving new best trained model...')

        
    return args['best_auc']

