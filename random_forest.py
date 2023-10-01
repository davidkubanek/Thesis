import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from load_data import *
from support_funcs import *
from sweep_run import *
from train import *

def prep_data_matrix(data_split, args):
    X_train, y_train = [], []

    for data in data_split:
        # reshape fp to batch_size x fp_dim
        # fp = data.fp.reshape(data.shape[0], -1)
        # y = data.y.reshape(data.shape[0], -1)
        # Flatten and convert to numpy
        X_train.append(data.fp.numpy())
        y_train.append(data.y.numpy().reshape(-1, 1).flatten())

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)[:, args['assays_idx']]

    return X_train, y_train


def eval_RF(y_test, y_pred):

    accuracy = np.mean(y_test == y_pred)
    auc = roc_auc_score(y_test, y_pred)
    # Calculate macro-averaged precision, recall, and F1 Score
    precision = precision_score(
        y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return accuracy, auc, precision, recall, f1


def run_RF(data_splits, assay_groups, args):

    print('\n\n====================================================')
    print('Assays:', args['assay_list'], '| Model:', args['model'])
    print('====================================================\n')

    if args['num_assays']>1:
        print('RF for multiple assays is not implemented')
        print('Exiting...')
        return

    X_train, y_train = prep_data_matrix(data_splits['train'], args)

    print('train shape:', X_train.shape, y_train.shape)

    clf = RandomForestClassifier(n_estimators=100)
    print('Fitting RF...')
    clf.fit(X_train, y_train.flatten())


    print('Predicting...')
    X_test, y_test = prep_data_matrix(data_splits['test'], args)

    print('test shape:', X_test.shape, y_test.shape)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print('Evaluating...')
    accuracy, auc, precision, recall, f1 = eval_RF(y_test, y_pred)

    # print(f"\nAccuracy: {accuracy * 100:.2f}%")
    # print(f"AUC: {auc * 100:.2f}%")
    # print(f"Precision: {precision * 100:.2f}%")
    # print(f"Recall: {recall * 100:.2f}%")
    # print(f"F1 Score: {f1 * 100:.2f}%")

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
    

    if set(args['assay_list']).issubset(assay_groups['cell_based']):
        assay_type = 'cell-based'
    elif set(args['assay_list']).issubset(assay_groups['biochemical']):
        assay_type = 'biochemical'
    else:
        assay_type = 'mixed'


    # Append results to DataFrame
    new_results_df = pd.DataFrame({
        'assay': [assay_name],
        'type': [assay_type],
        'model_name': [args['model']],
        'model_type': [model_type],

    
        'auc_test': [auc],

        'f1_test': [f1],

        'precision_test': [precision],

        'recall_test': [recall],
    })

    # # add individual results for each assay
    # if len(args['assay_list']) > 1:
    #     for idx, auc_value in enumerate(auc_test):
    #         new_results_df[f'auc_test_ass_{idx+1}'] = [auc_value]
    #     for idx, f1_value in enumerate(f1_test):
    #         new_results_df[f'f1_test_ass_{idx+1}'] = [f1_value]
    #     for idx, precision_value in enumerate(precision_test):
    #         new_results_df[f'precision_test_ass_{idx+1}'] = [precision_value]
    #     for idx, recall_value in enumerate(recall_test):
    #         new_results_df[f'recall_test_ass_{idx+1}'] = [recall_value]

    results_df = pd.concat(
        [results_df, new_results_df], ignore_index=True)

    # save updated results
    results_df.to_csv(os.path.join(folder, filename), index=False)

