import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


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
