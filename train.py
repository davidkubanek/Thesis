

from models import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import wandb
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import time
import pandas as pd
import os
import numpy as np


class TrainManager:

    def __init__(self, dataloader, args, model=None):

        self.args = args
        self.num_assays = args['num_assays']
        self.num_node_features = args['num_node_features']
        self.hidden_channels = args['hidden_channels']

        if not model:
            # initialize model depending on model type
            if args['model'] in ['GCN_base', 'GCN_base_FP']:
                self.model = GCN(args)
            elif args['model'] in ['FP', 'GROVER', 'GROVER_FP']:
                self.model = MLP(args)
            elif args['model'] in ['LogReg']:
                self.model = LogisticRegression(args['fp_dim'], args['num_assays'])
            elif args['model'] in ['GCN', 'GCN_FP', 'GCN_FP_GROVER']:
                self.model = GCN_MLP(args)
        else:
            self.model = model

        self.model.to(args['device'])

        if args['verbose'] is True:
            print("Model is on device:", next(self.model.parameters()).device)
            total_params = sum(p.numel()
                            for p in self.model.parameters() if p.requires_grad)
            print(f'Total number of parameters: {total_params}')
            print('Model:', args['model'], '| Assays:', args['assay_list'])

        self.dataloader = dataloader

        self.optimizer = Adam(self.model.parameters(), lr=args['lr'])
        # decay learning rate
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 'min', factor=args['lr_decay_factor'])
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

        self.criterion = nn.BCEWithLogitsLoss()

        self.curr_epoch = 0

        # logging
        self.eval_metrics = {}
        self.eval_metrics['loss'] = []
        self.eval_metrics['acc_train'] = []
        self.eval_metrics['acc_test'] = []
        self.eval_metrics['auc_train'] = []
        self.eval_metrics['auc_test'] = []
        self.eval_metrics['precision_train'] = []
        self.eval_metrics['precision_test'] = []
        self.eval_metrics['recall_train'] = []
        self.eval_metrics['recall_test'] = []
        self.eval_metrics['f1_train'] = []
        self.eval_metrics['f1_test'] = []

        self.eval_metrics['auc_train_each'] = []
        self.eval_metrics['auc_test_each'] = []
        self.eval_metrics['precision_train_each'] = []
        self.eval_metrics['precision_test_each'] = []
        self.eval_metrics['recall_train_each'] = []
        self.eval_metrics['recall_test_each'] = []
        self.eval_metrics['f1_train_each'] = []
        self.eval_metrics['f1_test_each'] = []

    def train(self, epochs=100, log=False, wb_log=False, early_stop=False):
        '''
        Train the model for a given number of epochs.
        '''

        self.wb_log = wb_log

        for epoch in range(epochs):

            self.model.train()
            cum_loss = 0
            start_time = time.time()

            # Iterate in batches over the training dataset
            for data in tqdm(self.dataloader['train'], desc=f'Epoch [{self.curr_epoch+1}/{epochs}]', total=int(len(self.dataloader['train'].dataset)/self.args['batch_size']), disable=True):

                # clear gradients efficiently
                for param in self.model.parameters():
                    param.grad = None

                # forward pass based on model type
                if self.args['model'] in ['GCN_base', 'GCN']:
                    out = self.model(data.x, data.edge_index, data.batch)
                elif self.args['model'] in ['GCN_base_FP', 'GCN_FP']:
                    out = self.model(data.x, data.edge_index,
                                     data.batch, fp=data.fp)
                elif self.args['model'] == 'GCN_FP_GROVER':
                    out = self.model(data.x, data.edge_index, data.batch,
                                     fp=data.fp, grover=data.grover_fp)
                elif self.args['model'] in ['FP', 'GROVER', 'GROVER_FP']:
                    out = self.model(data)
                elif self.args['model'] in ['LogReg']:
                    out = self.model(data.fp)

                # data.y = data.y.unsqueeze(1)
                # print('data.y:',data.y.shape)
                # print('idx:', self.args['assays_idx'])
                # Compute the loss. (sigmoid inherent in loss)
                loss = self.criterion(out, data.y[:, self.args['assays_idx']])
                loss.backward()  # Derive gradients.
                self.optimizer.step()  # Update parameters based on gradients.
                cum_loss += loss.item()

            self.eval_metrics['loss'].append(
                cum_loss/len(self.dataloader['train']))
            if wb_log is True:
                wandb.log({'epoch': self.curr_epoch,
                          "loss": cum_loss/len(self.dataloader['train'])})

            epoch_time = time.time() - start_time

            if ((epoch+1) % (epochs/2) == 0) or (epoch == 0) or ((epoch+1) > (epochs-3)):
                # evaluate
                acc_train, auc_train, precision_train, recall_train, f1_train = self.eval(
                    self.dataloader['train'])
                acc_test, auc_test, precision_test, recall_test, f1_test = self.eval(
                    self.dataloader['test'])

                if self.args['num_assays'] > 1:  # multi-assay

                    # save results for individual assay
                    self.eval_metrics['auc_train_each'].append(auc_train)
                    self.eval_metrics['auc_test_each'].append(auc_test)
                    self.eval_metrics['precision_train_each'].append(
                        precision_train)
                    self.eval_metrics['precision_test_each'].append(
                        precision_test)
                    self.eval_metrics['recall_train_each'].append(recall_train)
                    self.eval_metrics['recall_test_each'].append(recall_test)
                    self.eval_metrics['f1_train_each'].append(f1_train)
                    self.eval_metrics['f1_test_each'].append(f1_test)
                    # get macro average of metrics
                    auc_train = np.mean(auc_train)
                    auc_test = np.mean(auc_test)
                    f1_train = np.mean(f1_train)
                    f1_test = np.mean(f1_test)
                    precision_train = np.mean(precision_train)
                    precision_test = np.mean(precision_test)
                    recall_train = np.mean(recall_train)
                    recall_test = np.mean(recall_test)

                self.eval_metrics['acc_train'].append(acc_train)
                self.eval_metrics['acc_test'].append(acc_test)
                self.eval_metrics['auc_train'].append(auc_train)
                self.eval_metrics['auc_test'].append(auc_test)
                self.eval_metrics['precision_train'].append(precision_train)
                self.eval_metrics['precision_test'].append(precision_test)
                self.eval_metrics['recall_train'].append(recall_train)
                self.eval_metrics['recall_test'].append(recall_test)
                self.eval_metrics['f1_train'].append(f1_train)
                self.eval_metrics['f1_test'].append(f1_test)

                
                if log:
                    print(
                        f'Epoch: {self.curr_epoch+1}, Loss: {loss.item():.4f}, Train AUC: {auc_train:.4f}, Test AUC: {auc_test:.4f}')
                    print(
                        f'                        Train F1: {f1_train:.4f}, Test F1: {f1_test:.4f}')
                    
            if early_stop and (epoch+1) in [23, 24, 25]:
                # evaluate
                acc_test, auc_test, precision_test, recall_test, f1_test = self.eval(
                    self.dataloader['test'])

                if self.args['num_assays'] > 1:  # multi-assay
                    # save results for individual assay
                    self.eval_metrics['auc_test_each'].append(auc_test)
                    self.eval_metrics['precision_test_each'].append(precision_test)
                    self.eval_metrics['recall_test_each'].append(recall_test)
                    self.eval_metrics['f1_test_each'].append(f1_test)
                    # get macro average of metrics
                    auc_test = np.mean(auc_test)
                    f1_test = np.mean(f1_test)
                    precision_test = np.mean(precision_test)
                    recall_test = np.mean(recall_test)

                self.eval_metrics['auc_test'].append(auc_test)
                self.eval_metrics['precision_test'].append(precision_test)
                self.eval_metrics['recall_test'].append(recall_test)
                self.eval_metrics['f1_test'].append(f1_test)

                best_latest_acu = np.max(self.eval_metrics['auc_test'][-3:])

                # if (epoch+1) == 25:
                #     print('Epochs: 23,24,25, AUC test: ',
                #           self.eval_metrics['auc_test'][-3:])
                #     print('         Current best AUC: ', self.args["best_auc"])
                if (epoch+1) == 25 and (best_latest_acu < 0.51) and (auc_test < self.args["best_auc"]):
                    if log:
                        print(
                            f'Early stopping at AUC test: {auc_test}, since best AUC test found: {self.args["best_auc"]}')
                    self.curr_epoch += 1
                    # end training
                    break

            self.curr_epoch += 1
            self.scheduler.step()
        
    def eval(self, loader):
        '''
        Evaluate the model on a given dataset (train/val/test).
        '''
        start_time = time.time()

        self.model.eval()

        # print("Model is on device for eval:", next(exp.model.parameters()).device)

        correct = 0

        gts = []
        preds = []
        with torch.no_grad():
            # Iterate in batches over the training/test dataset.
            for data in loader:

                # forward pass based on model type
                if self.args['model'] in ['GCN_base', 'GCN']:
                    out = self.model(data.x, data.edge_index, data.batch)
                elif self.args['model'] in ['GCN_base_FP', 'GCN_FP']:
                    out = self.model(data.x, data.edge_index,
                                     data.batch, fp=data.fp)
                elif self.args['model'] == 'GCN_FP_GROVER':
                    out = self.model(data.x, data.edge_index, data.batch,
                                     fp=data.fp, grover=data.grover_fp)
                elif self.args['model'] in ['FP', 'GROVER', 'GROVER_FP']:
                    out = self.model(data)
                elif self.args['model'] in ['LogReg']:
                    out = self.model(data.fp)

                # convert out to binary
                pred = torch.round(torch.sigmoid(out))
                preds.append(torch.round(torch.sigmoid(out)).tolist())
                gts.append(data.y[:, self.args['assays_idx']].tolist())
                # print('pred:', pred)
                # print('data.y:', data.y)
                # print('data.y eval:',data.y.shape)
                # data.y = data.y.unsqueeze(1)
                # Check against ground-truth labels.
                correct += int((pred ==
                               data.y[:, self.args['assays_idx']]).sum())

        preds = [b[i] for b in preds for i in range(len(b))]
        gts = [b[i] for b in gts for i in range(len(b))]

        if self.args['num_assays'] > 1:  # multi-assay
            auc = roc_auc_score(gts, preds, average=None)
            f1 = f1_score(gts, preds, average=None, zero_division=0)
            # Calculate macro-averaged precision, recall, and F1 Score
            precision = precision_score(
                gts, preds, average=None, zero_division=0)
            recall = recall_score(gts, preds, average=None, zero_division=0)
        else:
            auc = roc_auc_score(gts, preds, average='macro')
            f1 = f1_score(gts, preds, average='macro', zero_division=0)
            # Calculate macro-averaged precision, recall, and F1 Score
            precision = precision_score(
                gts, preds, average='macro', zero_division=0)
            recall = recall_score(gts, preds, average='macro', zero_division=0)

        # Derive ratio of correct predictions.
        acc = correct / (len(loader.dataset) * self.args['num_assays'])

        self.eval_time = time.time() - start_time

        if self.wb_log is True:
            wandb.log({'epoch': self.curr_epoch, "eval time": self.eval_time})

        return acc, auc, precision, recall, f1

    def analyze(self):
        '''
        Plot the model performance.
        '''

        # plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(self.eval_metrics['loss'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Losses')

        ax2.plot(self.eval_metrics['auc_train'], label='train')
        ax2.plot(self.eval_metrics['auc_test'], label='test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.set_title('Area Under Curve')
        ax2.legend()
        # make main title for the whole plot
        if self.args['model'] in ['GCN_base', 'GCN_base_FP']:
            plt.suptitle(f'Model: {self.args["model"]} | Node feats: {self.args["num_node_features"]}, Hidden dim: {self.args["hidden_channels"]}, Dropout: {self.args["dropout"]}, Num data points: {self.args["num_data_points"]}, Num assays: {self.args["num_assays"]}, Num epochs: {self.curr_epoch}')
        elif self.args['model'] in ['FP', 'GROVER', 'GROVER_FP']:
            plt.suptitle(f'Model: {self.args["model"]} | Num layers: {self.args["num_layers"]}, Hidden dim: {self.args["hidden_channels"]}, Dropout: {self.args["dropout"]}, Num data points: {self.args["num_data_points"]}, Num assays: {self.args["num_assays"]}, Num epochs: {self.curr_epoch}')
        plt.show()

    def save_results(self, folder, save_logs=True):

        if save_logs:


            filename = 'ass' + self.args['assay_list'][0] + '_results.csv'
            if len(self.args['assay_list']) > 1:
                filename = 'ass' + \
                    '+'.join([assay for assay in self.args['assay_list']]) + \
                    '_results.csv'

            # if file exists
            if os.path.exists(os.path.join(
                    folder, filename)):
                results_df = pd.read_csv(os.path.join(
                    folder, filename))
            else:
                # create new
                results_df = pd.DataFrame(columns=['model_name', 'batch_size', 'dropout', 'hidden_dims', 'fold', 'epochs', 'loss',
                                          'auc_train', 'auc_test', 'f1_train', 'f1_test', 'precision_train', 'precision_test', 'recall_train', 'recall_test'])

            # Append results to DataFrame
            new_results_df = pd.DataFrame({
                'model_name': [self.args['model']],
                'batch_size': [self.args['batch_size']],
                'dropout': [self.args['dropout']],
                'hidden_dims': [self.args['hidden_channels']],
                'fold': [self.args['fold']],
                # take the mean of the last 3 epochs
                'epochs': [self.curr_epoch],
                'loss': np.mean(self.eval_metrics['loss'][-3]),
                'auc_train': np.mean(self.eval_metrics['auc_train'][-3]),
                'auc_test': np.mean(self.eval_metrics['auc_test'][-3]),
                'f1_train': np.mean(self.eval_metrics['f1_train'][-3]),
                'f1_test': np.mean(self.eval_metrics['f1_test'][-3]),
                'precision_train': np.mean(self.eval_metrics['precision_train'][-3]),
                'precision_test': np.mean(self.eval_metrics['precision_test'][-3]),
                'recall_train': np.mean(self.eval_metrics['recall_train'][-3]),
                'recall_test': np.mean(self.eval_metrics['recall_test'][-3]),
            })



            results_df = pd.concat(
                [results_df, new_results_df], ignore_index=True)

            # save updated results
            results_df.to_csv(os.path.join(folder, filename), index=False)

    def save_model(self, folder):
        if self.args['num_assays'] > 1:
            filename = 'ass' + \
                '+'.join([assay for assay in self.args['assay_list']]) + \
                '_' + self.args['model']
        else:
            filename = 'ass' + \
                self.args['assay_list'][0] + '_' + self.args['model']
        print('saving trained model...')
        torch.save(self.model.state_dict(),
                   os.path.join(folder, filename+'.pt'))

    def load_model(self, folder):
        if self.args['num_assays'] > 1:
            filename = 'ass' + \
                '+'.join([assay for assay in self.args['assay_list']]) + \
                '_' + self.args['model']
        else:
            filename = 'ass' + \
                self.args['assay_list'][0] + '_' + self.args['model']
        print('loading model...')
        self.model.load_state_dict(torch.load(
            os.path.join(folder, filename+'.pt')))
