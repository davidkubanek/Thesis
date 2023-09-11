# %%
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from load_data import *
from support_funcs import *
from sweep_run import *
from train import *

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np


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
args['grover_fp_dim'] = 5000
args['fp_dim'] = 2215  # dim of fingerprints


# training parameters
args['num_layers'] = 3  # number of layers in MLP
args['num_epochs'] = 120
args['lr'] = 0.01


# create dataset splits (train, val, test) on device given args
data_splits = prepare_splits(data_list, args)


# %%

def prep_data_dataframe(data_split, args):
    X_train_list, y_train_list = [], []

    for data in data_split:
        X_train_list.append(data.fp.cpu().numpy())
        y_train_list.append(data.y.cpu().numpy().reshape(-1, 1).flatten())

    X_train_array = np.vstack(X_train_list)

    # Convert to dataframe
    X_train_df = pd.DataFrame(X_train_array)

    return X_train_df


df = prep_data_dataframe(data_splits['test'], args)


# %%
'''
Find optimal k for k-means clustering
'''
# Using the Elbow method to find the optimal number of clusters
wcss = []  # within-cluster sums of squares
clusters = 20
for i in range(1, clusters+1):  # trying up to 10 clusters, you can adjust this
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, clusters+1), wcss, marker='o', linestyle='--')
plt.title('K-means Clustering: Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# %%
'''
Use optimal k for k-means clustering
'''
for clusters in range(2, 21):
    # Based on the plot, choose the optimal number of clusters
    # this is just an example, replace with your choice based on the plot
    optimal_clusters = clusters
    kmeans = KMeans(n_clusters=optimal_clusters,
                    init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(df)

    # Adding cluster assignments back to the original dataframe
    # df['Cluster'] = y_kmeans

    # df = df.drop('Cluster', axis=1)
    kmeans.fit(df)
    inertia = kmeans.inertia_
    print(f"Inertia: {inertia}")

    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df, labels)
    print(
        f"Silhouette Score: {silhouette_avg}, where 1 is best and -1 is worst")

    db_score = davies_bouldin_score(df, labels)
    print(f"Davies-Bouldin Index: {db_score}, where 0 is best")

    # (unique, counts) = np.unique(labels, return_counts=True)
    # frequencies = np.asarray((unique, counts)).T
    # print(f"Cluster sizes: {frequencies}")

# %%
'''
find optimal number of components for PCA
'''


# Normalize the data

scaler = StandardScaler()
data_normalized = scaler.fit_transform(df)

# Fit PCA without limiting the number of components
pca = PCA()
pca.fit(data_normalized)


# Plot cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance)+1),
         cumulative_variance, marker='o', linestyle='--')
plt.title("Explained Variance by Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

# find where cumulative_variance is 0.9
n_components = np.where(cumulative_variance > 0.90)[0][0] + 1

print(
    f"Number of components required to explain 90% of variance: {n_components}")

# %%
'''
find optimal eps for DBSCAN
'''

# Assuming your dataset is named data_normalized
# Set the number of neighbors you want to look at (often equal to min_samples)
k = 100

# Compute the nearest neighbors
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(data_normalized)
distances, indices = nbrs.kneighbors(data_normalized)

# Sort distance values by ascending order and plot
distances_sorted = np.sort(distances, axis=0)[:, 1]
plt.plot(distances_sorted)
plt.xlabel('Data Points sorted by distance')
plt.ylabel(f'{k}-th nearest neighbor distance')
plt.title(f'{k}-distance Graph')
plt.grid(True)
plt.show()

# %%
'''
DBSCAN clustering
'''


# 2. Apply PCA for dimensionality reduction
# Reducing to 50 components, but this can be adjusted based on your dataset's characteristics
components = 1339
pca = PCA(n_components=components)
data_pca = pca.fit_transform(data_normalized)

# 3. Use DBSCAN for clustering
# The parameters eps and min_samples might need fine-tuning based on your data
eps = 50
print('eps:', eps, 'components:', components)
dbscan = DBSCAN(eps=eps, min_samples=5)
clusters = dbscan.fit_predict(data_pca)

# Add clusters to dataframe
# df['Cluster'] = clusters

# Evaluate clustering quality

# Filter out noise points (cluster label -1 in DBSCAN) for the evaluation metrics
filtered_data = data_pca[clusters != -1]
filtered_clusters = clusters[clusters != -1]

# print no. of clusters and noise points
print(
    f"Number of clusters: {len(set(filtered_clusters)) - (1 if -1 in filtered_clusters else 0)}")
# proportion of noise points
print(
    f"Proportion of noise points: {np.round(np.sum(clusters == -1)/len(clusters), 3)}")

# Silhouette Score
s_score = silhouette_score(filtered_data, filtered_clusters)
print(f"Silhouette Score: {s_score}")

# Davies-Bouldin Index
db_score = davies_bouldin_score(filtered_data, filtered_clusters)
print(f"Davies-Bouldin Index: {db_score}")

# %%
'''
Use HDSCAN for clustering
'''

# Assuming your data is in a pandas DataFrame named 'df'

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df)

# HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(data_normalized)

# Add the cluster labels to your original data for inspection
# df['Cluster'] = cluster_labels

# Print out cluster statistics
print(f"Number of clusters: {len(np.unique(cluster_labels))}")
print(f"Number of noise points: {(cluster_labels == -1).sum()}")

# You can explore other properties and methods provided by the clusterer object
# For example, to visualize the clustering hierarchy:
clusterer.condensed_tree_.plot(select_clusters=True)

# %%
with open(directory + 'trained_models/best_run_names.json', 'r') as file:
    best_run_names = json.load(file)

wandb.login(key='69f641df6e6f0934ab302070cf0b3bcd5399ddd3')

for assay in ['602274', '720582', '1259313', '624204', '652039']:
    for model in ['GROVER_FP']:

        # assay parameters
        args['assay_list'] = [assay]
        args['num_assays'] = len(args['assay_list'])
        args['assays_idx'] = find_assay_indeces(
            args['assay_list'], assay_order)
        args['model'] = model

        pre_trained_epochs = 120

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

        # TODO: add cluster assignment to data_splits['test']
        dataloader = {}
        dataloader['test'] = DataLoader(
            data_splits['test'], batch_size=args['batch_size'], shuffle=False)

        # train model
        exp = TrainManager(dataloader, args)
        # load saved model

        folder = args['directory'] + 'trained_models/' + \
            f'{pre_trained_epochs}epochs/'
        exp.load_model(folder)
        print(f'Pre-trained model on {pre_trained_epochs} epochs loaded.')

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
