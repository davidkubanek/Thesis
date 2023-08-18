

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle

# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_datalist(matrix_df, args, graph_fp=True, grover_fp=None):
    '''
    Convert matrix dataframe to a data_list with pytorch geometric graph data, fingerprints and labels.
    Inputs:
        matrix_df: dataframe of SMILES, assays and bioactivity labels
        args: arguments
        graph_fp: if True, includes graph embedding fingerprints into data_list
        grover_fp: if True, includes GROVER graph transformer embedding fingerprints into data_list
    Outputs:
        data_list: list of data objects
    '''
    # only use subset of data (assays and data points)
    assay_list = args['assay_list']
    num_assays = args['num_assays']
    assay_start = args['assay_start']
    num_data_points = args['num_data_points']

    # get binary target labels
    y = matrix_df[assay_list[assay_start:assay_start+num_assays]
                  ].values[:num_data_points]

    # get SMILES strings
    data = matrix_df['SMILES'].values[:num_data_points]

    if graph_fp is True:  # add graph fingerprint
        GraphDataset = GraphDatasetClass()
        # create pytorch geometric graph data list
        data_list = GraphDataset.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(
            data, y)
    else:  # create simple data_list without graph fingerprint
        data_list = []
        for label in y:
            # construct Pytorch Geometric data object and append to data list
            data_list.append(Data(y=label.reshape(1, -1)))

    # add fingerprint data to each graph
    for i, smile in tqdm(enumerate(data), desc='Adding fingerprints...', total=len(data)):
        fp = convert_smile_to_fp_bit_string(smile)
        data_list[i].fp = fp

    # add grover fingerprint to each graph
    if grover_fp is not None:
        for i, gfp in tqdm(enumerate(grover_fp['fps'][:args['num_data_points']]), desc='Adding grover embedding...', total=len(data)):
            data_list[i].grover_fp = torch.tensor(gfp)

    print(f'Example of a graph data object: {data_list[0]}')

    return data_list


def prepare_splits(data_list, args):

    data_list = data_list[:args['num_data_points']]

    data_splits = {}
    # split into train and test
    train_dataset = [d.to(args['device'])
                     for d in data_list[:int(len(data_list)*0.8)]]
    data_splits['test'] = [d.to(args['device'])
                           for d in data_list[int(len(data_list)*0.8):]]

    # split into train and validation
    data_splits['val'] = train_dataset[:int(len(train_dataset)*0.25)]
    data_splits['train'] = train_dataset[int(len(train_dataset)*0.25):]

    print(f'Number of training graphs:', len(data_splits['train']))
    print(f'Number of validation graphs:', len(data_splits['val']))
    print(f'Number of test graphs:', len(data_splits['test']))
    print(f'Example of a graph data object: {data_list[0]}')

    return data_splits


def prepare_dataloader(data_splits, args):
    '''
    Get dataloader dictionary from data_list with desired batch_size
    '''
    # create data loaders
    dataloader = {}
    dataloader['train'] = DataLoader(
        data_splits['train'], batch_size=args['batch_size'], shuffle=True)
    dataloader['val'] = DataLoader(
        data_splits['val'], batch_size=args['batch_size'], shuffle=False)
    dataloader['test'] = DataLoader(
        data_splits['test'], batch_size=args['batch_size'], shuffle=False)

    return dataloader


def analyze_dataset(dataset, args):
    '''
    Analyze the distribution of positive classes in the dataset
    '''
    positive = []
    for i in range(len(dataset)):
        positive.append(dataset[i].y[0].sum().item())

    num_assays = args['num_assays']
    # make histogram of the number of positive
    plt.figure(figsize=(7, 4))
    # define bins
    bins = np.linspace(0, num_assays, num_assays+1)-0.5
    plt.hist(positive, bins=bins, alpha=0.5, label='train')
    num_assays = args['num_assays']
    plt.xlabel(f'# of positive hits in target vector (out of {num_assays})')
    plt.ylabel('Number of data points')
    plt.title('Histogram of positive class distribution')
    plt.show()

    # for i in range(num_assays+1):
    #     print(f'Number of data points with {i} positive targets: ', (np.array(positive) == i).sum(), f'({(np.array(positive) == i).sum()/len(positive)*100:.2f}%)')


def data_explore(dataloader):
    '''
    Explore the data
    '''
    # check proportion of positive and negative samples across each assay
    pos = torch.zeros(args['num_assays'])
    for data in dataloader:  # Iterate in batches over the training dataset
        # print('inputs:')
        # print(' x:', data.x.shape, '| y:',data.y.shape, '| fp:',data.fp.shape, '| grover:', data.grover_fp.shape)
        pos += data.y.sum(axis=0)
        #  print(data.y.sum(axis=0))
    print('# positive samples:', pos)
    print(torch.round((pos/len(dataloader.dataset)*100), decimals=2), '% are positive')


def load_datalist(directory, load_path):
    '''
    Load the data_list from pickle file
    and load the assay groups and assay order.
    directory: root directory of data
    load_path: path to pickle file
    '''
    # Load the data_list using pickle
    with open(load_path, 'rb') as f:
        data_list = pickle.load(f)

    # load the assay groups
    assay_groups = {}
    with open(directory + 'info/cell_based_high_hr.txt', 'r') as file:
        lines = file.read().splitlines()
    assay_groups['cell_based_high_hr'] = list(map(str, lines))
    with open(directory + 'info/cell_based_med_hr.txt', 'r') as file:
        lines = file.read().splitlines()
    assay_groups['cell_based_med_hr'] = list(map(str, lines))
    with open(directory + 'info/cell_based_low_hr.txt', 'r') as file:
        lines = file.read().splitlines()
    assay_groups['cell_based_low_hr'] = list(map(str, lines))
    with open(directory + 'info/non_cell_based_high_hr.txt', 'r') as file:
        lines = file.read().splitlines()
    assay_groups['non_cell_based_high_hr'] = list(map(str, lines))
    with open(directory + 'info/non_cell_based_med_hr.txt', 'r') as file:
        lines = file.read().splitlines()
    assay_groups['non_cell_based_med_hr'] = list(map(str, lines))
    with open(directory + 'info/non_cell_based_low_hr.txt', 'r') as file:
        lines = file.read().splitlines()
    assay_groups['non_cell_based_low_hr'] = list(map(str, lines))
    # load assay order
    with open(directory + 'info/assay_order.txt', 'r') as f:
        assay_order = [line.strip() for line in f.readlines()]

    return data_list, assay_groups, assay_order
