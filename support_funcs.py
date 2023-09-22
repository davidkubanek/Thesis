
import numpy as np
import itertools
import random


# find indeces of assays in assay_list in assay_order
# return list of indeces


def find_assay_indeces(assay_list, assay_order):
    indeces = []
    for assay in assay_list:
        indeces.append(assay_order.index(assay))
    return indeces

def random_grid_search(hyperparams_dict, samples):
    # Generate all combinations
    all_combinations = list(itertools.product(*(hyperparams_dict[key]['values'] for key in hyperparams_dict)))

    # Randomly sample distinct combinations
    random_combinations = random.sample(all_combinations, samples)

    # Convert the tuples back to dictionary format for easier use in hyperparameter optimization
    param_names = list(hyperparams_dict.keys())
    random_combinations_dicts = [dict(zip(param_names, combo)) for combo in random_combinations]

    return random_combinations_dicts