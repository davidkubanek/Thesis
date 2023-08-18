
import numpy as np

# find indeces of assays in assay_list in assay_order
# return list of indeces


def find_assay_indeces(assay_list, assay_order):
    indeces = []
    for assay in assay_list:
        indeces.append(assay_order.index(assay))
    return indeces
