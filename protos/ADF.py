from functools import partial
from itertools import combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def gaussian(x, sigma):
    dev = -0.5 * ((x / sigma) ** 2)
    return np.exp(dev)


def cutoff_function(R, R_cutoff):
    fc = 0.5 * (1 + np.cos(np.pi * R / R_cutoff))
    fc[R > R_cutoff] = 0
    return fc


def angular_distribution(structure, sym, sigma, zeta, R_cutoff):
    list_Gi = []

    nn = structure.get_all_neighbors(R_cutoff, include_index=True)
    list_nn_idx = [[idx for _, _, idx in list_nn if idx != i] for i, list_nn in enumerate(nn)]

    dmat = structure.distance_matrix
    factor = gaussian(dmat, sigma) * cutoff_function(dmat, R_cutoff)

    for i, site in enumerate(structure):
        if site.specie.name != sym:
            continue

        Gi = 0.0

        for j, k in combinations(list_nn_idx[i], 2):
            theta = structure.get_angle(j, i, k)
            Gi += ((1 + np.cos(theta)) ** zeta) * factor[i, j] * factor[j, k] * factor[k, i]

        Gi /= 2 ** (zeta - 1)
        list_Gi.append(Gi)

    if list_Gi:
        return np.mean(list_Gi)
    else:
        return 0


def calc_ADF(structure):
    list_elements = ['Al', 'Ga', 'In', 'O']
    list_sigma = [1.0, 1.5, 2.0]
    zeta = 1
    R_cutoff = 6.0

    columns = ['ADF_{}_{}'.format(sym, sigma) for sym in list_elements for sigma in list_sigma]
    vals = [angular_distribution(structure, sym, sigma, zeta, R_cutoff) for sym in list_elements for sigma in list_sigma]
    dsc = {col: val for col, val in zip(columns, vals)}
    return dsc


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    ADF_train_tmp = Parallel(n_jobs=-1, verbose=10)(delayed(calc_ADF)(structure) for structure in structures_train)
    ADF_test_tmp = Parallel(n_jobs=-1, verbose=10)(delayed(calc_ADF)(structure) for structure in structures_test)

    ADF_train = pd.DataFrame(ADF_train_tmp)
    ADF_test = pd.DataFrame(ADF_test_tmp)

    dump(ADF_train, '../data/ADF_eta_df.train')
    dump(ADF_test, '../data/ADF_eta_df.test')
