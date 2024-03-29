from functools import partial

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def gaussian(x, sigma):
    dev = -0.5 * ((x / sigma) ** 2)
    return np.exp(dev)


def cutoff_function(R, R_cutoff):
    fc = 0.
    if R < R_cutoff:
        fc = 0.5 * (1 + np.cos(np.pi * R / R_cutoff))
    return fc


def GRDF(structure, ele_sym, sigma=0.2, R_center=3., R_cutoff=10.):
    func_cutoff = np.vectorize(partial(cutoff_function, R_cutoff=R_cutoff))

    list_gi = []

    for i, site in enumerate(structure):
        if site.specie.name != ele_sym:
            continue
        nn = structure.get_neighbors(site, R_cutoff)
        dist = np.array([e[1] for e in nn], dtype=float)
        cutoff_factor = func_cutoff(dist)

        gi_tmp = np.sum(gaussian(dist - R_center, sigma) * cutoff_factor)
        list_gi.append(gi_tmp)

    if list_gi:
        return np.mean(list_gi)
    else:
        return 0


def calc_GRDF(structure):
    list_elements = ['Al', 'Ga', 'In', 'O']
    list_sigma = [1.0, 1.5, 2.0]
    R_center = 0.0
    R_cutoff = 6.0

    columns = ['{}_{}'.format(sym, sigma) for sym in list_elements for sigma in list_sigma]
    vals = [GRDF(structure, sym, sigma, R_center, R_cutoff) for sym in list_elements for sigma in list_sigma]
    dsc = {col: val for col, val in zip(columns, vals)}
    return dsc


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    grdf_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_GRDF)(structure) for structure in structures_train)
    grdf_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_GRDF)(structure) for structure in structures_test)

    grdf_train = pd.DataFrame(grdf_train_tmp)
    grdf_test = pd.DataFrame(grdf_test_tmp)

    dump(grdf_train, '../data/grdf_eta2_df.train')
    dump(grdf_test, '../data/grdf_eta2_df.test')
