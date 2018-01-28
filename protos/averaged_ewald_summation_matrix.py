from itertools import combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from Ewald import EwaldSummationMatrix


def calc_averaged_ewald_summation_matrix(structure, acc_factor):
    ewald = EwaldSummationMatrix(structure, acc_factor)
    emat = ewald.get_ewald_summation_matrix()

    indices = [[i for i, specie in enumerate(structure.species)
                if specie.name == ele_sym] for ele_sym in ['Al', 'Ga', 'In', 'O']]
    n_ele = len(indices)
    mat_grouped = np.zeros((n_ele, n_ele))

    for i in range(n_ele):
        if indices[i]:
            mat_grouped[i, i] = np.average([emat[j, j] for j in indices[i]])

    for i, j in combinations(range(n_ele), 2):
        if indices[i] and indices[j]:
            mat_grouped[i, j] = np.average([emat[ii, jj] for ii in indices[i] for jj in indices[j]])
            mat_grouped[j, i] = mat_grouped[i, j]

    return mat_grouped.reshape(-1)


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    acc_factor = 12.

    emat_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_averaged_ewald_summation_matrix)(structure, acc_factor) for structure in structures_train)
    emat_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_averaged_ewald_summation_matrix)(structure, acc_factor) for structure in structures_test)

    emat_train = pd.DataFrame(emat_train_tmp, columns=['emat_{}'.format(i) for i in range(16)])
    emat_test = pd.DataFrame(emat_test_tmp, columns=['emat_{}'.format(i) for i in range(16)])

    dump(emat_train, '../data/averaged_ewald_summation_matrix.train')
    dump(emat_test, '../data/averaged_ewald_summation_matrix.test')
