import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from Ewald import EwaldSummationMatrix


def calc_total_coulomb_potential(structure, acc_factor=12.):
    ewald = EwaldSummationMatrix(structure, acc_factor)
    emat = ewald.get_ewald_summation_matrix()

    total = np.sum(emat[np.triu_indices(emat.shape[0])])

    return total


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    acc_factor = 12.

    tcp_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_total_coulomb_potential)(structure, acc_factor) for structure in structures_train)
    tcp_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_total_coulomb_potential)(structure, acc_factor) for structure in structures_test)

    tcp_train = np.array(tcp_train_tmp)
    tcp_test = np.array(tcp_test_tmp)

    dump(tcp_train, '../data/total_coulomb_potential.train')
    dump(tcp_test, '../data/total_coulomb_potential.test')
