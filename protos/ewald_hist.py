import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed, dump
from pymatgen.analysis.ewald import EwaldSummation

from load_structure import load_train_structure, load_test_structure


def get_ewald_matrix_hist(emat, bins=40):
    natoms = emat.shape[0]
    hist, _ = np.histogram(emat[np.triu_indices(emat.shape[0])], range=(-40, 40), bins=bins)
    hist = hist / (natoms * (natoms + 1) / 2)
    return hist


if __name__ == '__main__':
    emat_train = joblib.load('../data/ewald_matrix_list.train')
    emat_test = joblib.load('../data/ewald_matrix_list.test')

    bins = 30

    ewald_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(get_ewald_matrix_hist)(emat, bins) for emat in emat_train)
    ewald_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(get_ewald_matrix_hist)(emat, bins) for emat in emat_test)

    columns = ['energy_{}'.format(i) for i in range(bins)]

    ehist_train = pd.DataFrame(ewald_train_tmp, columns=columns)
    ehist_test = pd.DataFrame(ewald_test_tmp, columns=columns)

    dump(ehist_train, '../data/ehist_df_{}.train'.format(bins))
    dump(ehist_test, '../data/ehist_df_{}.test'.format(bins))
