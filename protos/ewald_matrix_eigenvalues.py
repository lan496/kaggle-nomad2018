import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from Ewald import EwaldSummationMatrix


def ewald_descriptor(structure, acc_factor=12., Nmax=80):
    ewald = EwaldSummationMatrix(structure, acc_factor)
    emat = ewald.get_ewald_summation_matrix()

    eigvals = np.linalg.eigvalsh(emat)
    eigvals = eigvals[np.argsort(-np.abs(eigvals))]

    numsites = eigvals.shape[0]
    if numsites > Nmax:
        raise Exception('Nmax should be greater than dimention of eigvals.')
    elif numsites < Nmax:
        ret = np.concatenate([eigvals, np.zeros(Nmax - numsites)], axis=0)
    else:
        ret = eigvals

    return ret


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    acc_factor = 12.
    Nmax = 80

    ewald_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(ewald_descriptor)(structure, acc_factor, Nmax) for structure in structures_train)
    ewald_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(ewald_descriptor)(structure, acc_factor, Nmax) for structure in structures_test)

    columns = ['em_{}'.format(i) for i in range(Nmax)]

    ewald_train = pd.DataFrame(ewald_train_tmp, columns=columns)
    ewald_test = pd.DataFrame(ewald_test_tmp, columns=columns)

    dump(ewald_train, '../data/ewald_df.train')
    dump(ewald_test, '../data/ewald_df.test')
