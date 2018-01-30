import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump
from pymatgen.analysis.ewald import EwaldSummation

from load_structure import load_train_structure, load_test_structure


def get_ewald_matrix(structure):
    structure_ox = structure.copy()
    structure_ox.add_oxidation_state_by_element({'Al': 3, 'Ga': 3, 'In': 3, 'O': -2})
    es = EwaldSummation(structure_ox)
    emat = es.total_energy_matrix
    total = es.total_energy

    return emat, total


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    acc_factor = 12.
    Nmax = 80

    ewald_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(get_ewald_matrix)(structure) for structure in structures_train)
    ewald_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(get_ewald_matrix)(structure) for structure in structures_test)

    columns = ['em_{}'.format(i) for i in range(Nmax)]

    total_train = pd.DataFrame([e[1] for e in ewald_train_tmp], columns=['total_energy'])
    total_test = pd.DataFrame([e[1] for e in ewald_test_tmp], columns=['total_energy'])

    ewald_train = [e[0] for e in ewald_train_tmp]
    ewald_test = [e[0] for e in ewald_test_tmp]

    dump(total_train, '../data/total_df.train')
    dump(total_test, '../data/total_df.test')

    dump(ewald_train, '../data/ewald_matrix_list.train')
    dump(ewald_test, '../data/ewald_matrix_list.test')
