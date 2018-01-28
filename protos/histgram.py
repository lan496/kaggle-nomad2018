from itertools import combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def get_M_O_M_angles(structure):
    angles = []

    poly = get_cation_polyhedron(structure, 'O')
    oxygen_idx = [i for i, specie in enumerate(structure.species) if specie.name == 'O']
    for i, cations in enumerate(poly):
        for j, k in combinations(cations, 2):
            angles.append(structure.get_angle(j, oxygen_idx[i], k))

    return angles


def get_O_M_O_angles(structure):
    angles = []

    for ele_sym in ['Al', 'Ga', 'In']:
        poly = get_anion_polyhedron(structure, ele_sym)
        cation_idx = [i for i, specie in enumerate(structure.species) if specie.name == ele_sym]
        for i, anions in enumerate(poly):
            for j, k in combinations(anions, 2):
                angles.append(structure.get_angle(j, cation_idx[i], k))

    return angles


def calc_angle_histgram(structure, bins):
    M_O_M_angles = np.radians(get_M_O_M_angles(structure))
    O_M_O_angles = np.radians(get_O_M_O_angles(structure))

    M_O_M_angles_hist, _ = np.histogram(M_O_M_angles, bins=bins, range=(0, np.pi))
    O_M_O_angles_hist, _ = np.histogram(O_M_O_angles, bins=bins, range=(0, np.pi))

    dsc = np.concatenate([M_O_M_angles_hist, O_M_O_angles_hist])
    return dsc


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    bins = 30

    ha_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_angle_histgram)(structure, bins) for structure in structures_train)
    ha_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_angle_histgram)(structure, bins) for structure in structures_test)

    columns = ['M_O_M_{}'.format(i) for i in range(bins)] + ['O_M_O_{}'.format(i) for i in range(bins)]

    ha_train = pd.DataFrame(ha_train_tmp, columns=columns)
    ha_test = pd.DataFrame(ha_test_tmp, columns=columns)

    dump(ha_train, '../data/ha_df_{}.train'.format(bins))
    dump(ha_test, '../data/ha_df_{}.test'.format(bins))
