from itertools import combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def get_dihedral_angles(structure):
    cations_sym = ['Al', 'Ga', 'In']

    poly_cations = get_cation_polyhedron(structure, 'O')
    poly_anions = {ele_sym: get_cation_polyhedron(structure, ele_sym) for ele_sym in cations_sym}

    oxygen_idx = [i for i, specie in enumerate(structure.species) if specie.name == 'O']
    cations_idx = {ele_sym: [i for i, specie in enumerate(structure.species) if specie.name == ele_sym] for ele_sym in cations_sym}

    dihedral = []

    for oidx, cations in enumerate(poly_cations):
        j = oxygen_idx[oidx]
        for i, k in combinations(cations, 2):
            ele_sym_k = structure[k].specie.name
            cidx = cations_idx[ele_sym_k].index(k)
            for ll in poly_anions[ele_sym_k][cidx]:
                if ll == i:
                    continue
                dihedral.append(structure.get_dihedral(i, j, k, ll))

    return dihedral


def calc_angle_histgram(structure, bins):
    dihedral_angles = np.radians(get_dihedral_angles(structure))

    dihedral_angles_hist, _ = np.histogram(dihedral_angles, bins=bins, range=(-np.pi, np.pi))

    return dihedral_angles_hist


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    bins = 20

    had_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_angle_histgram)(structure, bins) for structure in structures_train)
    had_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_angle_histgram)(structure, bins) for structure in structures_test)

    columns = ['dihedral_{}'.format(i) for i in range(bins)]

    had_train = pd.DataFrame(had_train_tmp, columns=columns)
    had_test = pd.DataFrame(had_test_tmp, columns=columns)

    dump(had_train, '../data/had_df_{}.train'.format(bins))
    dump(had_test, '../data/had_df_{}.test'.format(bins))
