from itertools import combinations_with_replacement, combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def calc_tetra_gram(structure):
    list_cation = ['Al', 'Ga', 'In']
    keys = ['{}_O_{}_O'.format(e1, e2) for e1 in list_cation for e2 in list_cation]
    bags = {key: 0 for key in keys}

    poly_cations = get_cation_polyhedron(structure, 'O')
    poly_anions = {ele_sym: get_cation_polyhedron(structure, ele_sym) for ele_sym in list_cation}

    oxygen_idx = [i for i, specie in enumerate(structure.species) if specie.name == 'O']
    cations_idx = {ele_sym: [i for i, specie in enumerate(structure.species) if specie.name == ele_sym] for ele_sym in list_cation}

    for oidx, cations in enumerate(poly_cations):
        j = oxygen_idx[oidx]
        for i, k in combinations(cations, 2):
            ele_sym_i = structure[i].specie.name
            ele_sym_k = structure[k].specie.name
            cidx = cations_idx[ele_sym_k].index(k)
            for ll in poly_anions[ele_sym_k][cidx]:
                if ll == i:
                    continue
                bags['{}_O_{}_O'.format(ele_sym_i, ele_sym_k)] += 1

    return bags


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    tetra_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_tetra_gram)(structure) for structure in structures_train)
    tetra_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_tetra_gram)(structure) for structure in structures_test)

    tetra_train = pd.DataFrame(tetra_train_tmp)
    tetra_test = pd.DataFrame(tetra_test_tmp)

    dump(tetra_train, '../data/tetra_df.train')
    dump(tetra_test, '../data/tetra_df.test')
