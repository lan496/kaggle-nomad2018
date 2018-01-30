from itertools import combinations_with_replacement, combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def calc_bi_gram(structure):
    keys = ['Al_O', 'Ga_O', 'In_O']
    bags = {key: 0 for key in keys}

    poly_cations = get_cation_polyhedron(structure, 'O')

    for cations in poly_cations:
        for j in cations:
            cation_j = structure[j].specie.name
            bags['{}_O'.format(cation_j)] += 1

    return bags


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    bi_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_bi_gram)(structure) for structure in structures_train)
    bi_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_bi_gram)(structure) for structure in structures_test)

    bi_train = pd.DataFrame(bi_train_tmp)
    bi_test = pd.DataFrame(bi_test_tmp)

    dump(bi_train, '../data/bi_df.train')
    dump(bi_test, '../data/bi_df.test')
