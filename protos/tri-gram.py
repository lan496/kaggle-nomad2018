from itertools import combinations_with_replacement, combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def update_M_O_M_cluster(structure, bags):
    poly = get_cation_polyhedron(structure, 'O')
    for i, cations in enumerate(poly):
        for j, k in combinations(cations, 2):
            cation_j = structure[j].specie.name
            cation_k = structure[k].specie.name
            if cation_j < cation_k:
                bags["{}_O_{}".format(cation_j, cation_k)] += 1
            else:
                bags["{}_O_{}".format(cation_k, cation_j)] += 1


def update_O_M_O_cluster(structure, bags):
    for ele_sym in ['Al', 'Ga', 'In']:
        poly = get_anion_polyhedron(structure, ele_sym)
        bags["O_{}_O".format(ele_sym)] = np.sum([len(anions) * (len(anions) - 1) // 2 for anions in poly])


def calc_tri_gram(structure):
    list_cation = ['Al', 'Ga', 'In']
    keys = ['{}_O_{}'.format(e1, e2) for e1, e2 in combinations_with_replacement(list_cation, 2)] + ['O_{}_O'.format(e) for e in list_cation]
    bags = {key: 0 for key in keys}
    update_M_O_M_cluster(structure, bags)
    update_O_M_O_cluster(structure, bags)

    return bags


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    tri_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_tri_gram)(structure) for structure in structures_train)
    tri_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_tri_gram)(structure) for structure in structures_test)

    tri_train = pd.DataFrame(tri_train_tmp)
    tri_test = pd.DataFrame(tri_test_tmp)

    dump(tri_train, '../data/tri_df.train')
    dump(tri_test, '../data/tri_df.test')
