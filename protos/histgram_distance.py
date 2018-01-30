from itertools import combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def get_nearest_neighbor_distances(structure):
    poly_cations = get_cation_polyhedron(structure, 'O')
    oxygen_idx = [i for i, specie in enumerate(structure.species) if specie.name == 'O']

    dists = []
    for oidx, cations in enumerate(poly_cations):
        for j in cations:
            i = oxygen_idx[oidx]
            dists.append(structure.get_distance(i, j))

    return dists


def calc_distance_histgram(structure, bins):
    dists = get_nearest_neighbor_distances(structure)

    dists_hist, _ = np.histogram(dists, bins=bins, range=(1., 3.))

    return dists_hist


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    bins = 20

    hdist_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_distance_histgram)(structure, bins) for structure in structures_train)
    hdist_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_distance_histgram)(structure, bins) for structure in structures_test)

    columns = ['distance_{}'.format(i) for i in range(bins)]

    hdist_train = pd.DataFrame(hdist_train_tmp, columns=columns)
    hdist_test = pd.DataFrame(hdist_test_tmp, columns=columns)

    dump(hdist_train, '../data/hdist_df_{}.train'.format(bins))
    dump(hdist_test, '../data/hdist_df_{}.test'.format(bins))
