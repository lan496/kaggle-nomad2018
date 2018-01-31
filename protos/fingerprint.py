from itertools import combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def gaussian(x, sigma):
    dev = -0.5 * ((x / sigma) ** 2)
    return np.exp(dev)


def fingerprint(structure, ele_A, ele_B, cutoff=5, delta=0.1, sigma=0.2):
    arr_R = np.arange(1, cutoff, delta, dtype=float)
    fp = np.zeros_like(arr_R)

    site_A_idx = [i for i, site in enumerate(structure) if site.specie.name == ele_A]

    list_species = ['Al', 'Ga', 'In', 'O']
    list_species.remove(ele_B)
    structure_B = structure.copy()
    structure_B.remove_species(list_species)

    N_A = len(site_A_idx)
    N_B = structure_B.num_sites
    if N_A == 0 or N_B == 0:
        return fp

    for i_A in site_A_idx:
        nn = structure_B.get_sites_in_sphere(structure[i_A].coords, cutoff + 3 * sigma)
        dist = np.array([e[1] for e in nn], dtype=float)

        fp += np.sum(gaussian(arr_R[:, np.newaxis] - dist[np.newaxis, :], sigma) / (dist[np.newaxis, :] ** 2), axis=1)

    factor = 4. * np.sqrt(2) * (np.pi ** 1.5) * N_A * N_B * sigma / structure.volume
    fp = fp / factor - 1
    return fp


def calc_fingerprint(structure, cutoff=5., delta=0.1, sigma=0.2):
    cations = ['Al', 'Ga', 'In']
    dsc = np.concatenate([fingerprint(structure, cation, 'O') for cation in cations])
    return dsc


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    cutoff = 5.
    delta = 0.5
    sigma = 0.1

    fp_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_fingerprint)(structure, cutoff, delta, sigma) for structure in structures_train)
    fp_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_fingerprint)(structure, cutoff, delta, sigma) for structure in structures_test)

    columns = ['fp_{}'.format(i) for i in range(fp_train_tmp[0].shape[0])]

    fp_train = pd.DataFrame(fp_train_tmp, columns=columns)
    fp_test = pd.DataFrame(fp_test_tmp, columns=columns)

    dump(fp_train, '../data/fp_cutoff={}_delta={}_sigma={}_df.train'.format(cutoff, delta, sigma))
    dump(fp_test, '../data/fp_cutoff={}_delta={}_sigma={}_df.test'.format(cutoff, delta, sigma))
