from itertools import combinations_with_replacement, combinations

import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump
from pymatgen.analysis.local_env import LocalStructOrderParas

from load_structure import load_train_structure, load_test_structure
from load_structure import get_anion_polyhedron, get_cation_polyhedron


def calc_oxygen_bop(structure, lidx=2, bins=10):
    list_cutoff = np.linspace(3., 10., num=bins)
    ops = ['q{}'.format(lidx), ]
    list_oxygen = [i for i, specie in enumerate(structure.species) if specie.name == 'O']

    bop = np.array([np.mean([LocalStructOrderParas(types=ops, cutoff=cutoff).get_order_parameters(structure, i)[0] for i in list_oxygen]) for cutoff in list_cutoff])

    return bop


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    bins = 10

    for l in [2, 4, 6]:
        bob_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_oxygen_bop)(structure, l, bins) for structure in structures_train)
        bob_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_oxygen_bop)(structure, l, bins) for structure in structures_test)

        bob_train = pd.DataFrame(bob_train_tmp)
        bob_test = pd.DataFrame(bob_test_tmp)

        dump(bob_train, '../data/q{}_{}_df.train'.format(l, bins))
        dump(bob_test, '../data/q{}_{}_df.test'.format(l, bins))
