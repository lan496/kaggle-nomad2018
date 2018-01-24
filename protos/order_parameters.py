import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from pymatgen.analysis.local_env import LocalStructOrderParas
from joblib import dump, Parallel, delayed


def calc_order_parameters_descriptor(structure, list_types=None, cutoff=10.):
    local = LocalStructOrderParas(types=list_types, cutoff=cutoff)

    mat_order_params = np.array([local.get_order_parameters(structure, i) for i in range(structure.num_sites)])

    dsc = np.mean(mat_order_params, axis=0)
    return dsc


def _calc_each_order_parameters_descriptor(structure, ele_sym, list_types, cutoff=10):
    local = LocalStructOrderParas(types=list_types, cutoff=cutoff)

    struct = structure.copy()
    list_elements = ['Al', 'Ga', 'In', 'O']
    list_elements.remove(ele_sym)
    struct.remove_species(list_elements)

    mat_order_params = np.array([local.get_order_parameters(struct, i) for i in range(struct.num_sites)])

    if struct.num_sites <= 1:
        dsc = np.zeros(len(list_types))
    else:
        dsc = np.mean(mat_order_params, axis=0)

    return dsc


def calc_each_order_parameters_descriptor(structure, list_types, cutoff=10.):
    list_elements = ['Al', 'Ga', 'In', 'O']
    dsc = np.concatenate([_calc_each_order_parameters_descriptor(structure, ele_sym, list_types, cutoff) for ele_sym in list_elements])
    return dsc


def load_structure(row, file_dir):
    filename = os.path.join(file_dir, str(int(row['id'])), 'geometry.json')
    with open(filename, 'r') as f:
        d = json.load(f)
        structure = mg.core.Structure.from_dict(d)

    return structure


def load_train_structure(row):
    return load_structure(row, '../data/train')


def load_test_structure(row):
    return load_structure(row, '../data/test')


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    list_elements = ['Al', 'Ga', 'In', 'O']

    list_types = ['q2', 'q4', 'q6']
    cutoff = 10.

    """
    for structure in structures_train:
        calc_each_order_parameters_descriptor(structure, list_types, cutoff)
    """

    ops_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_each_order_parameters_descriptor)(structure, list_types, cutoff) for structure in structures_train)
    ops_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_each_order_parameters_descriptor)(structure, list_types, cutoff) for structure in structures_test)

    columns = ['{}_{}'.format(ele, types) for ele in list_elements for types in list_types]

    ops_train = pd.DataFrame(ops_train_tmp, columns=columns)
    ops_test = pd.DataFrame(ops_test_tmp, columns=columns)

    dump(ops_train, '../data/ops_each_df.train')
    dump(ops_test, '../data/ops_each_df.test')
