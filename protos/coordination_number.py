import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from pymatgen.analysis.local_env import JMolNN
from joblib import dump, Parallel, delayed


def get_cation_coordination_number_portion(structure, cation_sym):
    nn = JMolNN()
    coo = [len([e for e in nn.get_nn_info(structure, i) if e['site'].specie.name == 'O'])
           for i, spc in enumerate(structure.species) if spc.name == cation_sym]
    portion = np.zeros(10)
    if coo:
        for e in coo:
            portion[e - 3] += 1
        portion /= len(coo)
    return portion


def get_anion_coordination_number_portion(structure, anion_sym):
    nn = JMolNN()
    coo = [len([e for e in nn.get_nn_info(structure, i) if e['site'].specie.name != 'O'])
           for i, spc in enumerate(structure.species) if spc.name == anion_sym]
    portion = np.zeros(10)
    if coo:
        for e in coo:
            portion[e - 3] += 1
        portion /= len(coo)

    return portion


def get_coordination_number_portion(structure):
    cation_portion = [get_cation_coordination_number_portion(structure, sym)
                      for sym in ['Al', 'Ga', 'In']]
    anion_portion = get_anion_coordination_number_portion(structure, 'O')

    ret = np.concatenate([*cation_portion, anion_portion], axis=0)
    return ret


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

    coo_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(get_coordination_number_portion)(structure) for structure in structures_train)
    coo_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(get_coordination_number_portion)(structure) for structure in structures_test)

    coo_train = np.array(coo_train_tmp)
    coo_test = np.array(coo_test_tmp)

    dump(coo_train, '../data/coo_mat.train')
    dump(coo_test, '../data/coo_mat.test')
