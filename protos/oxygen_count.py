import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from joblib import dump, Parallel, delayed


def calc_oxygen_descriptor(structure):
    oxygen_ave = len([1 for species in structure.species if species.name == 'O']) / structure.volume
    return oxygen_ave


def load_structure(row, file_dir):
    # import pdb; pdb.set_trace()
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

    oxygen_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_oxygen_descriptor)(structure) for structure in structures_train)
    oxygen_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_oxygen_descriptor)(structure) for structure in structures_test)

    oxygen_train = np.array(oxygen_train_tmp)
    oxygen_test = np.array(oxygen_test_tmp)

    dump(oxygen_train, '../data/oxygen_arr.train')
    dump(oxygen_test, '../data/oxygen_arr.test')
