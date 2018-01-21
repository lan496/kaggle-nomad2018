import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from pymatgen.analysis.local_env import JMolNN
from joblib import dump, Parallel, delayed


def get_anion_polyhedron(structure, cation_sym):
    nn = JMolNN()
    coo = [[e['site_index'] for e in nn.get_nn_info(structure, i) if e['site'].specie.name == 'O']
           for i, spc in enumerate(structure.species) if spc.name == cation_sym]
    return coo


def get_sharing_polyhedron_dist(list_polyhedron):
    n_ph = len(list_polyhedron)
    hist_shared = np.zeros(7)

    for i in np.arange(n_ph):
        for j in np.arange(i):
            shared_node = set(list_polyhedron[i]) & set(list_polyhedron[j])
            hist_shared[len(shared_node)] += 1
            # if shared_node:
            #     hist_shared[len(shared_node) - 1] += 1
    hist_shared /= n_ph * (n_ph - 1) / 2
    return hist_shared


def calc_pauling_third_descriptor(structure):
    polyhedron = [get_anion_polyhedron(structure, cation) for cation in ['Al', 'Ga', 'In']]
    ret = get_sharing_polyhedron_dist([e for list_polyhedron in polyhedron for e in list_polyhedron])
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

    pauling3_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_pauling_third_descriptor)(structure) for structure in structures_train)
    pauling3_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_pauling_third_descriptor)(structure) for structure in structures_test)

    pauling3_train = np.array(pauling3_train_tmp)
    pauling3_test = np.array(pauling3_test_tmp)

    dump(pauling3_train, '../data/pauling3_mat.train')
    dump(pauling3_test, '../data/pauling3_mat.test')
