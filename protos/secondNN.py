import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from pymatgen.analysis.local_env import JMolNN
from joblib import dump, Parallel, delayed


def get_cation_polyhedron(structure, anion_sym='O'):
    nn = JMolNN()
    coo = [[e['site_index'] for e in nn.get_nn_info(structure, i) if e['site'].specie.name != 'O']
           for i, spc in enumerate(structure.species) if spc.name == anion_sym]
    return coo


def oxygen_second_nearest_neighbors(structure, size=20):
    list_cation_polyhedron = get_cation_polyhedron(structure)
    list_oxygen = [i for i, specie in enumerate(structure.species) if specie.name == 'O']

    nn = JMolNN()

    ret = np.zeros(size)
    for i, oxygen in enumerate(list_oxygen):
        oxy_tmp = []
        for cation in list_cation_polyhedron[i]:
            oxy_tmp.extend([e['site_index'] for e in nn.get_nn_info(structure, cation) if e['site'].specie.name == 'O'])

        ret[len(set(oxy_tmp))] += 1

    ret /= np.sum(ret)

    return ret


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

    size = 20

    secondNN_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(oxygen_second_nearest_neighbors)(structure, size) for structure in structures_train)
    secondNN_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(oxygen_second_nearest_neighbors)(structure, size) for structure in structures_test)

    secondNN_train = pd.DataFrame(np.array(secondNN_train_tmp),
                                  columns=['num_secondNN_{}'.format(i) for i in range(size)])
    secondNN_test = pd.DataFrame(np.array(secondNN_test_tmp),
                                 columns=['num_secondNN_{}'.format(i) for i in range(size)])

    dump(secondNN_train, '../data/secondNN_df.train')
    dump(secondNN_test, '../data/secondNN_df.test')
