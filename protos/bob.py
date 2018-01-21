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


def get_bag_of_bonds(structure):
    list_cation_polyhedron = get_cation_polyhedron(structure, 'O')
    bag_of_bonds = {'Al': 0, 'Ga': 0, 'In': 0}
    for cation_polyhedron in list_cation_polyhedron:
        for idx in cation_polyhedron:
            bag_of_bonds[structure[idx].specie.name] += 1
    return bag_of_bonds


def calc_bag_of_bonds_descriptor(structure):
    bob = get_bag_of_bonds(structure)
    dsc = np.array([bob['Al'], bob['Ga'], bob['In']], dtype=float)
    dsc /= np.sum(dsc)
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

    bob_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_bag_of_bonds_descriptor)(structure) for structure in structures_train)
    bob_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_bag_of_bonds_descriptor)(structure) for structure in structures_test)

    bob_train = pd.DataFrame(np.array(bob_train_tmp),
                             columns=['bond_Al-O', 'bond_Ga-O', 'bond_In-O'])
    bob_test = pd.DataFrame(np.array(bob_test_tmp),
                            columns=['bond_Al-O', 'bond_Ga-O', 'bond_In-O'])

    dump(bob_train, '../data/bob_df.train')
    dump(bob_test, '../data/bob_df.test')
