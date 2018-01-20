import os
import json

import pandas as pd
import numpy as np
from pymatgen import Lattice, Structure


TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'


# https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
def get_xyz_data(filename):
    coo_data = []
    ele_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                coo_data.append(np.array(x[1:4], dtype=np.float))
                ele_data.append(x[4])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return coo_data, ele_data, np.array(lat_data)


def _create_structure(row, file_dir):
    filename = os.path.join(file_dir, str(int(row['id'])), 'geometry.xyz')
    coo_data, ele_data, lat_data = get_xyz_data(filename)

    lattice = Lattice.from_parameters(a=row['lattice_vector_1_ang'],
                                      b=row['lattice_vector_2_ang'],
                                      c=row['lattice_vector_3_ang'],
                                      alpha=row['lattice_angle_alpha_degree'],
                                      beta=row['lattice_angle_beta_degree'],
                                      gamma=row['lattice_angle_gamma_degree'])
    struct = Structure(lattice, ele_data, coo_data, coords_are_cartesian=True)

    output = os.path.join(file_dir, str(int(row['id'])), 'geometry.json')
    with open(output, 'w') as f:
        json.dump(struct.as_dict(), f)

    return struct


def create_structure_train(row):
    return _create_structure(row, 'data/train/')


def create_structure_test(row):
    return _create_structure(row, 'data/test/')


if __name__ == '__main__':
    df_train = pd.read_csv(TRAIN_DATA)
    df_test = pd.read_csv(TEST_DATA)

    structures_train = df_train.apply(create_structure_train, axis=1)
    structures_test = df_test.apply(create_structure_test, axis=1)
