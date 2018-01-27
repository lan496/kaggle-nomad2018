from logging import getLogger
from functools import partial

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'

TRAIN_ADDITIONAL_DATA = '../data/oxygen_arr.train'
TEST_ADDITIONAL_DATA = '../data/oxygen_arr.test'

TRAIN_CN_DATA = '../data/coo_mat.train'
TEST_CN_DATA = '../data/coo_mat.test'

TRAIN_CE_DATA = '../data/total_coulomb_potential.train'
TEST_CE_DATA = '../data/total_coulomb_potential.test'


logger = getLogger(__name__)


def read_csv(path, additional_path, CN_path, coulomb_energy_path):
    logger.debug('enter')
    df = pd.read_csv(path)

    df = df.round({'lattice_vector_1_ang': 4,
                   'lattice_vector_2_ang': 4,
                   'lattice_vector_3_ang': 4,
                   'lattice_angle_alpha_degree': 3,
                   'lattice_angle_beta_degree': 3,
                   'lattice_angle_gamma_degree': 3})

    df['density'] = df['number_of_total_atoms'] / (df['lattice_vector_1_ang'] * df['lattice_vector_2_ang'] * df['lattice_vector_3_ang'] * np.sin(df['lattice_angle_alpha_degree'] * np.pi / 180) * np.sin(df['lattice_angle_beta_degree'] * np.pi / 180) * np.sin(df['lattice_angle_gamma_degree'] * np.pi / 180))
    alpha = np.radians(df['lattice_angle_alpha_degree'])
    beta = np.radians(df['lattice_angle_beta_degree'])
    gamma = np.radians(df['lattice_angle_gamma_degree'])
    df['dihedral_angle_alpha'] = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / np.sin(beta) / np.cos(gamma)
    df['c_by_a'] = df['lattice_vector_3_ang'] / df['lattice_vector_1_ang']

    logger.debug('OneHotEncode space group')
    spacegroup = pd.get_dummies(df['spacegroup'].astype('str'), drop_first=True, prefix='spacegroup')

    logger.debug('OneHotEncode number of total atoms')
    total_atoms = pd.get_dummies(df['number_of_total_atoms'].astype('str'), drop_first=True, prefix='number_of_total_atoms')

    df2 = pd.merge(df, spacegroup, left_index=True, right_index=True)
    df2.drop('spacegroup', axis=1, inplace=True)

    df3 = pd.merge(df2, total_atoms, left_index=True, right_index=True)
    df3.drop('number_of_total_atoms', axis=1, inplace=True)

    df_oxygen_ave = pd.DataFrame(joblib.load(additional_path), columns=['oxygen_density'])

    cols_nonzero = np.all(joblib.load(TRAIN_CN_DATA), axis=0) + np.all(joblib.load(TEST_CN_DATA), axis=0)
    df_CN = pd.DataFrame(joblib.load(CN_path)[:, ~cols_nonzero])

    arr_tcp = joblib.load(coulomb_energy_path)
    df_tcp = pd.DataFrame(arr_tcp, columns=['coulomb_potential_by_atom'])
    df_tcp['coulomb_potential_by_atom'] /= df['number_of_total_atoms']

    # df4 = pd.concat([df3, df_oxygen_ave, df_CN, df_tcp], axis=1)
    df4 = pd.concat([df3, df_oxygen_ave, df_CN, ], axis=1)
    logger.debug('exit')

    return df4


def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA, TRAIN_ADDITIONAL_DATA, TRAIN_CN_DATA, TRAIN_CE_DATA)

    # https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/discussion/47998
    duplicate_index = np.array([395, 126, 1215, 1886, 2075, 353, 308, 2154, 531, 1379, 2319, 2337, 2370, 2333])

    df.drop(duplicate_index - 1, axis=0, inplace=True)

    logger.debug('exit')

    return df


def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA, TEST_ADDITIONAL_DATA, TEST_CN_DATA, TEST_CE_DATA)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
