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

logger = getLogger(__name__)


def read_csv(path, additional_path):
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

    oxygen_ave = joblib.load(additional_path)
    df4 = pd.concat([df3, pd.DataFrame(oxygen_ave)], axis=1)

    logger.debug('exit')

    return df4


def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA, TRAIN_ADDITIONAL_DATA)
    logger.debug('exit')

    return df


def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA, TEST_ADDITIONAL_DATA)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
