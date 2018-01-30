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

TRAIN_ANGLE_DATA = '../data/ha_df_20.train'
TEST_ANGLE_DATA = '../data/ha_df_20.test'

TRAIN_TRI_DATA = '../data/tri_df.train'
TEST_TRI_DATA = '../data/tri_df.test'

TRAIN_TOTAL_DATA = '../data/total_df.train'
TEST_TOTAL_DATA = '../data/total_df.test'

TRAIN_EHIST_DATA = '../data/ehist_df_20.train'
TEST_EHIST_DATA = '../data/ehist_df_20.test'

TRAIN_COULOMB_DATA = '../data/CM_mat.train'
TEST_COULOMB_DATA = '../data/CM_mat.test'

logger = getLogger(__name__)


def read_csv(path, additional_path, CN_path, angle_path, tri_path, total_path, ehist_path, coulomb_path, is_bg=False):
    logger.debug('enter')
    df = pd.read_csv(path)

    df = df.round({'lattice_vector_1_ang': 4,
                   'lattice_vector_2_ang': 4,
                   'lattice_vector_3_ang': 4,
                   'lattice_angle_alpha_degree': 3,
                   'lattice_angle_beta_degree': 3,
                   'lattice_angle_gamma_degree': 3})

    alpha = np.radians(df['lattice_angle_alpha_degree'])
    beta = np.radians(df['lattice_angle_beta_degree'])
    gamma = np.radians(df['lattice_angle_gamma_degree'])
    volume = df['lattice_vector_1_ang'] * df['lattice_vector_2_ang'] * df['lattice_vector_3_ang'] * np.sin(alpha) * np.sin(beta) * np.sin(gamma)

    # second features
    df['density'] = df['number_of_total_atoms'] / volume
    df['dihedral_angle_alpha'] = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / np.sin(beta) / np.cos(gamma)
    df['c_by_a'] = df['lattice_vector_3_ang'] / df['lattice_vector_1_ang']

    spacegroup = pd.get_dummies(df['spacegroup'].astype('str'), drop_first=True, prefix='spacegroup')

    total_atoms = pd.get_dummies(df['number_of_total_atoms'].astype('str'), drop_first=True, prefix='number_of_total_atoms')

    # features from crystal geometry
    df_oxygen_ave = pd.DataFrame(joblib.load(additional_path), columns=['oxygen_density'])

    cols_nonzero = np.all(joblib.load(TRAIN_CN_DATA), axis=0) + np.all(joblib.load(TEST_CN_DATA), axis=0)
    df_CN = pd.DataFrame(joblib.load(CN_path)[:, ~cols_nonzero])

    df_angle = joblib.load(angle_path)
    df_angle = df_angle.divide(df['number_of_total_atoms'], axis=0)

    df_tri = joblib.load(tri_path)
    df_tri = df_tri.divide(df['number_of_total_atoms'], axis=0)

    df_total = joblib.load(total_path)
    df_total = df_total.divide(df['number_of_total_atoms'], axis=0)

    df_ehist = joblib.load(ehist_path)

    common_dataframs = [df, spacegroup, total_atoms, df_oxygen_ave, df_CN, df_angle, ]
    if is_bg:
        df2 = pd.concat([*common_dataframs, df_tri, df_ehist], axis=1)
    else:
        df2 = pd.concat([*common_dataframs], axis=1)

    df2.drop(['spacegroup', 'number_of_total_atoms'], axis=1, inplace=True)
    logger.debug('exit')

    return df2


def load_train_data(is_bg=False):
    logger.debug('enter')
    df = read_csv(TRAIN_DATA,
                  TRAIN_ADDITIONAL_DATA,
                  TRAIN_CN_DATA,
                  TRAIN_ANGLE_DATA,
                  TRAIN_TRI_DATA,
                  TRAIN_TOTAL_DATA,
                  TRAIN_EHIST_DATA,
                  TRAIN_COULOMB_DATA,
                  is_bg)

    # https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/discussion/47998
    duplicate_index = np.array([395, 126, 1215, 1886, 2075, 353, 308, 2154, 531, 1379, 2319, 2337, 2370, 2333])

    df.drop(duplicate_index - 1, axis=0, inplace=True)

    logger.debug('exit')

    return df


def load_test_data(is_bg=False):
    logger.debug('enter')
    df = read_csv(TEST_DATA,
                  TEST_ADDITIONAL_DATA,
                  TEST_CN_DATA,
                  TEST_ANGLE_DATA,
                  TEST_TRI_DATA,
                  TEST_TOTAL_DATA,
                  TEST_EHIST_DATA,
                  TEST_COULOMB_DATA,
                  is_bg)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
