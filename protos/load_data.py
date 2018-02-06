from logging import getLogger
from functools import partial

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import scale, StandardScaler
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

TRAIN_FINGER_DATA = '../data/fp_df.train'
TEST_FINGER_DATA = '../data/fp_df.test'

TRAIN_MOMENT_DATA = '../data/fp_moment_df.train'
TEST_MOMENT_DATA = '../data/fp_moment_df.test'

TRAIN_GRDF_DATA = '../data/grdf_df.train'
TEST_GRDF_DATA = '../data/grdf_df.test'


logger = getLogger(__name__)


def calc_cations_ave(df, df_elements):
    df_ave = pd.DataFrame()

    use_cols = ['First Ionization energy', 'Second Ionization energy', 'Atomic mass', 'Liquid range', 'Boiling point']

    arr_ratio = df.loc[:, ['percent_atom_al', 'percent_atom_ga', 'percent_atom_in']].values
    arr_property = df_elements.loc[['Al', 'Ga', 'In'], use_cols].values

    columns = ['ave_{}'.format(col) for col in use_cols]

    arr_ave = np.sum(arr_ratio[:, :, np.newaxis] * arr_property[np.newaxis, :, :], axis=1)
    df_ave = pd.DataFrame(arr_ave, columns=columns)

    return df_ave


def calc_elements_ave(df, df_elements):
    use_cols = ['First Ionization energy', 'Second Ionization energy', 'Atomic mass', 'Liquid range', 'Boiling point']
    # use_cols = df_elements.columns

    df_ratio = pd.DataFrame()
    df_ratio['Al'] = 0.4 * df.loc[:, 'percent_atom_al']
    df_ratio['Ga'] = 0.4 * df.loc[:, 'percent_atom_ga']
    df_ratio['In'] = 0.4 * df.loc[:, 'percent_atom_in']
    df_ratio['O'] = 0.6

    arr_ratio = df_ratio.loc[:, ['Al', 'Ga', 'In', 'O']].values

    arr_property = df_elements.loc[['Al', 'Ga', 'In', 'O'], use_cols].values

    columns = ['ave_{}'.format(col) for col in use_cols]

    arr_ave = np.sum(arr_ratio[:, :, np.newaxis] * arr_property[np.newaxis, :, :], axis=1)
    df_ave = pd.DataFrame(arr_ave, columns=columns)

    return df_ave


def read_csv(path, additional_path, CN_path, angle_path, tri_path, total_path, ehist_path,
             finger_path, moment_path, grdf_path,
             is_bg=False, is_nn=False):
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
    # df['dihedral_angle_alpha'] = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / np.sin(beta) / np.cos(gamma)
    df['c_by_a'] = df['lattice_vector_3_ang'] / df['lattice_vector_1_ang']

    spacegroup = pd.get_dummies(df['spacegroup'].astype('str'), drop_first=True, prefix='spacegroup')

    total_atoms = pd.get_dummies(df['number_of_total_atoms'].astype('str'), drop_first=True, prefix='number_of_total_atoms')

    if is_nn:
        df2 = pd.concat([df, spacegroup, total_atoms], axis=1)
        df2.drop(['density', 'dihedral_angle_alpha', 'c_by_a'], axis=1, inplace=True)
    else:
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

        df_fp = joblib.load(finger_path)

        df_moment = joblib.load(moment_path)

        # df_elements = joblib.load('../data/elements_property.joblib')
        # df_ele_ave = calc_elements_ave(df, df_elements)

        common_dataframs = [df, spacegroup, total_atoms, df_oxygen_ave,
                            df_CN, df_angle, ]
        if is_bg:
            df2 = pd.concat([*common_dataframs, df_tri, df_ehist], axis=1)
        else:
            df2 = pd.concat([*common_dataframs, df_fp, df_moment], axis=1)

    df2.drop(['spacegroup', 'number_of_total_atoms'], axis=1, inplace=True)

    logger.debug('exit')

    return df2


def load_train_data(is_bg=False, is_nn=False):
    logger.debug('enter')
    df = read_csv(TRAIN_DATA,
                  TRAIN_ADDITIONAL_DATA,
                  TRAIN_CN_DATA,
                  TRAIN_ANGLE_DATA,
                  TRAIN_TRI_DATA,
                  TRAIN_TOTAL_DATA,
                  TRAIN_EHIST_DATA,
                  TRAIN_FINGER_DATA,
                  TRAIN_MOMENT_DATA,
                  TRAIN_GRDF_DATA,
                  is_bg,
                  is_nn)

    # https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/discussion/47998
    duplicate_index = np.array([395, 126, 1215, 1886, 2075, 353, 308, 2154, 531, 1379, 2319, 2337, 2370, 2333])

    df.drop(duplicate_index - 1, axis=0, inplace=True)

    logger.debug('exit')

    return df


def load_test_data(is_bg=False, is_nn=False):
    logger.debug('enter')
    df = read_csv(TEST_DATA,
                  TEST_ADDITIONAL_DATA,
                  TEST_CN_DATA,
                  TEST_ANGLE_DATA,
                  TEST_TRI_DATA,
                  TEST_TOTAL_DATA,
                  TEST_EHIST_DATA,
                  TEST_FINGER_DATA,
                  TEST_MOMENT_DATA,
                  TEST_GRDF_DATA,
                  is_bg,
                  is_nn)
    logger.debug('exit')
    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
