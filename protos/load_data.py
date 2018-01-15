from logging import getLogger

import pandas as pd

TRAIN_DATA = '../data/train.csv'
TEST_DATA = '../data/test.csv'

logger = getLogger(__name__)


def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)

    logger.debug('OneHotEncode space group')
    spacegroup = pd.get_dummies(df['spacegroup'].astype('str'), drop_first=True)

    df2 = pd.merge(df, spacegroup, left_index=True, right_index=True)
    df2.drop('spacegroup', axis=1, inplace=True)

    logger.debug('exit')

    return df2


def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')

    return df


def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    logger.debug('exit')

    return df


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
