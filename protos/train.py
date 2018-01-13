from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
from sklearn.svm import SVR

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../data/sample_submission.csv'

if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train = load_train_data()
    X_train = df_train.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_fe_train = df_train['formation_energy_ev_natom'].values
    y_bg_train = df_train['bandgap_energy_ev'].values

    use_cols = X_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(X_train.shape))

    clf_fe = SVR(kernel='linear')
    clf_fe.fit(X_train, y_fe_train)

    logger.info('formation_energy_ev_natom train end')

    clf_bg = SVR(kernel='linear')
    clf_bg.fit(X_train, y_bg_train)

    logger.info('bandgap_energy_ev train end')

    df_test = load_test_data()
    X_test = df_test[use_cols].sort_values('id')

    logger.info('test data load end {}'.format(X_test.shape))

    y_fe_pred_test = clf_fe.predict(X_test)
    y_bg_pred_test = clf_bg.predict(X_test)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = y_fe_pred_test
    df_submit['bandgap_energy_ev'] = y_bg_pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)
