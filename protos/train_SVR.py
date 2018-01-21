from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, ParameterGrid, GridSearchCV
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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

    handler = FileHandler(DIR + 'train_SVR.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train = load_train_data()
    X_train = df_train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_fe_train = np.log1p(df_train['formation_energy_ev_natom'].values)
    y_bg_train = np.log1p(df_train['bandgap_energy_ev'].values)

    use_cols = X_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(X_train.shape))

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    all_params = {
        'C': [10 ** i for i in range(-2, 4)],
        'epsilon': [10 ** i for i in range(-4, 1)],
        'kernel': ['rbf'],
        'gamma': ['auto'],
        'shrinking': [True, False],
        'tol': [10 ** i for i in range(-4, 1)],
    }

    fe_gs = GridSearchCV(SVR(), all_params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=1)
    fe_gs.fit(X_train, y_fe_train)
    clf_fe = SVR(**fe_gs.best_params_)
    clf_fe.fit(X_train, y_fe_train)

    logger.info('formation_energy_ev_natom train end')

    bg_gs = GridSearchCV(SVR(), all_params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=1)
    bg_gs.fit(X_train, y_bg_train)
    clf_bg = SVR(**bg_gs.best_params_)
    clf_bg.fit(X_train, y_bg_train)

    logger.info('bandgap_energy_ev train end')

    df_test = load_test_data()
    X_test = df_test[use_cols]

    logger.info('test data load end {}'.format(X_test.shape))

    min_score_fe = np.sqrt(-fe_gs.best_score_)
    min_score_bg = np.sqrt(-bg_gs.best_score_)
    logger.info('argmin RMSE: {}'.format(fe_gs.best_params_))
    logger.info('minimum RMSE: {}'.format(min_score_fe))
    logger.info('argmin RMSE: {}'.format(bg_gs.best_params_))
    logger.info('minimum RMSE: {}'.format(min_score_bg))

    logger.info('estimated RMSE: {}'.format((min_score_fe + min_score_bg) / 2))

    y_fe_pred_test = np.expm1(clf_fe.predict(X_test))
    y_bg_pred_test = np.expm1(clf_bg.predict(X_test))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, y_fe_pred_test)
    df_submit['bandgap_energy_ev'] = np.maximum(0, y_bg_pred_test)

    df_submit.to_csv(DIR + 'submit_SVR.csv', index=False)
