from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_log_error, mean_squared_error
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

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train = load_train_data()
    X_train = df_train.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_fe_train = np.log(1e-8 + df_train['formation_energy_ev_natom'].values)
    y_fe_train_org = df_train['formation_energy_ev_natom'].values
    y_bg_train = np.log(1e-8 + df_train['bandgap_energy_ev'].values)
    y_bg_train_org = df_train['bandgap_energy_ev'].values

    use_cols = X_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(X_train.shape))

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    all_params = {
        'C': [10 ** i for i in range(-2, 3)],
        'epsilon': [10 ** i for i in range(-3, 0)],
        'kernel': ['rbf'],
        'gamma': ['auto'],
        'shrinking': [True, False],
        'tol': [10 ** i for i in range(-5, 0)],
    }

    min_score_fe = 100
    argmin_params_fe = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.debug('params: {}'.format(params))

        list_fe_rmse = []
        list_fe_rmsle = []
        for train_idx, valid_idx in cv.split(X_train, y_fe_train):
            trn_X = X_train.iloc[train_idx, :]
            val_X = X_train.iloc[valid_idx, :]

            trn_y_fe = y_fe_train[train_idx]
            val_y_fe = y_fe_train[valid_idx]

            clf_fe = SVR(**params)
            clf_fe.fit(X_train, y_fe_train)
            pred = clf_fe.predict(val_X)
            sc_rmse = np.sqrt(mean_squared_error(np.exp(val_y_fe), np.exp(pred)))
            sc_rmsle = np.sqrt(mean_squared_log_error(np.exp(val_y_fe), np.exp(pred)))

            list_fe_rmse.append(sc_rmse)
            list_fe_rmsle.append(sc_rmsle)
            logger.debug('  RMSE: {}, RMSLE: {}'.format(sc_rmse, sc_rmsle))

        sc_rmse = np.mean(list_fe_rmse)
        sc_rmsle = np.mean(list_fe_rmsle)
        logger.debug('RMSE: {}, RMSLE: {}'.format(sc_rmse, sc_rmsle))
        if min_score_fe > sc_rmsle:
            min_score_fe = sc_rmsle
            argmin_params_fe = params

    logger.info('argmin RMSLE: {}'.format(argmin_params_fe))
    logger.info('minimum RMSLE: {}'.format(min_score_fe))

    clf_fe = SVR(**argmin_params_fe)
    clf_fe.fit(X_train, y_fe_train)

    logger.info('formation_energy_ev_natom train end')

    min_score_bg = 100
    argmin_params_bg = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.debug('params: {}'.format(params))

        list_bg_rmse = []
        list_bg_rmsle = []
        for train_idx, valid_idx in cv.split(X_train, y_fe_train):
            trn_X = X_train.iloc[train_idx, :]
            val_X = X_train.iloc[valid_idx, :]

            trn_y_bg = y_bg_train[train_idx]
            val_y_bg = y_bg_train[valid_idx]

            clf_bg = SVR(**params)
            clf_bg.fit(X_train, y_bg_train)
            pred = clf_bg.predict(val_X)
            sc_rmse = np.sqrt(mean_squared_error(np.exp(val_y_bg), np.exp(pred)))
            sc_rmsle = np.sqrt(mean_squared_log_error(np.exp(val_y_bg), np.exp(pred)))

            list_bg_rmse.append(sc_rmse)
            list_bg_rmsle.append(sc_rmsle)
            logger.debug('  RMSE: {}, RMSLE: {}'.format(sc_rmse, sc_rmsle))

            sc_rmse = np.mean(list_bg_rmse)
            sc_rmsle = np.mean(list_bg_rmsle)
        logger.debug('RMSE: {}, RMSLE: {}'.format(sc_rmse, sc_rmsle))
        if min_score_bg > sc_rmsle:
            min_score_bg = sc_rmsle
            argmin_params_bg = params

    logger.info('argmin RMSLE: {}'.format(argmin_params_bg))
    logger.info('minimum RMSLE: {}'.format(min_score_bg))

    clf_bg = SVR(**argmin_params_bg)
    clf_bg.fit(X_train, y_bg_train)

    logger.info('bandgap_energy_ev train end')

    df_test = load_test_data()
    X_test = df_test[use_cols].sort_values('id')

    logger.info('test data load end {}'.format(X_test.shape))

    logger.info('estimated RMSLE: {}'.format((min_score_fe + min_score_bg) / 2))

    y_fe_pred_test = clf_fe.predict(X_test)
    y_bg_pred_test = clf_bg.predict(X_test)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.exp(y_fe_pred_test)
    df_submit['bandgap_energy_ev'] = np.exp(y_bg_pred_test)

    df_submit.to_csv(DIR + 'submit.csv', index=False)
