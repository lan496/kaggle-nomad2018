from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    handler = FileHandler(DIR + 'train_xgb.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train_fe = load_train_data(is_bg=False)
    df_train_bg = load_train_data(is_bg=True)
    X_train_fe = df_train_fe.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    X_train_bg = df_train_bg.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_fe_train = np.log1p(df_train_fe['formation_energy_ev_natom'].values)
    y_bg_train = np.log1p(df_train_bg['bandgap_energy_ev'].values)

    logger.info('data preparation end {}'.format(X_train_fe.shape))

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    """
    all_params = {
        'max_depth': [4, 6],
        'learning_rate': [0.1],
        'min_child_weight': [10, 12],
        'n_estimators': [1000],
        'colsample_bytree': [0.6, 0.7],
        'colsample_bylevel': [0.7, 0.8],
        'reg_alpha': [0.1],
        'max_delta_step': [0, 0.01],
        'random_state': [0],
        'n_jobs': [-1],
        'silent': [True],
        'objective': ['reg:linear'],
        'booster': ['gbtree', 'dart'],
        'gamma': [0],
        'subsample': [1],
        'reg_lambda': [1],
    }
    """
    all_params = {'random_state': [0]}

    min_score_fe = 100
    argmin_params_fe = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.debug('params: {}'.format(params))

        list_fe_rmse = []
        list_best_iterations = []
        list_best_ntree_limit = []
        for train_idx, valid_idx in cv.split(X_train_fe, y_fe_train):
            trn_X = X_train_fe.iloc[train_idx, :]
            val_X = X_train_fe.iloc[valid_idx, :]

            trn_y_fe = y_fe_train[train_idx]
            val_y_fe = y_fe_train[valid_idx]

            clf_fe = xgb.XGBRegressor(**params)
            clf_fe.fit(trn_X, trn_y_fe,
                       eval_set=[(val_X, val_y_fe)],
                       early_stopping_rounds=100,
                       eval_metric='rmse',
                       verbose=False)
            pred = clf_fe.predict(val_X, ntree_limit=clf_fe.best_ntree_limit)
            sc_rmse = np.sqrt(mean_squared_error(val_y_fe, pred))

            list_fe_rmse.append(sc_rmse)
            list_best_iterations.append(clf_fe.best_iteration)
            list_best_ntree_limit.append(clf_fe.best_ntree_limit)
            logger.debug('  RMSE: {}'.format(sc_rmse))

        params['n_estimators'] = int(np.mean(list_best_iterations))
        params['ntree_limit'] = int(np.mean(list_best_ntree_limit))
        sc_rmse = np.mean(list_fe_rmse)
        logger.debug('RMSE: {}'.format(sc_rmse))
        if min_score_fe > sc_rmse:
            min_score_fe = sc_rmse
            argmin_params_fe = params

    clf_fe = xgb.XGBRegressor(**argmin_params_fe)
    clf_fe.fit(X_train_fe, y_fe_train)

    logger.info('formation_energy_ev_natom train end')

    min_score_bg = 100
    argmin_params_bg = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.debug('params: {}'.format(params))

        list_bg_rmse = []
        list_best_iterations = []
        list_best_ntree_limit = []
        for train_idx, valid_idx in cv.split(X_train_bg, y_fe_train):
            trn_X = X_train_bg.iloc[train_idx, :]
            val_X = X_train_bg.iloc[valid_idx, :]

            trn_y_bg = y_bg_train[train_idx]
            val_y_bg = y_bg_train[valid_idx]

            clf_bg = xgb.XGBRegressor(**params)
            clf_bg.fit(trn_X, trn_y_bg,
                       eval_set=[(val_X, val_y_bg)],
                       early_stopping_rounds=100,
                       eval_metric='rmse',
                       verbose=False)
            pred = clf_bg.predict(val_X, ntree_limit=clf_bg.best_ntree_limit)
            sc_rmse = np.sqrt(mean_squared_error(val_y_bg, pred))

            list_bg_rmse.append(sc_rmse)
            list_best_iterations.append(clf_bg.best_iteration)
            list_best_ntree_limit.append(clf_bg.best_ntree_limit)
            logger.debug('  RMSE: {}'.format(sc_rmse))

            sc_rmse = np.mean(list_bg_rmse)

        params['n_estimators'] = int(np.mean(list_best_iterations))
        params['ntree_limit'] = int(np.mean(list_best_ntree_limit))
        sc_rmse = np.mean(list_bg_rmse)
        logger.debug('RMSE: {}'.format(sc_rmse))
        if min_score_bg > sc_rmse:
            min_score_bg = sc_rmse
            argmin_params_bg = params

    logger.info('argmin fe RMSE: {}'.format(argmin_params_fe))
    logger.info('minimum fe RMSE: {}'.format(min_score_fe))
    logger.info('argmin bg RMSE: {}'.format(argmin_params_bg))
    logger.info('minimum bg RMSE: {}'.format(min_score_bg))

    clf_bg = xgb.XGBRegressor(**argmin_params_bg)
    clf_bg.fit(X_train_bg, y_bg_train)

    logger.info('bandgap_energy_ev train end')

    df_test_fe = load_test_data(is_bg=False)
    df_test_bg = load_test_data(is_bg=True)
    X_test_fe = df_test_fe.sort_values('id')
    X_test_bg = df_test_bg.sort_values('id')
    X_test_fe.drop(['id'], axis=1, inplace=True)
    X_test_bg.drop(['id'], axis=1, inplace=True)

    logger.info('estimated RMSE: {}'.format((min_score_fe + min_score_bg) / 2))

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    xgb.plot_importance(clf_fe, ax=ax1)
    xgb.plot_importance(clf_bg, ax=ax2)
    plt.show()

    y_fe_pred_test = np.expm1(clf_fe.predict(X_test_fe, ntree_limit=argmin_params_fe['ntree_limit']))
    y_bg_pred_test = np.expm1(clf_bg.predict(X_test_bg, ntree_limit=argmin_params_bg['ntree_limit']))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, y_fe_pred_test)
    df_submit['bandgap_energy_ev'] = np.maximum(0, y_bg_pred_test)

    df_submit.to_csv(DIR + 'submit_xgb.csv', index=False)
