from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import xgboost as xgb
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../data/sample_submission.csv'


def loss(params, X_train, y_train, cv, early_stopping_rounds=50):
    logger.debug('params: {}'.format(params))

    list_loss = []
    list_best_iterations = []
    list_best_ntree_limit = []

    for train_idx, valid_idx in cv.split(X_train, y_train):
        trn_X = X_train.iloc[train_idx, :]
        val_X = X_train.iloc[valid_idx, :]

        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(trn_X, trn_y,
                  eval_set=[(val_X, val_y)],
                  early_stopping_rounds=early_stopping_rounds,
                  eval_metric='rmse',
                  verbose=False)
        pred = model.predict(val_X, ntree_limit=model.best_ntree_limit)
        loss = np.sqrt(mean_squared_error(val_y, pred))

        list_loss.append(loss)
        list_best_iterations.append(model.best_iteration)
        list_best_ntree_limit.append(model.best_ntree_limit)
        logger.debug('  RMSE: {}'.format(loss))

    params['n_estimators'] = int(np.mean(list_best_iterations))
    params['ntree_limit'] = int(np.mean(list_best_ntree_limit))
    loss = np.mean(list_loss)

    logger.debug('RMSE: {}'.format(loss))

    return {'loss': loss, 'status': STATUS_OK}


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
    X_train = df_train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_fe_train = np.log1p(df_train['formation_energy_ev_natom'].values)
    y_bg_train = np.log1p(df_train['bandgap_energy_ev'].values)

    use_cols = X_train.columns.values
    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('data preparation end {}'.format(X_train.shape))

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    space = {
        # Control complexity of model
        'max_depth': hp.choice('max_depth', np.arange(1, 11, dtype=int)),
        'learning_rate': 0.1,
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 11, dtype=int)),
        'gamma': hp.quniform('gamma', 0, 1, 0.1),

        # Improve noise robustness
        'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
        'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1, 0.1),
        'reg_alpha': hp.quniform('reg_alpha', 0, 1, 0.1),
        'reg_lambda': hp.quniform('reg_lambda', 0, 1, 0.1),

        # booster
        'booster': 'dart',

        # fixed
        'n_estimators': 1000,
        'n_jobs': -1,
        'silent': True,
        'random_state': 0,
        'objective': 'reg:linear',
    }

    max_evals = 300
    early_stopping_rounds = 50

    trials_fe = Trials()
    loss_fe = partial(loss, X_train=X_train, y_train=y_fe_train, cv=cv, early_stopping_rounds=early_stopping_rounds)
    best_fe = fmin(loss_fe, space, algo=tpe.suggest, trials=trials_fe, max_evals=max_evals)
    best_params_fe = space_eval(space, best_fe)
    best_loss_fe = loss_fe(best_params_fe)['loss']

    logger.info('argmin RMSE: {}'.format(best_fe))
    logger.info('minimum RMSE: {}'.format(best_loss_fe))

    model_fe = xgb.XGBRegressor(**best_fe)
    model_fe.fit(X_train, y_fe_train)

    logger.info('formation_energy_ev_natom train end')

    trials_bg = Trials()
    loss_bg = partial(loss, X_train=X_train, y_train=y_bg_train, cv=cv, early_stopping_rounds=early_stopping_rounds)
    best_bg = fmin(loss_bg, space, algo=tpe.suggest, trials=trials_bg, max_evals=max_evals)
    best_params_bg = space_eval(space, best_bg)
    best_loss_bg = loss_bg(best_params_bg)['loss']

    logger.info('argmin RMSE: {}'.format(best_bg))
    logger.info('minimum RMSE: {}'.format(best_loss_bg))

    model_bg = xgb.XGBRegressor(**best_params_bg)
    model_bg.fit(X_train, y_bg_train)

    logger.info('bandgap_energy_ev train end')

    df_test = load_test_data()
    X_test = df_test.sort_values('id')
    X_test.drop(['id'], axis=1, inplace=True)

    logger.info('test data load end {}'.format(X_test.shape))

    logger.info('estimated RMSE: {}'.format((best_loss_fe + best_loss_bg) / 2))

    y_fe_pred_test = np.expm1(model_fe.predict(X_test, ntree_limit=best_params_fe['ntree_limit']))
    y_bg_pred_test = np.expm1(model_bg.predict(X_test, ntree_limit=best_params_bg['ntree_limit']))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, y_fe_pred_test)
    df_submit['bandgap_energy_ev'] = np.maximum(0, y_bg_pred_test)

    df_submit.to_csv(DIR + 'submit_xgb_hyperopt.csv', index=False)
