from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from catboost import CatBoostRegressor
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../data/sample_submission.csv'


def loss(params, X_train, y_train, cv):

    list_loss = []
    list_best_iterations = []

    for train_idx, valid_idx in cv.split(X_train, y_train):
        trn_X = X_train.iloc[train_idx, :]
        val_X = X_train.iloc[valid_idx, :]

        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(trn_X, trn_y,
                  eval_set=(val_X, val_y),
                  use_best_model=True)
        pred = model.predict(val_X)
        loss = np.sqrt(mean_squared_error(val_y, pred))

        list_loss.append(loss)
        list_best_iterations.append(model.tree_count_)
        logger.debug('  RMSE: {}'.format(loss))

    params['iterations'] = int(np.mean(list_best_iterations))
    loss = np.mean(list_loss)

    logger.debug('params: {}'.format(params))
    logger.debug('RMSE: {}'.format(loss))

    return {'loss': loss, 'status': STATUS_OK}


if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_catboost_hyperopt.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df_train_fe = load_train_data()
    df_train_bg = load_train_data(is_bg=True)
    X_train_fe = df_train_fe.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    X_train_bg = df_train_bg.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_fe_train = np.log1p(df_train_fe['formation_energy_ev_natom'].values)
    y_bg_train = np.log1p(df_train_bg['bandgap_energy_ev'].values)

    logger.info('data preparation end {}'.format(X_train_fe.shape))

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    space = {
        # Control complexity of model
        'depth': hp.choice('depth', np.arange(2, 11, dtype=int)),
        'learning_rate': 0.1,

        # Improve noise robustness
        'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0, 1.0, 0.1),

        # fixed
        'iterations': 1000,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
        'od_wait': 50,
        'thread_count': 8,
        'logging_level': 'Silent',
        'random_seed': 0,
        'od_type': 'Iter'
    }

    max_evals = 300

    trials_fe = Trials()
    loss_fe = partial(loss, X_train=X_train_fe, y_train=y_fe_train, cv=cv)
    best_fe = fmin(loss_fe, space, algo=tpe.suggest, trials=trials_fe, max_evals=max_evals)
    best_params_fe = space_eval(space, best_fe)
    best_loss_fe = loss_fe(best_params_fe)['loss']

    logger.info('argmin RMSE: {}'.format(best_params_fe))
    logger.info('minimum RMSE: {}'.format(best_loss_fe))

    model_fe = CatBoostRegressor(**best_params_fe)
    model_fe.fit(X_train_fe, y_fe_train)

    logger.info('formation_energy_ev_natom train end')

    trials_bg = Trials()
    loss_bg = partial(loss, X_train=X_train_bg, y_train=y_bg_train, cv=cv)
    best_bg = fmin(loss_bg, space, algo=tpe.suggest, trials=trials_bg, max_evals=max_evals)
    best_params_bg = space_eval(space, best_bg)
    best_loss_bg = loss_bg(best_params_bg)['loss']

    logger.info('argmin RMSE: {}'.format(best_params_bg))
    logger.info('minimum RMSE: {}'.format(best_loss_bg))

    model_bg = CatBoostRegressor(**best_params_bg)
    model_bg.fit(X_train_bg, y_bg_train)

    logger.info('bandgap_energy_ev train end')

    df_test_fe = load_test_data()
    df_test_bg = load_test_data(is_bg=True)
    X_test_fe = df_test_fe.sort_values('id')
    X_test_bg = df_test_bg.sort_values('id')
    X_test_fe.drop(['id'], axis=1, inplace=True)
    X_test_bg.drop(['id'], axis=1, inplace=True)

    logger.info('test data load end {}'.format(X_test_fe.shape))

    logger.info('estimated RMSE: {}'.format((best_loss_fe + best_loss_bg) / 2))

    y_fe_pred_test = np.expm1(model_fe.predict(X_test_fe))
    y_bg_pred_test = np.expm1(model_bg.predict(X_test_bg))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, y_fe_pred_test)
    df_submit['bandgap_energy_ev'] = np.maximum(0, y_bg_pred_test)

    df_submit.to_csv(DIR + 'submit_catboost_hyperopt.csv', index=False)
