from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from tqdm import tqdm

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../data/sample_submission.csv'


def run(all_params, X_train, y_train, X_test, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    min_loss = 100
    argmin_loss = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.debug('params: {}'.format(params))

        list_loss = []
        list_best_iterations = []

        for train_idx, valid_idx in cv.split(X_train, y_train):
            trn_X = X_train.iloc[train_idx, :]
            val_X = X_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(trn_X, trn_y,
                      eval_set=[(val_X, val_y)],
                      eval_metric='l2',
                      early_stopping_rounds=10,
                      verbose=False)
            pred = model.predict(val_X, num_iteration=model.best_iteration_)
            loss_rmse = np.sqrt(mean_squared_error(val_y, pred))

            list_loss.append(loss_rmse)
            list_best_iterations.append(model.best_iteration_)
            logger.debug('  RMSE: {}'.format(loss_rmse))

        params['n_estimators'] = int(np.mean(list_best_iterations))
        loss_rmse = np.mean(list_loss)
        logger.debug('RMSE: {}'.format(loss_rmse))
        if min_loss > loss_rmse:
            min_loss = loss_rmse
            argmin_loss = params

    logger.info('argmin RMSE: {}'.format(argmin_loss))
    logger.info('minimum RMSE: {}'.format(min_loss))

    model = lgb.LGBMRegressor(**argmin_loss)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred, min_loss, argmin_loss


if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_lightgbm.py.log', 'a')
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

    df_test_fe = load_test_data()
    df_test_bg = load_test_data(is_bg=True)
    X_test_fe = df_test_fe.sort_values('id')
    X_test_bg = df_test_bg.sort_values('id')
    X_test_fe.drop(['id'], axis=1, inplace=True)
    X_test_bg.drop(['id'], axis=1, inplace=True)

    logger.info('test data load end {}'.format(X_test_fe.shape))

    all_params = {
        # core paramters
        'n_estimators': [1000],
        'random_state': [0],
        'n_jobs': [-1],
        'silent': [False],

        # controling complexity parameters
        'max_depth': [-1],
        'num_leaves': [2 ** i for i in range(2, 11)],
        'min_child_samples': [32],
        'max_bin': [255],

        'learning_rate': [0.01, 0.1],
        'boosting_type': ['gbdt', 'dart'],
    }
    # all_params = {'random_state': [0]}

    y_fe_pred_test, fe_min_loss, fe_argmin_loss = run(all_params, X_train_fe, y_fe_train, X_test_fe, n_splits=5)

    logger.info('formation_energy_ev_natom end')

    y_bg_pred_test, bg_min_loss, bg_argmin_loss = run(all_params, X_train_bg, y_bg_train, X_test_bg, n_splits=5)

    logger.info('bandgap_energy_ev end')

    logger.info('estimated RMSE: {}'.format((fe_min_loss + bg_min_loss) / 2))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, np.expm1(y_fe_pred_test))
    df_submit['bandgap_energy_ev'] = np.maximum(0, np.expm1(y_bg_pred_test))

    df_submit.to_csv(DIR + 'submit_lightgbm.csv', index=False)
