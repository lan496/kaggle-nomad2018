from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../data/sample_submission.csv'


class MetaFeatures(object):

    def __init__(self, generalizers, n_splits=5):
        self.generalizers = generalizers
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    def guess_metafeatures_with_partition(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        pred = np.array([self._guess_metafeatures_with_partition(generalizer) for generalizer in self.generalizers])

        return np.transpose(pred)

    def _guess_metafeatures_with_partition(self, generalizer):
        pred = np.empty_like(self.y_train)

        for train_idx, valid_idx in self.cv.split(self.X_train, self.y_train):
            trn_X = self.X_train.iloc[train_idx, :]
            val_X = self.X_train.iloc[valid_idx, :]

            trn_y = self.y_train[train_idx]
            val_y = self.y_train[valid_idx]

            generalizer.fit(trn_X, trn_y)
            pred_tmp = generalizer.predict(val_X)

            pred[valid_idx] = pred_tmp

        return pred

    def guess_metafeatures_with_whole(self, X_test):
        self.X_test = X_test

        pred = np.array([self._guess_metafeatures_with_whole(generalizer) for generalizer in self.generalizers])

        return np.transpose(pred)

    def _guess_metafeatures_with_whole(self, generalizer):
        generalizer.fit(self.X_train, self.y_train)
        pred = generalizer.predict(self.X_test)
        return pred


def loss(params, X_train, y_train, cv):
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
        loss = np.sqrt(mean_squared_error(val_y, pred))

        list_loss.append(loss)
        list_best_iterations.append(model.best_iteration_)
        logger.debug('  RMSE: {}'.format(loss))

    params['n_estimators'] = int(np.mean(list_best_iterations))
    loss = np.mean(list_loss)

    logger.debug('params: {}'.format(params))
    logger.debug('RMSE: {}'.format(loss))

    return {'loss': loss, 'status': STATUS_OK}


def run(space, max_evals, generalizers, X_train, y_train, X_test, n_splits=5):
    mf = MetaFeatures(generalizers, n_splits)
    X_meta_train = mf.guess_metafeatures_with_partition(X_train, y_train)
    X_meta_test = mf.guess_metafeatures_with_whole(X_test)

    for i in range(X_meta_train.shape[1]):
        loss_rmse = np.sqrt(mean_squared_error(y_train, X_meta_train[:, i]))
        logger.info('   meta feature RMSE: {}'.format(loss_rmse))

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    trials = Trials()
    loss_partial = partial(loss, X_train=X_train, y_train=y_train, cv=cv)
    best = fmin(loss_partial, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    best_params = space_eval(space, best)
    best_loss = loss_partial(best_params)['loss']

    logger.info('argmin RMSE: {}'.format(best_params))
    logger.info('minimum RMSE: {}'.format(best_loss))

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred, best_loss, best_params


if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_stacked_generalization.py.log', 'a')
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

    df_test_fe = load_test_data(is_bg=False)
    df_test_bg = load_test_data(is_bg=True)
    X_test_fe = df_test_fe.sort_values('id')
    X_test_bg = df_test_bg.sort_values('id')
    X_test_fe.drop(['id'], axis=1, inplace=True)
    X_test_bg.drop(['id'], axis=1, inplace=True)

    logger.info('test data load end {}'.format(X_test_fe.shape))

    space = {
        # controling complexity parameters
        'max_depth': hp.choice('max_depth', np.arange(2, 11)),
        'num_leaves': hp.choice('num_leaves', np.arange(8, 128, 8)),
        'min_child_samples': hp.choice('min_child_samples', np.arange(8, 128, 8)),  # min_data_in_leaf
        'max_bin': hp.choice('max_bin', np.arange(128, 512, 8)),
        'subsample': hp.quniform('subsample', 0.5, 1.0, 0.1),  # bagging_fraction
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1., 0.1),  # feature_fraction
        'reg_alpha': hp.quniform('reg_alpha', 0, 1., 0.1),
        'reg_lambda': hp.quniform('reg_lambda', 0, 1., 0.1),

        'learning_rate': hp.quniform('learning_rate', 0.05, 0.1, 0.01),
        'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),

        # fixed
        'n_estimators': 1000,
        'random_state': 0,
        'n_jobs': 8,
        'silent': False,
        'application': 'regression_l2'
    }

    max_evals = 300

    generalizers_fe = [
        KernelRidge(alpha=0.01, gamma=0.001, kernel='laplacian'),
        RandomForestRegressor(criterion='mse',
                              max_depth=17,
                              max_features='auto',
                              min_samples_split=0.0001,
                              n_estimators=90,
                              n_jobs=-1,
                              random_state=0),
        KNeighborsRegressor(algorithm='auto',
                            leaf_size=5,
                            n_jobs=-1,
                            n_neighbors=5,
                            p=1,
                            weights='distance'),
        xgb.XGBRegressor(colsample_bylevel=0.7,
                         colsample_bytree=0.8,
                         gamma=0.0,
                         max_depth=4,
                         min_child_weight=5,
                         reg_alpha=0.0,
                         reg_lambda=0.4,
                         subsample=1.0,
                         learning_rate=0.1,
                         booster='dart',
                         n_estimators=369,
                         silent='True',
                         n_jobs=-1,
                         random_state=0,
                         objective='reg:linear'),
        CatBoostRegressor(depth=6,
                          eval_metric='RMSE',
                          iterations=650,
                          l2_leaf_reg=1.0,
                          learning_rate=0.1,
                          logging_level='Silent',
                          loss_function='RMSE',
                          od_type='Iter',
                          od_wait=50,
                          random_seed=0,
                          thread_count=8),
        lgb.LGBMRegressor(application='regression_l2',
                          boosting_type='gbdt',
                          colsample_bytree=0.6,
                          learning_rate=0.06,
                          max_bin=160,
                          max_depth=10,
                          min_child_samples=56,
                          n_estimators=224,
                          n_jobs=8,
                          num_leaves=120,
                          random_state=0,
                          reg_alpha=0.1,
                          reg_lambda=0.4,
                          silent=True,
                          subsample=0.8),
    ]

    generalizers_bg = [
        KernelRidge(alpha=0.01, gamma=0.001, kernel='laplacian'),
        RandomForestRegressor(criterion='mse',
                              max_depth=13,
                              max_features='auto',
                              min_samples_split=0.0001,
                              n_estimators=80,
                              n_jobs=-1,
                              random_state=0),
        KNeighborsRegressor(algorithm='auto',
                            leaf_size=30,
                            n_jobs=-1,
                            n_neighbors=15,
                            p=1,
                            weights='distance'),
        xgb.XGBRegressor(colsample_bylevel=0.5,
                         colsample_bytree=0.5,
                         gamma=0.0,
                         max_depth=3,
                         min_child_weight=9,
                         reg_alpha=0.8,
                         reg_lambda=1.0,
                         subsample=0.8,
                         learning_rate=0.1,
                         booster='dart',
                         n_estimators=209,
                         silent='True',
                         n_jobs=-1,
                         random_state=0,
                         objective='reg:linear'),
        CatBoostRegressor(depth=4,
                          eval_metric='RMSE',
                          iterations=266,
                          l2_leaf_reg=0.5,
                          learning_rate=0.1,
                          logging_level='Silent',
                          loss_function='RMSE',
                          od_type='Iter',
                          od_wait=50,
                          random_seed=0,
                          thread_count=8),
        lgb.LGBMRegressor(application='regression_l2',
                          boosting_type='gbdt',
                          colsample_bytree=0.5,
                          learning_rate=0.09,
                          max_bin=216,
                          max_depth=7,
                          min_child_samples=96,
                          n_estimators=144,
                          n_jobs=8,
                          num_leaves=120,
                          random_state=0,
                          reg_alpha=0.1,
                          reg_lambda=0.5,
                          silent=True,
                          subsample=0.8),
    ]

    y_fe_pred_test, fe_min_loss, fe_argmin_loss = run(space, max_evals, generalizers_fe, X_train_fe, y_fe_train, X_test_fe, n_splits=5)

    logger.info('formation_energy_ev_natom end')

    y_bg_pred_test, bg_min_loss, bg_argmin_loss = run(space, max_evals, generalizers_bg, X_train_bg, y_bg_train, X_test_bg, n_splits=5)

    logger.info('bandgap_energy_ev end')

    logger.info('estimated RMSE: {}'.format((fe_min_loss + bg_min_loss) / 2))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, np.expm1(y_fe_pred_test))
    df_submit['bandgap_energy_ev'] = np.maximum(0, np.expm1(y_bg_pred_test))

    df_submit.to_csv(DIR + 'submit_stacked_generalization.csv', index=False)
