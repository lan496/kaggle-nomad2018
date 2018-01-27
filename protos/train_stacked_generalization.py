from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import xgboost as xgb

from tqdm import tqdm

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


def run(all_params, generalizers, X_train, y_train, X_test, n_splits=5):
    mf = MetaFeatures(generalizers, n_splits)
    X_meta_train = mf.guess_metafeatures_with_partition(X_train, y_train)
    X_meta_test = mf.guess_metafeatures_with_whole(X_test)

    for i in range(X_meta_train.shape[1]):
        loss_rmse = np.sqrt(mean_squared_error(y_train, X_meta_train[:, i]))
        logger.info('   meta feature RMSE: {}'.format(loss_rmse))

    min_loss = 100
    argmin_loss = None

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.debug('params: {}'.format(params))

        list_loss = []
        list_best_iterations = []
        list_bets_ntree_limit = []

        for train_idx, valid_idx in cv.split(X_meta_train, y_train):
            trn_X = X_meta_train[train_idx, :]
            val_X = X_meta_train[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(trn_X, trn_y,
                      eval_set=[(val_X, val_y)],
                      early_stopping_rounds=50,
                      eval_metric='rmse',
                      verbose=False)
            pred = model.predict(val_X, ntree_limit=model.best_ntree_limit)
            loss_rmse = np.sqrt(mean_squared_error(val_y, pred))

            list_loss.append(loss_rmse)
            list_best_iterations.append(model.best_iteration)
            list_bets_ntree_limit.append(model.best_ntree_limit)
            logger.debug('  RMSE: {}'.format(loss_rmse))

        params['n_estimators'] = int(np.mean(list_best_iterations))
        params['ntree_limit'] = int(np.mean(list_bets_ntree_limit))
        loss_rmse = np.mean(list_loss)
        logger.debug('RMSE: {}'.format(loss_rmse))
        if min_loss > loss_rmse:
            min_loss = loss_rmse
            argmin_loss = params

    logger.info('argmin RMSE: {}'.format(argmin_loss))
    logger.info('minimum RMSE: {}'.format(min_loss))

    model = xgb.XGBRegressor(**argmin_loss)
    model.fit(X_train, y_train)

    pred = model.predict(X_test, ntree_limit=argmin_loss['ntree_limit'])

    return pred, min_loss, argmin_loss


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

    df_train = load_train_data()
    X_train = df_train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_fe_train = np.log1p(df_train['formation_energy_ev_natom'].values)
    y_bg_train = np.log1p(df_train['bandgap_energy_ev'].values)

    use_cols = X_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(X_train.shape))

    df_test = load_test_data()
    X_test = df_test.sort_values('id')
    X_test.drop(['id'], axis=1, inplace=True)

    logger.info('test data load end {}'.format(X_test.shape))

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

    generalizers_fe = [
        KernelRidge(alpha=0.01, gamma=0.01, kernel='laplacian'),
        RandomForestRegressor(criterion='mse',
                              max_depth=13,
                              max_features='auto',
                              min_samples_split=0.0001,
                              n_estimators=70,
                              n_jobs=-1,
                              random_state=0),
        xgb.XGBRegressor(colsample_bylevel=1.0,
                         colsample_bytree=0.8,
                         gamma=0.0,
                         max_depth=2,
                         min_child_weight=3,
                         reg_alpha=0.1,
                         reg_lambda=0.1,
                         subsample=0.7,
                         learning_rate=0.1,
                         booster='dart',
                         n_estimators=1000,
                         silent='True',
                         n_jobs=-1,
                         random_state=0,
                         objective='reg:linear'),
        CatBoostRegressor(depth=6,
                          eval_metric='RMSE',
                          iterations=284,
                          l2_leaf_reg=0.1,
                          learning_rate=0.1,
                          logging_level='Silent',
                          loss_function='RMSE',
                          od_type='Iter',
                          od_wait=50,
                          random_seed=0,
                          thread_count=8),
    ]

    generalizers_bg = [
        KernelRidge(alpha=0.01, gamma=0.001, kernel='laplacian'),
        RandomForestRegressor(criterion='mse',
                              max_depth=17,
                              max_features='auto',
                              min_samples_split=0.0001,
                              n_estimators=70,
                              n_jobs=-1,
                              random_state=0),
        xgb.XGBRegressor(colsample_bylevel=1.0,
                         colsample_bytree=0.5,
                         gamma=0.0,
                         max_depth=1,
                         min_child_weight=3,
                         reg_alpha=0.9,
                         reg_lambda=0.9,
                         subsample=0.6,
                         learning_rate=0.1,
                         booster='dart',
                         n_estimators=1000,
                         silent='True',
                         n_jobs=-1,
                         random_state=0,
                         objective='reg:linear'),
        CatBoostRegressor(depth=4,
                          eval_metric='RMSE',
                          iterations=186,
                          l2_leaf_reg=0.1,
                          learning_rate=0.1,
                          logging_level='Silent',
                          loss_function='RMSE',
                          od_type='Iter',
                          od_wait=50,
                          random_seed=0,
                          thread_count=8),
    ]

    y_fe_pred_test, fe_min_loss, fe_argmin_loss = run(all_params, generalizers_fe, X_train, y_fe_train, X_test, n_splits=5)

    logger.info('formation_energy_ev_natom end')

    y_bg_pred_test, bg_min_loss, bg_argmin_loss = run(all_params, generalizers_bg, X_train, y_bg_train, X_test, n_splits=5)

    logger.info('bandgap_energy_ev end')

    logger.info('estimated RMSE: {}'.format((fe_min_loss + bg_min_loss) / 2))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, np.expm1(y_fe_pred_test))
    df_submit['bandgap_energy_ev'] = np.maximum(0, np.expm1(y_bg_pred_test))

    df_submit.to_csv(DIR + 'submit_stacked_generalization.csv', index=False)
