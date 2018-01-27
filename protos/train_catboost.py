from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
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

            model = CatBoostRegressor(**params)
            model.fit(trn_X, trn_y,
                      eval_set=(val_X, val_y),
                      use_best_model=True)
            pred = model.predict(val_X)
            loss_rmse = np.sqrt(mean_squared_error(val_y, pred))

            list_loss.append(loss_rmse)
            list_best_iterations.append(model.tree_count_)
            logger.debug('  RMSE: {}'.format(loss_rmse))

        params['iterations'] = int(np.mean(list_best_iterations))
        loss_rmse = np.mean(list_loss)
        logger.debug('RMSE: {}'.format(loss_rmse))
        if min_loss > loss_rmse:
            min_loss = loss_rmse
            argmin_loss = params

    logger.info('argmin RMSE: {}'.format(argmin_loss))
    logger.info('minimum RMSE: {}'.format(min_loss))

    model = CatBoostRegressor(**argmin_loss)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return pred, min_loss, argmin_loss


if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_catboost.py.log', 'a')
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
        'iterations': [50000],
        'eval_metric': ['RMSE'],
        'loss_function': ['RMSE'],
        'random_seed': [0],
        'thread_count': [8],
        'logging_level': ['Silent'],
        'od_type': ['Iter'],
        'od_wait': [50],
        'learning_rate': [0.1],
        'depth': [3, 4, 5, 6, 7],
        'l2_leaf_reg': [0, 0.01, 0.1],
    }

    y_fe_pred_test, fe_min_loss, fe_argmin_loss = run(all_params, X_train, y_fe_train, X_test, n_splits=5)

    logger.info('formation_energy_ev_natom end')

    y_bg_pred_test, bg_min_loss, bg_argmin_loss = run(all_params, X_train, y_bg_train, X_test, n_splits=5)

    logger.info('bandgap_energy_ev end')

    logger.info('estimated RMSE: {}'.format((fe_min_loss + bg_min_loss) / 2))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, np.expm1(y_fe_pred_test))
    df_submit['bandgap_energy_ev'] = np.maximum(0, np.expm1(y_bg_pred_test))

    df_submit.to_csv(DIR + 'submit_catboost.csv', index=False)
