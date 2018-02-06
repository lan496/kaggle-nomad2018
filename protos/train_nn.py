from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../data/sample_submission.csv'


def mean_rmse(y_true, y_pred):
    mrmse = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.squared_difference(y_true, y_pred), 0)))
    return mrmse


def run(all_params, X_train, y_train, X_test, n_splits=5):
    n_features = X_train.shape[1]

    min_loss = 100
    argmin_loss = None

    trn_X, val_X, trn_y, val_y = train_test_split(X_train, y_train,
                                                  test_size=0.2, shuffle=True, random_state=0)

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.debug('params: {}'.format(params))

        l2_reg = params['l2_reg']
        dropout = params['dropout']
        epochs = params['epochs']
        batch_size = params['batch_size']

        # Model definition
        model = Sequential()
        model.add(Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(l2_reg),
                        input_dim=n_features))
        model.add(Dropout(dropout))
        model.add(Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(l2_reg)))
        model.add(Dropout(dropout))
        model.add(Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(l2_reg)))
        model.add(Dropout(dropout))
        model.add(Dense(2,
                        kernel_regularizer=keras.regularizers.l2(l2_reg)))

        model.compile(optimizer='rmsprop',
                      loss=mean_rmse,
                      metrics=[mean_rmse])

        model.fit(trn_X, trn_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(val_X, val_y),
                  steps_per_epoch=None)

        loss_rmse = model.evaluate(val_X, val_y,
                                   batch_size=batch_size,
                                   verbose=1,
                                   steps=None)

        loss_rmse = np.mean(loss_rmse)

        logger.debug('RMSE: {}'.format(loss_rmse))
        if min_loss > loss_rmse:
            min_loss = loss_rmse
            argmin_loss = params

    logger.info('argmin RMSE: {}'.format(argmin_loss))
    logger.info('minimum RMSE: {}'.format(min_loss))

    l2_reg = argmin_loss['l2_reg']
    dropout = argmin_loss['dropout']
    epochs = argmin_loss['epochs']
    batch_size = argmin_loss['batch_size']

    model = Sequential()
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                    input_dim=n_features))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(2,
                    kernel_regularizer=keras.regularizers.l2(l2_reg)))

    model.compile(optimizer='rmsprop',
                  loss=mean_rmse,
                  metrics=[mean_rmse])

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              steps_per_epoch=None)

    pred = model.predict(X_test,
                         batch_size=batch_size,
                         verbose=1,
                         steps=None)

    return pred, min_loss, argmin_loss


if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train_nn.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    # df_train = load_train_data()
    df_train = load_train_data(is_nn=True)
    X_train = df_train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    y_train = np.log1p(df_train[['formation_energy_ev_natom', 'bandgap_energy_ev']].values)

    logger.info('data preparation end {}'.format(X_train.shape))

    # df_test = load_test_data()
    df_test = load_test_data(is_nn=True)
    X_test = df_test.sort_values('id')
    X_test.drop(['id'], axis=1, inplace=True)

    logger.info('test data load end {}'.format(X_test.shape))

    all_params = {'random_state': [0],
                  'batch_size': [2],
                  'l2_reg': [0],
                  'epochs': [300],
                  'dropout': [0.1],
                  }

    y_pred_test, min_loss, argmin_loss = run(all_params, X_train, y_train, X_test, n_splits=5)

    logger.info('formation_energy_ev_natom end')
    logger.info('bandgap_energy_ev end')

    logger.info('estimated RMSE: {}'.format(min_loss))

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['formation_energy_ev_natom'] = np.maximum(0, np.expm1(y_pred_test[:, 0]))
    df_submit['bandgap_energy_ev'] = np.maximum(0, np.expm1(y_pred_test[:, 1]))

    df_submit.to_csv(DIR + 'submit_nn.csv', index=False)
