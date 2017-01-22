import os, sys
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as d
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.path.dirname(os.getcwd()))
from data.tweet import load_sen_df_from_local
from neural_network import MLPClassifier
from neural_network import RNNClassifier


def to_one_hot(vector):
    result = np.zeros((len(vector), 2))
    result[np.arange(len(vector)), vector] = 1
    return result


def get_trend_df(code, end_date):
    # get stock trends from 2016-08-01 to end_date_plus_one
    start_date = '2016-08-01'
    end_date_plus_one = str(datetime.datetime.strptime(end_date, "%Y-%m-%d") + pd.tseries.offsets.BDay(1))[:10]
    prices = d.DataReader(code, 'yahoo', start_date, end_date_plus_one)['Adj Close'].values
    trends = []
    for idx in range(1, len(prices)):
            trends.append(1) if prices[idx] > prices[idx - 1] else trends.append(0)

    df = load_sen_df_from_local()
    df = df[df['company'] == '$' + code] if code != '^GSPC' else df[df['company'] == '$SPX']
    # get integer location for requested label
    target_end_idx = df.index.get_loc(df[df['date'] == end_date].index[0])
    df = df.iloc[: target_end_idx + 1]

    # exception handling
    for _ in range(10):
        try:
            assert len(trends) == len(df), "Unequal Length! len(trends): %d, len(df): %d" % (len(trends), len(df))
        except AssertionError:
            print("Error! Re-obtaining Stock Data")
            prices = d.DataReader(code, 'yahoo', start_date, end_date_plus_one)['Adj Close'].values
            trends = []
            for idx in range(1, len(prices)):
                    trends.append(1) if prices[idx] > prices[idx - 1] else trends.append(0)
        else:
            break
    
    df['trend'] = trends
    if code == 'AAPL':
        df = df[df['tweets'] != 0]

    return df


def cross_val_for_svm(n_folds, X, y, clf):
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(acc)
    return scores


def cross_val_for_rnn(n_folds, X, y, params, n_step, n_in):
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = RNNClassifier(**params)
        clf.fit(X_train.reshape(-1, n_step, n_in), y_train, n_epoch=1000, n_batch=5)
        y_pred = clf.predict(X_test.reshape(-1, n_step, n_in))
        acc = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(acc)
        tf.reset_default_graph()
    return scores


def cross_val_for_mlp(n_folds, X, y, params):
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = MLPClassifier(**params)
        clf.fit(X_train, y_train, n_epoch=10000, n_batch=5)
        y_pred = clf.predict(X_test)
        acc = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(acc)
        tf.reset_default_graph()
    return scores
