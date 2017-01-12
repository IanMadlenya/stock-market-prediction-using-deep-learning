import os, sys
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as d
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
sys.path.append(os.path.dirname(os.getcwd()))
from twitter_data import load_sen_df_from_local
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
    for idx in range(len(prices)):
        if idx < 1:
            continue
        else:
            trends.append(1) if prices[idx] > prices[idx - 1] else trends.append(0)

    df = load_sen_df_from_local()
    # get data from 2016-08-01 to end_date
    df = df[:df[df['date'] == end_date_plus_one].index[0]]
    # select company (stock code of S&P 500 is different from others)
    if code != '^GSPC':
        df = df[df['company'] == '$' + code]
    else:
        df = df[df['company'] == '$SPX']
    # check length
    assert len(trends) == len(df), "Unequal length"
    # add stock trends into df
    df['trend'] = trends
    # we have bugs in Apple, remove them
    if code == 'AAPL':
        df = df[df['tweets'] != 0]

    return df


def cross_validation(n_folds, X, y, clf, n_step, n_in):
    kf = KFold(n_splits=n_folds, shuffle=False)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train.reshape(-1, n_step, n_in), y_train, batch_size=16, n_epoch=3000)
        y_test_pred = clf.predict(X_test.reshape(-1, n_step, n_in))
        test_acc = clf.get_accuracy(y_test_pred, y_test)
        scores.append(test_acc)
    return scores


code = '^GSPC'
n_folds = 3
n_hidden_units = 16
learn_rate = 1e-3
df = get_trend_df(code, end_date='2016-12-01')

features = ['tweets', 'liked']
with tf.variable_scope('rnn1'):
    clf = RNNClassifier(len(features), 1, n_hidden_units, 2, learn_rate)
X = df[features].values
y = to_one_hot(df['trend'].values)
scores1 = cross_validation(n_folds, X, y, clf, n_step=len(features), n_in=1)

features = ['pos_score', 'neg_score', 'com_score']
with tf.variable_scope('rnn2'):
    clf = RNNClassifier(len(features), 1, n_hidden_units, 2, learn_rate)
X = df[features].values
y = to_one_hot(df['trend'].values)
scores2 = cross_validation(n_folds, X, y, clf, n_step=len(features), n_in=1)

features = ['tweets', 'liked', 'pos_score', 'neg_score', 'com_score']
with tf.variable_scope('rnn3'):
    clf = RNNClassifier(len(features), 1, n_hidden_units, 2, learn_rate)
X = scale(df[features].values)
y = to_one_hot(df['trend'].values)
scores3 = cross_validation(n_folds, X, y, clf, n_step=len(features), n_in=1)

print("%s" % code)
print("%d folds accuracy: %.2f%% based on volume" % (n_folds, sum(scores1)/len(scores1)*100))
print("%d folds accuracy: %.2f%% based on sentiment" % (n_folds, sum(scores2)/len(scores2)*100))
print("%d folds accuracy: %.2f%% based on all features" % (n_folds, sum(scores3)/len(scores3)*100))
