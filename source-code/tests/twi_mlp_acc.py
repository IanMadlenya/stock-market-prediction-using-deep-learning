import os, sys
import numpy as np
import pandas as pd
import pandas_datareader.data as d
import datetime
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
sys.path.append(os.path.dirname(os.getcwd()))
from twitter_data import load_sen_df_from_local
from neural_network import MLPClassifier


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


def cross_validation(n_folds, X, y, clf):
    kf = KFold(n_splits=n_folds, shuffle=False)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train, n_epoch=3000, batch_size=16)
        y_pred = clf.predict(X_test)
        acc = clf.get_accuracy(y_pred, y_test)
        scores.append(acc)
    return scores


if __name__ == '__main__':
    code = 'AAPL'
    n_folds = 3
    hidden_units = [10, 20, 10]
    df = get_trend_df(code, end_date='2016-12-01')

    features = ['tweets', 'liked']
    clf = MLPClassifier(len(features), hidden_units, 2)
    X = df[features].values
    y = to_one_hot(df['trend'].values)
    scores1 = cross_validation(n_folds, X, y, clf)

    features = ['pos_score', 'neg_score', 'com_score']
    clf = MLPClassifier(len(features), hidden_units, 2)
    X = df[features].values
    y = to_one_hot(df['trend'].values)
    scores2 = cross_validation(n_folds, X, y, clf)

    features = ['tweets', 'liked', 'pos_score', 'neg_score', 'com_score']
    clf = MLPClassifier(len(features), hidden_units, 2)
    X = scale(df[features].values)
    y = to_one_hot(df['trend'].values)
    scores3 = cross_validation(n_folds, X, y, clf)

    print("%s" % code)
    print("%d folds accuracy: %.2f%% based on volume" % (n_folds, sum(scores1)/len(scores1)*100))
    print("%d folds accuracy: %.2f%% based on sentiment" % (n_folds, sum(scores2)/len(scores2)*100))
    print("%d folds accuracy: %.2f%% based on all features" % (n_folds, sum(scores3)/len(scores3)*100))
