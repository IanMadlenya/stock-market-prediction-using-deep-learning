import os, sys
import numpy as np
import pandas_datareader.data as d
import datetime
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.svm import SVC
sys.path.append(os.path.dirname(os.getcwd()))
from twitter_data import load_sen_df_from_local


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
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(acc)
    return scores


code = '^GSPC'
n_folds = 3
df = get_trend_df(code, end_date='2016-12-01')

clf = SVC(kernel='rbf')
print("%s" % code)

scores = cross_validation(n_folds, df[['tweets', 'liked']].values, df['trend'].values, clf)
print("%d folds accuracy: %.2f%% based on volume" % (n_folds, sum(scores)/len(scores)*100))

scores = cross_validation(n_folds, df[['pos_score', 'neg_score', 'com_score']].values, df['trend'].values, clf)
print("%d folds accuracy: %.2f%% based on sentiment" % (n_folds, sum(scores)/len(scores)*100))

scores = cross_validation(n_folds, scale(df[['tweets', 'liked', 'pos_score', 'neg_score', 'com_score']].values),
                          df['trend'].values, clf)
print("%d folds accuracy: %.2f%% based on all features" % (n_folds, sum(scores)/len(scores)*100))
