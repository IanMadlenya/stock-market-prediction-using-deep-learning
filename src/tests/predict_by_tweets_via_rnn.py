import os, sys
import tensorflow as tf
from sklearn.preprocessing import scale
sys.path.append(os.path.dirname(os.getcwd()))
from utils import get_trend_df, cross_val_for_rnn


if __name__ == '__main__':
    code = 'GOOG'
    end_t = '2016-12-22'
    n_folds = 3
    n_hidden_units = 16
    df = get_trend_df(code, end_t)

    features = ['tweets', 'liked']
    params = {'n_step':len(features), 'n_in':1, 'n_hidden_units':n_hidden_units, 'n_out':2}
    X = df[features].values
    y = df['trend'].values
    scores1 = cross_val_for_rnn(n_folds, X, y, params, n_step=len(features), n_in=1)

    features = ['pos_score', 'neg_score', 'com_score']
    params = {'n_step':len(features), 'n_in':1, 'n_hidden_units':n_hidden_units, 'n_out':2}
    X = df[features].values
    y = df['trend'].values
    scores2 = cross_val_for_rnn(n_folds, X, y, params, n_step=len(features), n_in=1)

    features = ['tweets', 'liked', 'pos_score', 'neg_score', 'com_score']
    params = {'n_step':len(features), 'n_in':1, 'n_hidden_units':n_hidden_units, 'n_out':2}
    X = scale(df[features].values)
    y = df['trend'].values
    scores3 = cross_val_for_rnn(n_folds, X, y, params, n_step=len(features), n_in=1)

    print("{} {}".format(code, end_t))
    print("{} folds accuracy: {:.2f}% based on volume".format(n_folds, sum(scores1)/len(scores1)*100))
    print("{} folds accuracy: {:.2f}% based on sentiment".format(n_folds, sum(scores2)/len(scores2)*100))
    print("{} folds accuracy: {:.2f}% based on all features".format(n_folds, sum(scores3)/len(scores3)*100))
