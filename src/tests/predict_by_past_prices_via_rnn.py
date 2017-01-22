import os, sys
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.path.dirname(os.getcwd()))
from data.past_price import Generator
from neural_network import RNNClassifier


def cross_validation(n_folds, X, y, clf):
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train.reshape(-1, n_step, n_in), y_train, n_epoch=10000, n_batch=5)
        y_pred = clf.predict(X_test.reshape(-1, n_step, n_in))
        acc = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(acc)
    return scores


if __name__ == '__main__':
    code = 'JPM'
    n_past = 30
    t_start = str(datetime.datetime.strptime('2016-08-01', "%Y-%m-%d") - datetime.timedelta(days=n_past))[:10]
    t_end = '2017-01-11'
    n_step = n_past
    n_in = 1
    n_hidden_units = 16
    dc = Generator(code, t_start, t_end, n_past_val=n_past, n_look_forward=1)
    X, y = dc.get_tensor()
    clf = RNNClassifier(n_step, n_in, n_hidden_units, 2)
    scores = cross_validation(3, np.array(X), np.array(y), clf)
    print(code, ['{:.3f}'.format(score) for score in scores], '{:.3f}'.format(sum(scores)/len(scores)))
