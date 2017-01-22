import os, sys
import numpy as np
import datetime
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
sys.path.append(os.path.dirname(os.getcwd()))
from data.past_price import Generator


def cross_validation(n_folds, X, y, clf):
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(accuracy)
    return scores


if __name__ == '__main__':
    code = 'JPM'
    n_past = 30
    t_start = str(datetime.datetime.strptime('2016-08-01', "%Y-%m-%d") - datetime.timedelta(days=n_past))[:10]
    t_end = '2017-01-11'
    dc = Generator(code, t_start, t_end, n_past_val=n_past, n_look_forward=1)
    X, y = dc.get_tensor()
    clf = SVC(kernel='rbf')
    scores = cross_validation(3, np.array(X), np.array(y), clf)
    print(code, ['{:.3f}'.format(score) for score in scores], '{:.3f}'.format(sum(scores)/len(scores)))
