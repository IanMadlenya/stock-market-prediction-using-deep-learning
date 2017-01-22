import os, sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
sys.path.append(os.path.dirname(os.getcwd()))
from data.image import Generator


def cross_validation(n_folds, X, y, clf):
    kf = KFold(n_splits=n_folds, shuffle=False)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(accuracy)
    return scores


if __name__ == '__main__':
    code, t_start, t_end = 'AAPL', '2006-01-01', '2016-01-01'
    n_look_forward = 14
    dc = Generator(code, t_start, t_end, 30, n_look_forward)
    X, y = dc.get_tensor()
    clf = SVC(kernel='rbf')
    scores = cross_validation(3, np.array(X), np.array(y), clf)
    print(code, n_look_forward, ['{:.3f}'.format(score) for score in scores], '{:.3f}'.format(sum(scores)/len(scores)))
