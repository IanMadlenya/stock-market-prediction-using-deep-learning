import os, sys
import numpy as np
from sklearn.model_selection import KFold
sys.path.append(os.path.dirname(os.getcwd()))
from image_data import DataCenter
from neural_network import CNNClassifier


def cross_validation(n_folds, X, y, clf):
    kf = KFold(n_splits=n_folds, shuffle=False)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train, n_epoch=50)
        y_pred = clf.predict(X_test)
        accuracy = clf.get_accuracy(y_pred, y_test)
        scores.append(accuracy)
    return scores


code, t_start, t_end = 'AAPL', '2006-01-01', '2016-01-01'
dc = DataCenter(code, t_start, t_end, n_past_val=30, n_look_forward=18, is_one_hot=True)
X, y = dc.get_tensor()
clf = CNNClassifier(56*56, 2)
scores = cross_validation(3, np.array(X), np.array(y), clf)
print(code, ['%.3f' % score for score in scores], '%.3f' % (sum(scores)/len(scores)))
