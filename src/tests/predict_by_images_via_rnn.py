import os, sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
sys.path.append(os.path.dirname(os.getcwd()))
from data.image import Generator
from neural_network import RNNClassifier


def cross_validation(n_folds, X, y):
    kf = KFold(n_splits=n_folds, shuffle=False)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = RNNClassifier(n_step, n_in, n_hidden, n_out)
        clf.fit(X_train.reshape(-1, n_step, n_in), y_train, n_epoch=50, n_batch=20)
        y_pred = clf.predict(X_test.reshape(-1, n_step, n_in))
        acc = np.equal(y_pred, y_test).astype(float).mean()
        scores.append(acc)
        tf.reset_default_graph()
    return scores


if __name__ == '__main__':
    code, t_start, t_end = 'GS', '2006-01-01', '2016-01-01'
    n_step, n_in, n_hidden, n_out = 56, 56, 16, 2
    dc = Generator(code, t_start, t_end, n_past_val=30, n_look_forward=10)
    X, y = dc.get_tensor()
    scores = cross_validation(3, np.array(X), np.array(y))
    print(code, ['{:.3f}'.format(score) for score in scores], '{:.3f}'.format(sum(scores)/len(scores)))
