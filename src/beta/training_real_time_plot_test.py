import os, sys
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
sys.path.append(os.path.dirname(os.getcwd()))
from data.past_price import Generator
from neural_network import MLPClassifier


if __name__ == '__main__':
    sns.set(style='whitegrid')
    code, t_start, t_end = 'AAPL', '2006-01-01', '2016-01-01'
    n_in, hidden_units, n_out = 30, [10, 20, 10], 2
    dc = Generator(code, t_start, t_end, n_past_val=30, n_look_forward=24)
    X, y = dc.get_tensor()
    clf = MLPClassifier(n_in, hidden_units, n_out)
    clf.fit_plot(X, y)
