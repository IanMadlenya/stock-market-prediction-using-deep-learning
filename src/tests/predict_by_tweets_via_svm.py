import os, sys
from sklearn.preprocessing import scale
from sklearn.svm import SVC
sys.path.append(os.path.dirname(os.getcwd()))
from utils import get_trend_df, cross_val_for_svm


if __name__ == '__main__':
    code = 'GOOG'
    n_folds = 3
    end_t = '2016-12-22'
    df = get_trend_df(code, end_t)
    clf = SVC(kernel='rbf')

    scores1 = cross_val_for_svm(n_folds, df[['tweets', 'liked']].values, df['trend'].values, clf)
    scores2 = cross_val_for_svm(n_folds, df[['pos_score', 'neg_score', 'com_score']].values, df['trend'].values,
                                clf)
    scores3 = cross_val_for_svm(n_folds, scale(df[['tweets', 'liked', 'pos_score', 'neg_score', 'com_score']].values),
                                df['trend'].values, clf)

    print("{} {}".format(code, end_t))
    print("{} folds accuracy: {:.2f}% based on volume".format(n_folds, sum(scores1)/len(scores1)*100))
    print("{} folds accuracy: {:.2f}% based on sentiment".format(n_folds, sum(scores2)/len(scores2)*100))
    print("{} folds accuracy: {:.2f}% based on all features".format(n_folds, sum(scores3)/len(scores3)*100))
