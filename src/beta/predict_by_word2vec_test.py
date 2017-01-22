import os, sys
import gensim
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as d
from sklearn.model_selection import KFold
sys.path.append(os.path.dirname(os.getcwd()))
from tests.utils import to_one_hot, cross_val_for_mlp


def cross_val(n_folds, X, y, clf):
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


if __name__ == '__main__':
    code = '^GSPC'
    end_date = '2016-12-21'
    model = gensim.models.Word2Vec.load_word2vec_format('D:/word2vec/GoogleNews-vectors-negative300.bin',
                                                        binary=True)

    # load all csvs into one dataframe
    folder_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'parts')
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file[11:14] == 'all']
    df_list = [pd.read_csv(file) for file in files]
    df = pd.concat(df_list, ignore_index=True)

    # select rows before end_date
    df = df[df['company'] == '$'+code] if code != '^GSPC' else df[df['company'] == '$SPX']
    df = df[df['language'] == 'en']
    end_index = df[df['date'] == end_date].index[-1]
    df = df.iloc[: df.index.get_loc(end_index)+1]

    # get X matrix
    X = []
    for date in df['date'].unique():
        sentence_vecs = []
        for key, sentence in enumerate(df[df['date'] == date].text.values):
            word_vecs = []
            for word in sentence.split():
                try:
                    word_vec = model[word]
                except:
                    pass
                else:
                    word_vecs.append(word_vec)
            if len(word_vecs) != 0:
                sentence_vecs.append(sum(word_vecs) / len(word_vecs))
        X.append(sum(sentence_vecs)/len(sentence_vecs))
        print("{} {} finished".format(code, date))
    X = np.vstack(X)

    # get y matrix
    start_date = '2016-08-01'
    end_date_plus_one = str(datetime.datetime.strptime(end_date, "%Y-%m-%d") + pd.tseries.offsets.BDay(1))[:10]
    prices = d.DataReader(code, 'yahoo', start_date, end_date_plus_one)['Adj Close'].values
    y = []
    for idx in range(len(prices)):
        if idx < 1:
            continue
        else:
            y.append(1) if prices[idx] > prices[idx - 1] else y.append(0)

    params = {'n_in':300, 'hidden_units':[10, 20, 10], 'n_out':2}
    scores = cross_val_for_mlp(3, X, to_one_hot(np.array(y)), params)
    print(code, ['{:.3f}'.format(score) for score in scores], '{:.3f}'.format(sum(scores)/len(scores)))
