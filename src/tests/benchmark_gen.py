import os, sys
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.svm import SVC
sys.path.append(os.path.dirname(os.getcwd()))
from utils import get_trend_df, cross_val_for_rnn, cross_val_for_mlp, cross_val_for_svm


def register(code, feature_name, scores, model_name):
    global_stock_name_list.append(code)
    global_feature_name_list.append(feature_name)
    global_accuracy_list.append('%.2f'%(sum(scores)/len(scores)))
    global_model_name_list.append(model_name)


def run_test(code, model_name):
    df = get_trend_df(code, end_date=end_t)
    local_feature_list = [['tweets', 'liked'], ['pos_score', 'neg_score', 'com_score'],
                    ['tweets', 'liked', 'pos_score', 'neg_score', 'com_score']]
    local_feature_name_list = ['volume', 'sentiment', 'compound']
    for features, feature_name in zip(local_feature_list, local_feature_name_list):
        if model_name == 'RNN':
            params = {'n_step':len(features), 'n_in':1, 'n_hidden_units':n_hidden_units, 'n_out':2}
            if feature_name == 'compound':
                scores = cross_val_for_rnn(n_folds, scale(df[features].values), df['trend'].values, params,
                                           n_step=len(features), n_in=1)
            else:
                scores = cross_val_for_rnn(n_folds, df[features].values, df['trend'].values, params,
                                           n_step=len(features), n_in=1)
            register(code, feature_name, scores, model_name)
        if model_name == 'SVM':
            if feature_name == 'compound':
                scores = cross_val_for_svm(n_folds, scale(df[features].values), df['trend'].values, svc)
            else:
                scores = cross_val_for_svm(n_folds, df[features].values, df['trend'].values, svc)
            register(code, feature_name, scores, model_name)
        if model_name == 'MLP':
            params = {'n_in':len(features), 'hidden_units':hidden_units, 'n_out':2}
            if feature_name == 'compound':
                scores = cross_val_for_mlp(n_folds, scale(df[features].values), df['trend'].values, params)
            else:
                scores = cross_val_for_mlp(n_folds, df[features].values, df['trend'].values, params)
            register(code, feature_name, scores, model_name)


if __name__ == '__main__':
    end_t = '2017-01-18'
    code_list = ['^GSPC', 'XLK', 'XLF', 'AAPL', 'GOOG', 'GS', 'JPM']
    n_folds = 3
    global_stock_name_list = []
    global_feature_name_list = []
    global_accuracy_list = []
    global_model_name_list = []

    svc = SVC(kernel='rbf')
    svm_step = 0
    for code in code_list:
        run_test(code, 'SVM')
        svm_step += 1
        print('SVM Step %d' % svm_step)

    hidden_units = [10, 20, 10]
    for code in code_list:
        run_test(code, 'MLP')

    n_hidden_units = 16
    for code in code_list:
        run_test(code, 'RNN')

    pd.DataFrame({'Code': global_stock_name_list,
                  'Feature': global_feature_name_list,
                  'Accuracy': global_accuracy_list,
                  'Algorithm': global_model_name_list}).to_csv('benchmark.csv', index=False)
    print("{} Benchmark Successfully Created".format(end_t))
