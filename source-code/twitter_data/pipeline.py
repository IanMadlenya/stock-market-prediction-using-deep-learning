import os
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_main_df(drop_empty_rows=True, save_to_local=False):
    base_path = os.path.dirname(os.getcwd())
    all_files = os.listdir(os.path.join(base_path, 'twitter_data', 'parts'))
    files = [os.path.join(base_path, 'twitter_data', 'parts', file) for file in all_files if file[11:15] == 'main']
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    if drop_empty_rows:
        df = df[df['tweets'] != 0]
    if save_to_local:
        df.to_csv(os.path.join(base_path, 'twitter_data', 'files', 'main_matrix.csv'), index=False)
    return df


def get_sen_df(code):
    sia = SentimentIntensityAnalyzer()
    avg_com_scores = []
    avg_pos_scores = []
    avg_neg_scores = []
    dates = []

    files = [os.path.join('parts', file) for file in os.listdir('parts/') if file[11:14] == 'all']
    for file in files:
        df = pd.read_csv(file)
        df = df[df['company'] == code]
        df = df[df['language'] == 'en']
        com_scores = [sia.polarity_scores(sentence)['compound'] for sentence in df['text'].values]
        pos_scores = [sia.polarity_scores(sentence)['pos'] for sentence in df['text'].values]
        neg_scores = [sia.polarity_scores(sentence)['neg'] for sentence in df['text'].values]
        avg_com_scores.append(sum(com_scores)/len(com_scores))
        avg_pos_scores.append(sum(pos_scores)/len(pos_scores))
        avg_neg_scores.append(sum(neg_scores)/len(neg_scores))
        dates.append(df['date'].values[0])
        print(code, file, 'completed')
    
    sen_df = pd.DataFrame({
         'company': [code] * len(avg_com_scores), 'date': dates,
         'com_score': avg_com_scores,
         'pos_score': avg_pos_scores,
         'neg_score': avg_neg_scores})
    return sen_df


def save_sen_df_to_local():
    codes = ['$SPX', '$XLK', '$XLF', '$AAPL', '$GOOG', '$GS', '$JPM']
    sen_dfs = []
    for code in codes:
        sen_dfs.append(get_sen_df(code))
        print("%s completed" % code)
    df = pd.concat(sen_dfs, ignore_index=True)
    main_df = get_main_df(drop_empty_rows=False)
    df = pd.merge(main_df, df, on=['company', 'date'])
    df.to_csv('files/senti_matrix.csv', index=False)


def load_sen_df_from_local():
    base_path = os.path.dirname(os.getcwd())
    sen_df = pd.read_csv(os.path.join(base_path, 'twitter_data', 'files', 'senti_matrix.csv'))
    return sen_df


if __name__ == '__main__':
    save_sen_df_to_local()
