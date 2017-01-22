import os, sys
import pandas as pd
import datetime
import pandas_datareader.data as d
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.getcwd()))
from data.tweet import get_main_df


def get_trend_df(code, end_date, n_look_back=1):
    # get stock trends from 2016-08-01
    prices = d.DataReader(code, 'yahoo', '2016-07-29', end_date)['Adj Close'].values
    trends = []
    for idx in range(len(prices)):
        if idx < n_look_back:
            continue
        else:
            trends.append(1) if prices[idx] > prices[idx - n_look_back] else trends.append(0)

    # we want tweets, endorses and trends in a df
    df = get_main_df(drop_empty_rows=False)
    df = df[df['company'] == '$' + code] if code != '^GSPC' else df[df['company'] == '$SPX']
    target_end_idx = df.index.get_loc(df[df['date'] == end_date].index[0])
    df = df.iloc[: target_end_idx + 1]
    
    # exception handling
    for _ in range(10):
        try:
            assert len(trends) == len(df), "Unequal Length! len(trends): %d, len(df): %d" % (len(trends), len(df))
        except AssertionError:
            print("Error! Re-obtaining Stock Data")
            prices = d.DataReader(code, 'yahoo', start_date, end_date_plus_one)['Adj Close'].values
            trends = []
            for idx in range(len(prices)):
                if idx < 1:
                    continue
                else:
                    trends.append(1) if prices[idx] > prices[idx - 1] else trends.append(0)
        else:
            break

    df['trend'] = trends
    if code == 'AAPL':  # we have missing data in Apple
        df = df[df['tweets'] != 0]
    return df


def render(code):
    df = get_trend_df(code, end_date='2017-01-10')
    plt.scatter(df[df['trend'] == 1]['tweets'].values, df[df['trend'] == 1]['liked'].values, label='rise',
                color='cornflowerblue')
    plt.scatter(df[df['trend'] == 0]['tweets'].values, df[df['trend'] == 0]['liked'].values, label='fall',
                color='indianred')
    plt.legend(frameon=True, bbox_to_anchor=[1.05, 1.05])
    plt.title(names[code])
    plt.xlabel('daily posting volume')
    plt.ylabel('daily endorsing volume')


if __name__ == '__main__':
    codes = ['^GSPC', 'XLK', 'XLF', 'AAPL', 'GOOG', 'GS', 'JPM']
    names = {'^GSPC':'S&P 500', 'XLK':'Technology Sector', 'XLF':'Finance Sector', 'AAPL': 'Apple', 'GOOG':'Google',
            'GS':'Goldman Sachs', 'JPM':'JP Morgan'}
    sns.set(context='poster', style='whitegrid', font_scale=.7)
    for scr_pos, idx in zip([1, 2, 3, 5, 7, 8, 9], [1, 2, 3, 0, 4, 5, 6]):
        plt.subplot(3, 3, scr_pos)
        render(codes[idx])
    plt.tight_layout()
    plt.savefig('savefig/twi_vol_vs_trend')
    plt.show()
