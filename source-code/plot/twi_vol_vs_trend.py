import os, sys
import pandas_datareader.data as d
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.getcwd()))
from twitter_data import get_main_df


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
    assert len(trends) == len(df), "Unequal length"
    df['trend'] = trends
    if code == 'AAPL':  # we have missing data in Apple
        df = df[df['tweets'] != 0]
    return df


def render(code):
    df = get_trend_df(code, files[-1][:10])
    plt.scatter(df[df['trend'] == 1]['tweets'].values, df[df['trend'] == 1]['liked'].values, label='Rise',
                color='blue', alpha=0.5)
    plt.scatter(df[df['trend'] == 0]['tweets'].values, df[df['trend'] == 0]['liked'].values, label='Not Rise',
                color='red', alpha=0.5)
    plt.legend(frameon=True)
    plt.title(code)
    plt.xlabel('posting volume per day')
    plt.ylabel('endorsing volume per day')


files = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'twitter_data', 'parts'))
codes = ['^GSPC', 'XLK', 'XLF', 'AAPL', 'GOOG', 'GS', 'JPM']
sns.set(context='poster', style='whitegrid', font_scale=.7)
for scr_pos, idx in zip([1, 2, 3, 5, 7, 8, 9], [1, 2, 3, 0, 4, 5, 6]):
    plt.subplot(3, 3, scr_pos)
    render(codes[idx])
plt.tight_layout()
plt.savefig('savefig/twi_vol_vs_trend')
plt.show()
