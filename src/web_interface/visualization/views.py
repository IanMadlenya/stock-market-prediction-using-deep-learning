from django.shortcuts import render
from django.http import HttpResponse
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import os, sys
import pandas as pd
import pandas_datareader.data as d
import datetime
sys.path.append(os.path.dirname(os.getcwd()))
from data.tweet import load_sen_df_from_local


# Create your views here.
def home(request):
    return render(request, 'visualization/index.html')


def plot(request):
    img_path = 'visualization/static/visualization/img/buffer.png'

    if request.method == 'POST':
        company = request.POST['company']
        end_date = request.POST['end_date']

    name_dict = {'S&P 500': '$SPX', 'Technology SPDR': '$XLK', 'Finance SPDR': '$XLF', 'Apple': '$AAPL',
                 'Google': '$GOOG', 'Goldman Sachs': '$GS', 'JP Morgan': '$JPM'}
    df = load_sen_df_from_local()
    df = df[df['company'] == name_dict[company]]
    target_end_idx = df.index.get_loc(df[df['date'] == end_date].index[0])
    df = df.iloc[: target_end_idx + 1]

    # get stock trends from 2016-08-01 to end_date_plus_one
    code_dict = {'S&P 500': '^GSPC', 'Technology SPDR': 'XLK', 'Finance SPDR': 'XLF', 'Apple': 'AAPL',
                 'Google': 'GOOG', 'Goldman Sachs': 'GS', 'JP Morgan': 'JPM'}
    start_date = '2016-08-01'
    end_date_plus_one = str(datetime.datetime.strptime(end_date, "%Y-%m-%d") + pd.tseries.offsets.BDay(1))[:10]
    prices = d.DataReader(code_dict[company], 'yahoo', start_date, end_date_plus_one)['Adj Close'].values
    trends = []
    for idx in range(len(prices)):
        if idx < 1:
            continue
        else:
            trends.append(1) if prices[idx] > prices[idx - 1] else trends.append(0)
    
    # exception handling
    for _ in range(10):
        try:
            assert len(trends) == len(df), "Unequal Length! len(trends): %d, len(df): %d" % (len(trends), len(df))
        except AssertionError:
            print("Error! Re-obtaining Stock Data")
            prices = d.DataReader(code_dict[company], 'yahoo', start_date, end_date_plus_one)['Adj Close'].values
            trends = []
            for idx in range(len(prices)):
                if idx < 1:
                    continue
                else:
                    trends.append(1) if prices[idx] > prices[idx - 1] else trends.append(0)
        else:
            break
    
    df['trend'] = trends
    if company == 'Apple':
        df = df[df['tweets'] != 0]
    
    sns.set(style='whitegrid', font_scale=0.8)

    plt.subplot(2, 2, 1)
    plt.scatter(df['tweets'], df['adj_close'], color='cornflowerblue')
    plt.xlabel('Tweet Volume')
    plt.ylabel('Price')

    plt.subplot(2, 2, 2)
    plt.scatter(df['neg_score'], df['adj_close'], color='indianred')
    plt.xlabel('Average Negative Sentiment Score')
    plt.ylabel('Price')

    plt.subplot(2, 2, 3)
    plt.scatter(df[df['trend'] == 1]['tweets'].values, df[df['trend'] == 1]['liked'].values, label='rise',
                color='cornflowerblue')
    plt.scatter(df[df['trend'] == 0]['tweets'].values, df[df['trend'] == 0]['liked'].values, label='fall',
                color='indianred')
    plt.legend(frameon=True)
    plt.xlabel('Tweet Volume')
    plt.ylabel('Endorsed Volume')

    plt.subplot(2, 2, 4)
    plt.scatter(df[df['trend'] == 1]['pos_score'].values, df[df['trend'] == 1]['neg_score'].values, label='rise',
                color='cornflowerblue')
    plt.scatter(df[df['trend'] == 0]['pos_score'].values, df[df['trend'] == 0]['neg_score'].values, label='fall',
                color='indianred')
    plt.legend(frameon=True)
    plt.xlabel('Average Positive Sentiment Score')
    plt.ylabel('Average Negative Sentiment Score')

    plt.suptitle(company, y=1.0, fontsize=10)
    plt.tight_layout()

    plt.savefig(img_path)
    plt.close()
    with open(img_path, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    return HttpResponse(encoded_string, content_type="image/png")
