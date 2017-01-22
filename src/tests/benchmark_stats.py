import pandas as pd 
from utils import get_trend_df
from collections import Counter


df = pd.read_csv('./benchmark.csv')
result_dfs = []
for stock in ['^GSPC', 'XLK', 'XLF', 'AAPL', 'GOOG', 'GS', 'JPM']:
    stock_df = df[df['Code'] == stock]
    max_vals = stock_df['Accuracy'].values.max()
    result_df = stock_df[stock_df['Accuracy'] == max_vals]
    result_dfs.append(result_df)
print(pd.concat(result_dfs), '\n')

for stock in ['^GSPC', 'XLK', 'XLF', 'AAPL', 'GOOG', 'GS', 'JPM']:
    stock_df = get_trend_df(stock, end_date='2017-01-11')
    counts = Counter(stock_df['trend'].values)
    print('{}: {}%'.format(stock, int(counts[1]/(counts[1]+counts[0])*100)))
