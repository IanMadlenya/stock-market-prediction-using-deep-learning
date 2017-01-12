from crawler import crawl

if __name__ == '__main__':
    dates = ['2016-12-05']
    tags = ['$SPX', '$XLK', '$XLF', '$AAPL', '$GOOG', '$GS', '$JPM']
    for date in dates:
        crawl(date, tags)
