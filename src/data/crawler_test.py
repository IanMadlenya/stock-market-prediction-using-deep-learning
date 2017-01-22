from crawler import crawl


if __name__ == '__main__':
    dates = ['2017-01-23']
    tags = ['$SPX', '$XLK', '$XLF', '$AAPL', '$GOOG', '$GS', '$JPM']
    for date in dates:
        crawl(date, tags)
