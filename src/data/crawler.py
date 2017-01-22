import twitter
import csv
import pandas_datareader.data as d
import datetime
from api.get_access import get_consumer_key, get_consumer_secret, get_access_token_key, get_access_token_secret


def get_tweets(api, tag, date):
    # initialization
    next_d = str(datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1))[:10]
    tweets_data, cycle_count, liked_count = [], 1, 0

    results = api.GetSearch(tag, count=100, until=next_d)

    # deals with tweets > 100
    if int(results[-1].created_at[8:10]) == int(date[-2:]):
        # deals with the first cycle
        flag_id = results[-1].id  # update flag id for next cycle
        tweets_num = len(results)
        for result in results:
            liked_count += result.favorite_count
            tweets_data.append(
                {'company': tag, 'date': date, 'time': result.created_at[:19], 'lang': result.lang,
                 'liked': result.favorite_count, 'retweet': result.retweet_count, 'id': result.id,
                 'text': result.text})
        print("%s %s Cycle %d <%s>" % (tag, date, cycle_count, results[-1].created_at))
        # deals with all the remaining cycles except final cycle
        while True:
            try:
                results = api.GetSearch(tag, count=100, max_id=flag_id)
            except Exception as error:
                print("Error: %s" % error)
                continue
            if results[-1].id == flag_id:
                print("LOCKED")
                return 0, tweets_data, 0
            if int(results[-1].created_at[8:10]) != int(date[-2:]):
                break
            cycle_count += 1
            flag_id = results[-1].id  # update flag id for next cycle
            tweets_num += (len(results) - 1)
            for result in results[1:]:
                liked_count += result.favorite_count
                tweets_data.append(
                    {'company': tag, 'date': date, 'time': result.created_at[:19], 'lang': result.lang,
                     'liked': result.favorite_count, 'retweet': result.retweet_count, 'id': result.id,
                     'text': result.text})
            print("%s %s Cycle %d <%s>" % (tag, date, cycle_count, results[-1].created_at))
        # deals with the final cycle
        for idx, result in enumerate(results[1:]):
            if int(result.created_at[8:10]) != int(date[-2:]):
                tweets_num += idx
                print("%s %s Final Cycle <%s>" % (tag, date, result.created_at))
                return tweets_num, tweets_data, liked_count
            liked_count += result.favorite_count
            tweets_data.append(
                {'company': tag, 'date': date, 'time': result.created_at[:19], 'lang': result.lang,
                 'liked': result.favorite_count, 'retweet': result.retweet_count, 'id': result.id,
                 'text': result.text})
    # deals with tweets < 100
    else:
        for idx, result in enumerate(results):
            if int(result.created_at[8:10]) != int(date[-2:]) and idx > 5:
                tweets_num = idx
                print("%s %s Final Cycle <%s>" % (tag, date, result.created_at))
                return tweets_num, tweets_data, liked_count
            liked_count += result.favorite_count
            tweets_data.append(
                {'company': tag, 'date': date, 'time': result.created_at[:19], 'lang': result.lang,
                 'liked': result.favorite_count, 'retweet': result.retweet_count, 'id': result.id,
                 'text': result.text})


def crawl(date, tags):
    api = twitter.Api(get_consumer_key(), get_consumer_secret(), get_access_token_key(), get_access_token_secret())
    stats, tweets = [], []
    for tag in tags:
        tweets_num, tweets_data, liked_count = get_tweets(api, tag, date)
        stock_df = d.DataReader(tag[1:], 'yahoo', date, date) if tag != '$SPX' else d.DataReader('^GSPC', 'yahoo',
            date, date)
        op, cp, volume = stock_df.loc[date, 'Open'], stock_df.loc[date, 'Adj Close'], stock_df.loc[date, 'Volume']
        stats.append(
            {'company': tag, 'date': date, 'tweets': tweets_num, 'liked': liked_count, 'open': op, 'close': cp,
             'volume': volume})
        tweets.append(tweets_data)

    with open('parts/' + date + '-main.csv', 'w', newline='') as csv_file:
        lines_writer = csv.writer(csv_file, delimiter=',')
        lines_writer.writerow(['company', 'date', 'tweets', 'liked', 'open', 'adj_close', 'volume'])
        for stat in stats:
            lines_writer.writerow(
                [stat['company'], stat['date'], stat['tweets'], stat['liked'], stat['open'], stat['close'],
                 stat['volume']])

    with open('parts/' + date + '-all.csv', 'w', newline='', encoding='utf-8') as csv_file:
        lines_writer = csv.writer(csv_file, delimiter=',')
        lines_writer.writerow(['company', 'date', 'time', 'language', 'liked', 'retweet', 'id', 'text'])
        for tweets_data in tweets:
            for tweet in tweets_data:
                lines_writer.writerow(
                    [tweet['company'], tweet['date'], tweet['time'], tweet['lang'], tweet['liked'],
                     tweet['retweet'], tweet['id'], tweet['text']])
