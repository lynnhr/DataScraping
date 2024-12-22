import pandas as pd
from ntscraper import Nitter

scraper = Nitter()
def get_tweets(name,modes,no):
    tweets = scraper.get_tweets(name,mode=modes,number=no)
    final_tweets=[]
    for tweet in tweets['tweets']:
        data=[tweet['link'],tweet['text'],tweet['date'],tweet['stats']['likes'],tweet['stats']['comments'],tweet['stats']['retweets']]
        final_tweets.append(data)
    data=pd.DataFrame(final_tweets,columns=['link','text','date','n of likes','n of comments','n of retweets'])
    return data

data = get_tweets('samsung','hashtag',100)
data.to_csv('sams.csv')
