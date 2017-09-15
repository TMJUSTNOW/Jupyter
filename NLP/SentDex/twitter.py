from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

import os
os.chdir(r"D:/My Computer/DATA/SentDex")

#consumer key, consumer secret, access token, access secret.
ckey="gosegjymtSfQxeDKMA8FQlNog"
csecret="v16xZCWqTyrOJsRxiTLs87iEB9mSQ3jJkJleTtNqhcM1K8JrA1"
atoken="4050900493-U79b5ZJJ68Y3WWLst4iIK4DwiyIbThbBfXDI39E"
asecret="8WbXy2Ja87n84lEcgGnAQxGaf5asekVQBzTgA5KyGgK0B"

#from twitterapistuff import *

class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)

        tweet = ascii(all_data["text"])
        #tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)   

        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["altright", "Alt-Right"])