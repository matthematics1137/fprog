#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 03:21:52 2023

@author: somebody
"""

import requests
import subprocess
import time
from bs4 import BeautifulSoup
from dateutil.parser import parse
import weaviate
from requests.exceptions import RequestException
import re
import pandas as pd
from datetime import datetime
import os
import hashlib


class NitterScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.client = weaviate.Client("http://localhost:8080")

    def connect_vpn(self):
        try:
            result = subprocess.run(["protonvpn-cli", "c", "-r"], capture_output=True, text=True, timeout=60)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f"Error while connecting to VPN: {e}")
        except subprocess.TimeoutExpired:
            print("Timeout expired while connecting to VPN")
        else:
            print(result.stdout)


    def disconnect_vpn(self):
        try:
            result = subprocess.run(["protonvpn-cli", "d"], capture_output=True, text=True, timeout=60)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f"Error while disconnecting from VPN: {e}")
        except subprocess.TimeoutExpired:
            print("Timeout expired while disconnecting from VPN")
        else:
            print(result.stdout)
            
    def safe_int(self, value, default=0):
        try:
            return int(value)
        except ValueError:
            return default
        
    def clean_string(self, text):
        print('removing non ascii from:', text)
        # Remove any non-ASCII characters
        cleaned_text = re.sub(r'[^\x00-\x7F]+', ',', text)
        print('removed non ascii from', cleaned_text)
        

        return cleaned_text
    
    def create_tweet_hash(self, username, timestamp, content):
        # Combine the username, date, and content into a single string
        tweet_str = f"{username}{timestamp}{content}"
        
        # Create a SHA256 hash of the string
        hash_object = hashlib.sha256(tweet_str.encode())
        hash_str = hash_object.hexdigest()
        
        return hash_str

            
    def fetch_tweet_data(self, tweet):
        timestamp_element = tweet.find("span", class_="tweet-date")
        title = timestamp_element.a['title']
        title = title.replace(" UTC", "")
        title = self.clean_string(title)
        utc_timestamp = parse(title).timestamp()
        tweet_stats = tweet.find_all("span", class_="tweet-stat")
        '''hashtags = [hashtag.text for hashtag in tweet.find_all("a", class_="hashtag")]
        mentions = [mention.text for mention in tweet.find_all("a", class_="mention")]
        links = [link['href'] for link in tweet.find_all("a", class_="link-overlay")]
        '''
        replies = self.safe_int(tweet_stats[0].text.strip().replace(',', '')) if len(tweet_stats) > 0 else 0
        retweets = self.safe_int(tweet_stats[1].text.strip().replace(',', '')) if len(tweet_stats) > 1 else 0
        quotes = self.safe_int(tweet_stats[2].text.strip().replace(',', '')) if len(tweet_stats) > 2 else 0
        likes = self.safe_int(tweet_stats[3].text.strip().replace(',', '')) if len(tweet_stats) > 3 else 0
        username = tweet.find("a", class_="username").text.strip()
        content = tweet.find("div", class_="tweet-content").text.strip()
        hash_id = self.create_tweet_hash(username, utc_timestamp, content)
    
        return {
            "id": hash_id,
            "timestamp": utc_timestamp,
            "retweeted_by": tweet.find("div", class_="retweet-header").text.strip()[:-10] if tweet.find("div", class_="retweet-header") else None,
            "username": username,
            "fullname": tweet.find("a", class_="fullname").text.strip(),
            "content": content,
            "replies": replies,
            "retweets": retweets,
            "quotes": quotes,
            "likes": likes,
        }






    def scrape_tweets(self, profile_name, tweet_limit=None):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        }
    
        retry_count = 3
        delay_seconds = 10
    
        tweet_count = 0
        tweets_already_stored = 0
        stop_loop = False
    
        url = f"https://nitter.net/{profile_name}"
    
        while not stop_loop:
            for attempt in range(retry_count):
                try:
                    response = requests.get(url, headers=headers)
    
                    if response.status_code == 200:
                        break
                    else:
                        print(f"Request failed with status {response.status_code}. Retrying...")
                except RequestException as e:
                    print(f"Request failed with error {e}. Retrying...")
    
                if attempt < retry_count - 1:
                    time.sleep(delay_seconds)
    
                if response.status_code != 200:
                    print("Failed to fetch tweets:", response.status_code)
                    return
    
            soup = BeautifulSoup(response.content, "html.parser")
            tweets = soup.find_all("div", class_="timeline-item")
            if not tweets:
                break

    
            for tweet in tweets:
                if tweet_limit and tweet_count >= tweet_limit:
                    stop_loop = True
                    break
    
    
                content = tweet.find("div", class_="tweet-content")
                if content:
                    tweet_data = self.fetch_tweet_data(tweet)
                    tweet_id = tweet_data["id"]
    
                    if self.tweet_exists(tweet_id):
                        tweets_already_stored += 1
                        if tweets_already_stored >= 10:
                            stop_loop = True
                            break
                    else:
                        self.store_tweet(tweet_data)
                        tweet_count += 1
                        tweets_already_stored = 0
    
                    print(f"Processed tweet {tweet_count}: {tweet_id}")
    
                if tweet_limit and tweet_count >= tweet_limit:
                    break
            
            
            show_more_divs = soup.find_all('div', class_='show-more')
            
            
            
            if len(show_more_divs) > 1:
                load_more_link = show_more_divs[1].find('a', href=True)
                
            else:
                load_more_link = show_more_divs[0].find('a', href=True)
                
            if soup.find('h2', class_='timeline-end'):
                print('no more pages')
                break
            
            else:
                # Extract the "cursor" value from the link
                cursor = load_more_link["href"].split("=")[1]
                
                # Use this "cursor" value to form the URL for the next page
                url = f"https://nitter.net/{profile_name}/?cursor={cursor}"



            time.sleep(1)  # Add a delay between requests to avoid rate limiting
        self.update_log(profile_name, tweet_count)
        print("Scraping completed.")


 


           
    def tweet_exists(self, tweet_id):
        weaviate_url = f"http://localhost:8080/v1/objects?q=tweetId:{tweet_id}&class=Tweet"
        response = requests.get(weaviate_url)
    
        if response.status_code == 200:
            data = response.json()
            return "data" in data and len(data["data"]) > 0
        else:
            print(f"Error checking for tweet {tweet_id}: {response.status_code}, {response.text}")
            return False


                    
                    
    def store_tweet(self, tweet):
        tweet_id = tweet['id']
        if not self.tweet_exists(tweet_id):
    
            # Prepare the tweet object according to your schema
            tweet_object = {
                "class": "Tweet",
                "text": tweet["content"],
                "timestamp": tweet["timestamp"],
                "username": tweet["username"],
                "retweets": tweet["retweets"],
                "favorites": tweet["likes"],
                "tweetId": tweet["id"],  # Store the hash id
            }
    
            # Send the tweet object to Weaviate using the client's create() method
            response = self.client.data_object.create(
                tweet_object, class_name="Tweet"
            )
            
            # Check the response for success
            if response is not None:
                print(f"Tweet {tweet['id']} stored in Weaviate with hash id {id}.")
            else:
                print(f"Error storing tweet {tweet['id']}.")
                
                
    def update_log(self, username, num_tweets):
        filename = 'scrape_log.csv'
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_csv(filename) if os.path.isfile(filename) else pd.DataFrame(columns=['Username', 'Last Scraped Date', 'Num Tweets Scraped'])
    
        if username in df['Username'].values:
            df.loc[df['Username'] == username, 'Last Scraped Date'] = now
            df.loc[df['Username'] == username, 'Num Tweets Scraped'] += num_tweets
        else:
            new_row = pd.DataFrame({'Username': [username], 'Last Scraped Date': [now], 'Num Tweets Scraped': [num_tweets]})
            df = pd.concat([df, new_row]).reset_index(drop=True)
    
        df.to_csv(filename, index=False)

    
        df.to_csv(filename, index=False)


    def main(self, usernames, tweet_limit=None):
        for username in usernames:
            print(f"Connecting to VPN for {username}...")
            self.connect_vpn()
    
            print("Waiting for 10 seconds to ensure connection...")
            time.sleep(10)
    
            print(f"Scraping tweets for {username}...")
            self.scrape_tweets(username, tweet_limit=tweet_limit)
    
            print(f"Disconnecting from VPN for {username}...")
            self.disconnect_vpn()
    
            # If you want to add a delay between disconnecting and connecting to a new VPN, you can use:
            print("Waiting 10 seconds before new connection...") 
            time.sleep(10)



usernames_list = ['pmarca']

if __name__ == "__main__":
    scraper = NitterScraper("https://nitter.net")
    scraper.main(usernames_list)
