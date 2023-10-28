#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 23:35:15 2023

@author: somebody
"""

### THIS IS CURRENTLY MAIN REPO FOR CODE ### 

import requests
from bs4 import BeautifulSoup
from googlesearch import search

import os
import re

from newspaper import Article
import pandas as pd



from transformers import pipeline




from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

model_name = 'sshleifer/distilbart-cnn-12-6'
revision = 'a4f8f3e'

tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
summarizer = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)



from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load pre-trained model and tokenizer for sentiment analysis
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_name)

import torch

my_google_api_key = 'key'
my_google_search_engine_id = 'id'

my_News_API_key = 'key2'



import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred
import matplotlib.pyplot as plt

import urllib 


my_fred_key = 'key3'


from urllib.parse import urlparse

import feedparser

import json

import weaviate

weaviate_client = weaviate.Client("http://localhost:8080")

import hashlib


class NewsSentimentPipeline:
    def __init__(self, stock_name, num_articles, num_results,
                 keywords, my_google_api_key,
                 my_google_search_engine_id):
        self.stock_name = stock_name
        self.num_articles = num_articles
        self.num_results = num_results
        self.keywords = keywords
        self.my_google_api_key = my_google_api_key
        self.my_google_search_engine_id = my_google_search_engine_id
        # Initialize sentiment model and tokenizer
        self.sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(self.sentiment_model_name)
        self.sentiment_tokenizer = DistilBertTokenizer.from_pretrained(self.sentiment_model_name)
        
        # Initialize summarization model and tokenizer
        self.model_name = 'sshleifer/distilbart-cnn-12-6'
        self.revision = 'a4f8f3e'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       revision=self.revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name,
                                                           revision=self.revision)
        self.weaviate_url = "http://localhost:8080"  # Replace this with your Weaviate instance URL
        self.weaviate_client = weaviate.Client(self.weaviate_url)


    def google_news_search(self, query):
        base_url = "https://www.googleapis.com/customsearch/v1"
        results_per_request = 10
        num_results = self.num_results
        num_requests = num_results // results_per_request
        remaining_results = num_results % results_per_request
        
        links = []

        for i in range(num_requests):
            start = 1 + i * results_per_request

            params = {
                "q": query,
                "cx": self.my_google_search_engine_id,
                "key": self.my_google_api_key,
                "num": results_per_request,
                "start": start,
                "sort": "date:d:type:2",
                "lr": "lang_en"
            }

            response = requests.get(base_url, params=params)
            data = response.json()

            if "items" not in data:
                break

            links.extend([item["link"] for item in data["items"]])

        # Request remaining results
        if remaining_results > 0:
            start = 1 + num_requests * results_per_request

            params = {
                "q": query,
                "cx": self.my_google_search_engine_id,
                "key": self.my_google_api_key,
                "num": remaining_results,
                "start": start,
                "sort": "date:d:type:2",
                "lr": "lang_en"
            }

            response = requests.get(base_url, params=params)
            data = response.json()

            if "items" in data:
                links.extend([item["link"] for item in data["items"]])

        return links

    def is_paywalled(self, soup):
        # This is just an example of some common paywall-related elements.
        # You may need to update this list based on the websites you encounter.
        paywall_selectors = [
            {"tag": "meta", "attrs": {"name": "robots", "content": "noindex,nofollow"}},
            {"tag": "div", "attrs": {"class": "paywall"}},
            {"tag": "section", "attrs": {"class": "meteredContent"}},
        ]
    
        for selector in paywall_selectors:
            if soup.find(selector["tag"], attrs=selector["attrs"]):
                return True
    
        return False



    def get_publisher(self, article):
        meta_data = article.meta_data
    
        # Method 1: Check for "og:site_name"
        publisher = meta_data.get("og", {}).get("site_name")
        if publisher:
            return publisher.replace(".", "_").replace(" ", "_")
    
        # Method 2: Check for <span class="author-name">
        html = article.html
        soup = BeautifulSoup(html, "html.parser")
        publisher_span = soup.find("span", class_="author-name")
        if publisher_span:
            return publisher_span.text.replace(".", "_").replace(" ", "_")
    
        # Method 3: Extract domain from URL
        url = article.source_url
        if url.startswith('https://news.google.com/rss/articles/'):
            url = url.split("articles/")[1]
            actual_url = url.split("?")[0]
            actual_url = urllib.parse.unquote(actual_url)
        else:
            actual_url = url
    
        if actual_url:
            domain_name = actual_url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
            publisher = domain_name.replace(".", "_")
            if publisher != "news_google_com":
                return publisher
    
        # If no publisher is found after all methods, print a warning and return "unknown_publisher"
        print("Warning: Publisher not found for URL:", article.source_url)
        return "unknown_publisher"




    def format_filename(self, publisher, company, date):
        date_string = date.strftime("%Y-%m-%d") if date else "unknown_date"
        return f"{publisher}_{company}_{date_string}.txt"

    def file_exists(self, filepath):
        return os.path.exists(filepath)

    def extract_article_content_newspaper(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()

            content = article.text
            publisher = self.get_publisher(article)
            date = article.publish_date


            return {"content": content, "publisher": publisher, "date": date,
                    "article": article}

        except Exception as e:
            print(f"Error: {e}")
            return None

    def scrape_articles(self, company_name):
        num_results = self.num_results
        # List of financial news websites you want to scrape.
        '''financial_sites = [
            "seekingalpha.com",
            "fool.com",
            "marketwatch.com",
            "finance.yahoo.com",
            "businessinsider.com",
            "reuters.com",
            "bloomberg.com",
            "investing.com",
        ]
        
        site_filter = " OR ".join([f"site:{site}" for site in financial_sites])'''
        
        query = f'"{company_name}"'



        urls = self.google_news_search(query)


        articles = []
        for url in urls:
            article_data = self.extract_article_content_newspaper(url)
            if article_data:
                content = article_data["content"]
                publisher = self.get_publisher(article_data['article'])
                date = article_data["date"]

                filename = self.format_filename(company_name, publisher, date)
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(content)

                articles.append({"company": company_name, "url": url, "content": content, "publisher": publisher, "date": date})

        return articles

    def is_relevant(self, article, keywords):
        snippet = article.get('content', '').lower()
        return any(keyword.lower() in snippet for keyword in keywords)
    
    def save_articles(self, articles, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        for article_data in articles:
            company = article_data["company"]
            content = article_data["content"]
            publisher = article_data["publisher"]
            date = article_data["date"]
    
            filename = self.format_filename(publisher, company, date)
            filepath = os.path.join(output_dir, filename)
    
            if self.file_exists(filepath):
                print(f"File {filepath} already exists. Skipping.")
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Saved: {filepath}")

    def prettify_text(self, text):
        text = re.sub(r'\n', ' ', text) # Replace newline characters with spaces
        text = re.sub(r'\s{2,}', ' ', text) # Replace multiple spaces with a single space
        return text

    def summarize_articles(self, articles):
        summaries = []
        for article in articles:
            content = str(article['content'])
            # Truncate the content to the maximum allowed sequence length
            content = content[:1024]
            tokens = tokenizer.encode(content, return_tensors="pt")
            summary_ids = model.generate(tokens, num_return_sequences=1, max_length=512)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries

    def sentiment_analysis(self, summaries):
        sentiments = []
        for summary in summaries:
            tokens = sentiment_tokenizer.encode(summary, return_tensors="pt")
            logits = sentiment_model(tokens)[0]
            sentiment = torch.argmax(logits).item()
            sentiment_label = "positive" if sentiment == 1 else "negative"
            sentiments.append(sentiment_label)
        return sentiments
    
    def calculate_rolling_sharpe_ratio(self, stock_ticker, risk_free_rate_series,
                                       window_size=12, force_download=False):
        # Define the CSV filename
        csv_filename = f'/media/somebody/big_store/facc_stocks_1mo/{stock_ticker}_data.csv'

        if not force_download and os.path.exists(csv_filename):
            # Load the existing CSV file and get the date of the last entry
            stock_data = pd.read_csv(csv_filename, index_col='Date', parse_dates=True)
            last_entry_date = stock_data.index[-1].date()

            # Check if the last entry date is up-to-date
            if last_entry_date >= datetime.now().date() - timedelta(days=1):
                print(f'The last entry in the file ({last_entry_date}) is up-to-date. Skipping download.')
            else:
                print(f'The last entry in the file ({last_entry_date}) is outdated. Downloading new data...')
                new_data = yf.download(stock_ticker, start=last_entry_date + timedelta(days=1), end=datetime.now().strftime('%Y-%m-%d'), interval='1mo', progress=False)
                stock_data = stock_data.append(new_data[~new_data.index.isin(stock_data.index)])


        else:
            # Download the stock data
            stock_data = yf.download(stock_ticker, interval='1mo', period='max', progress=False)
        
        # Check if the last row is for the current month and not the first trading day of the month
        now = datetime.now()
        last_row_date = stock_data.index[-1]
        if now.month == last_row_date.month and now.year == last_row_date.year and (now.day - last_row_date.day) < now.day:
            stock_data = stock_data.iloc[:-1]
        
        # Calculate monthly returns
        monthly_returns = stock_data['Adj Close'].pct_change().dropna()

        # Calculate rolling average monthly returns
        rolling_avg_monthly_returns = monthly_returns.rolling(window=window_size).mean()

        # Calculate rolling standard deviation of monthly returns
        rolling_std_dev_monthly_returns = monthly_returns.rolling(window=window_size).std()

        # Calculate annual risk-free rate from daily risk-free rate
        daily_risk_free_rate = risk_free_rate_series / 100 / 252
        annual_risk_free_rate = (1 + daily_risk_free_rate)**252 - 1

        # Calculate the rolling annual Sharpe ratio
        rolling_sharpe_ratio = (rolling_avg_monthly_returns * 12 - annual_risk_free_rate) / (rolling_std_dev_monthly_returns * (12**0.5))

        # Add the rolling Sharpe ratio to the stock_data DataFrame
        stock_data['Rolling Sharpe Ratio'] = rolling_sharpe_ratio

        # Save the updated stock_data DataFrame to the CSV file
        stock_data.to_csv(csv_filename)

        return stock_data

            
    def run(self):
        articles = self.scrape_articles(self.stock_name)
        relevant_articles = articles if not self.keywords else [article for article in articles if self.is_relevant(article, self.keywords)]
    
        summaries = self.summarize_articles(relevant_articles)
        sentiments = self.sentiment_analysis(summaries)
    
        # Loop through the relevant_articles, summaries, and sentiments
        for i, (article, summary, sentiment) in enumerate(zip(relevant_articles, summaries, sentiments)):
            article_data = {
                "url": article["url"],
                "content": article["content"],
                "publisher": article["publisher"],
                "date": article["date"].isoformat() if article["date"] else None,
                "summary": summary,
                "sentiment": sentiment,
            }
    
            # Save the article data to Weaviate
            self.save_article_data_to_weaviate(article_data)
            print(f"Article {i+1} added to Weaviate")
    
        return relevant_articles, summaries, sentiments

    
    def check_if_url_exists(self, url_hash):
        query = {
            "query": {
                "exists": {
                    "operator": "Equal",
                    "value": True,
                    "path": ["url_hash"],
                    "valueType": "string",
                    "valueString": url_hash
                }
            }
        }
    
        response = self.weaviate_client.query.get(class_name='Article',
        params=query)
    
        return response["data"]["Get"]["Article"] is not None

    
    def save_article_data_to_weaviate(self, article_data):
        weaviate_object = {
            "url": article_data["url"],
            "content": article_data["content"],
            "publisher": article_data["publisher"],
            "date": article_data["date"],
            "summary": article_data["summary"],
            "sentiment": article_data["sentiment"],
        }
    
        try:
            weaviate_client.data_object.create(weaviate_object, "Article")
        except Exception as e:
            print(f"Error saving article data to Weaviate: {e}")

        
pipeline = NewsSentimentPipeline(stock_name="Microsoft",
 num_articles=5, num_results=10, keywords=['Microsoft', "OpenAI"],
 my_google_api_key=my_google_api_key,
 my_google_search_engine_id=my_google_search_engine_id)
 
articles, summaries, sentiments = pipeline.run()



