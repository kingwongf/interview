#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import pandas as pd
import numpy as np
import json


# In[2]:


from lxml.html import fromstring
import requests
from itertools import cycle
import traceback
import yfinance as yf

def get_proxies(n_proxies):
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = []
    for i in parser.xpath('//tbody/tr')[:n_proxies]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.append(proxy)
    return proxies


def proxy_get(url, n_proxies=100):
    proxies = get_proxies(n_proxies)
    proxy_pool = cycle(set(proxies))
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    for i in range(1, n_proxies+1):
        # Get a proxy from the pool
        proxy = next(proxy_pool)
        # print("Request #%d" % i)
        try:
            response = requests.get(url, proxies={"http": proxy, "https": proxy}, headers=headers)
            return response
        except Exception as e:
            pass
    return requests.get(url)

# In[4]:


## First we need components of the Stoxx600
## however there seems to be no available list of tickers 
## instead we need to find the name => ticker =>download historical data

## This is the first approach, which yields 136 tickers, 
## please skip to cell 9 for the final approach of 412 tickers

## investing.com has names of the company, and each has a href which leads to ticker/ symbol
html_index = r"https://uk.investing.com/indices/stoxx-600-components"
r = proxy_get(html_index)
index_soup = BeautifulSoup(r.text, 'html.parser')

names_list = set()
href_list = set()
for trs in index_soup.find_all('tbody'):
    for a in trs.find_all('a'):
        if 'equities' in a['href']:
            names_list.add(*a.contents)
            href_list.add(a['href'])


# In[5]:


len(names_list)


# In[6]:


with open("stoxx600_component_names.txt", "w") as text_file:
    text_file.write(",".join(names_list))


# In[7]:


## join with a file that supposed to have all yahoo finance stock tickers
## seems the list is a bit short 
names2ticker = pd.read_csv("name2ticker.csv", encoding='ISO-8859-1')
europe_countries = ["Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland","Italy","Kazakhstan","Latvia","Liechtenstein","Lithuania","Luxembourg","Macedonia","Malta","Moldova","Monaco","Montenegro","Netherlands","Norway","Poland","Portugal","Romania","Russia","San Marino","Serbia","Slovakia","Slovenia","Spain","Sweden","Switzerland","Turkey","Ukraine","United Kingdom","Vatican City"]
names2ticker.loc[(names2ticker['Name'].isin(names_list))&(names2ticker['Country'].isin(europe_countries))]


# In[8]:


## we'll save this for now, let see if we can get ticker by going to each href link
fornow_li_stoxx600 = names2ticker.loc[(names2ticker['Name'].isin(names_list))&(names2ticker['Country'].isin(europe_countries))]['Ticker'].tolist()


### Comment out Below to Run 
# 
# ## rather than joining to the file, we can go to each href and obtain ticker,
# ## would need proxies and investing.com bans traffic even with headers/ low request frequency
# ## this found to be very slow, due to free proxies 
# 
# new_new_li_yhoo_tickers = []
# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
# for href in href_links[::-1]:
#     url_ticker = f"https://www.investing.com{href}"
#     r = proxy_get(url_ticker)
#     ticker_index_soup = BeautifulSoup(r.text, 'html.parser')
#     for sth in ticker_index_soup.find_all("div", {"class":"instrumentHead"}):
#         for sth_else in sth.find_all("h1"):
#             t = sth_else.contents[0]
#             ticker = re.search(r'\((.*?)\)',t).group(1)
#             # print(re.search(r"\[([A-Za-z0-9_]+)\]", t))
#         for exchange in sth.find_all("div", {"class":"exchangeDropdownContainer"}):
#             location = exchange.find("i").contents[0]
#     try:  
#         ex_suffix = city2ticker.loc[(city2ticker['Market'].str.contains(location))|(city2ticker['City'].str.contains(location))|(city2ticker['Country'].str.contains(location))].head(1)['Suffix'].values[0]
#         print(f"{ticker}{ex_suffix}")
#         
#         
#         new_new_li_yhoo_tickers.append(f"{ticker}{ex_suffix}")
#     
#     
#     except:
#         
#         print(f"fail {ticker} {location}")
#         
#         
#         
#         
# ###############################################################################3        

# In[9]:


## The successful method
## finding out hidden deep in stoxx website, a list of component companies name
## New approach would be search for autocomplete from yahoo finance then download with ticker
company_names = pd.read_csv('SXXGR.csv')
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
tickers = []
for name in company_names['Company'].tolist():
    yhoo_url = f"http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={name}&region=1&lang=en&callback=YAHOO.Finance.SymbolSuggest.ssCallback"
    r = requests.get(yhoo_url, headers=headers)
    try:
        ticker = [x for x in json.loads(re.search(r'\((.*?)\)',r.text).group(1))['ResultSet']['Result'] if x['exch']!='PNK' and x['exchDisp']!='OTC Markets' and x['exchDisp']!='NYSE' ]
        print(f"{name}: {ticker[0]['symbol']}")
        tickers.append(ticker[0]['symbol'])
    except Exception as e:
        print(f"fail {name}")




# In[10]:


## filter out possible US companies, not within stoxx 600 based on it must have an . extension
tickers = [ticker for ticker in tickers if '.' in ticker]
print(len(tickers))
tickers_str = " ".join(tickers)
new_data = yf.download(tickers=tickers_str,period='max')


# In[11]:


print(new_data['Close'])




# In[22]:


new_data.dropna(how='all', axis=1, inplace=True)
new_data['Close'].to_pickle('new_close_stoxx600.pkl')


# In[13]:


## Stoxx 600, index price
stoxx = yf.Ticker('^STOXX')
stoxx = stoxx.history(period="max")
print(stoxx)
stoxx.to_pickle('stoxx600.pkl')





