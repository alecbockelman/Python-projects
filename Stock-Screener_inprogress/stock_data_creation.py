# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:53:37 2022

@author: Owner
"""


import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

#Determine point of business cycle

csv= "stock_tickers.csv"
raw_df = pd.read_csv(csv)

list_tickers = []
list_longname = []
list_price = []
list_sector = []
list_float = []
list_fiftydayavg = []

series_tickers = raw_df['Symbol']
list_raw_tickers = series_tickers.tolist()


def sector(ticker):
    info = yf.Tickers(ticker).tickers[ticker].info
    if bool({info['sector']}) == True:
        list_tickers.append(ticker)
        list_longname.append(info['longName'])
        list_sector.append(info['sector'])
        list_price.append(info['currentPrice'])
        list_float.append(info['floatShares'])
        list_fiftydayavg.append(info['fiftyDayAverage'])
    return list_tickers, list_longname, list_sector, list_price, list_float, list_fiftydayavg

with ThreadPoolExecutor() as executor:
    executor.map(sector, list_raw_tickers)

d = list(zip(list_tickers,list_longname))
stocks = pd.DataFrame(d, columns = ['Symbol', 'Name'])
stocks.insert(2, "Sector", list_sector, True)
stocks.insert(3, "Price", list_price, True)
stocks.insert(4, "Float", list_float, True)
stocks.insert(5, "50 Day AVG", list_fiftydayavg, True)


stocks.dropna()

# def price(ticker):
#     price = yf.Ticker(ticker).info.get('currentPrice')
#     listprice.append(price)
#     return listprice
    
# result = [price(x) for x in stocks['Symbol']]

# stocks.insert(2, "Price", price, True)



filename = "stock_tickers_with_sector.csv"
stocks.to_csv(filename, encoding='utf-8', index=False)


