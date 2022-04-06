# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:53:37 2022

@author: Owner
"""


import yfinance as yf
import pandas as pd

freefloat = yf.Ticker('AAPL').info

#Determine point of business cycle

csv= "stock_tickers.csv"
stocks = pd.read_csv(csv)


listprice =[]
listsector =[]
listfloat = []
list_fiftydayavg  =[]


def price(ticker):
    price = yf.Ticker(ticker).info.get('currentPrice')
    listprice.append(price)
    return listprice
    
result = [price(x) for x in stocks['Symbol']]

stocks.insert(2, "Price", price, True)

def floatshs(ticker):
    freefloat = yf.Ticker(ticker).info.get('floatShares')
    listfloat.append(freefloat)
    return listfloat
    
result = [floatshs(x) for x in stocks['Symbol']]

stocks.insert(3, "Float", listfloat, True)

def sector(ticker):
    sector = yf.Ticker(ticker).info.get('sector')
    listsector.append(sector)
    return listsector
    
result = [sector(x) for x in stocks['Symbol']]

stocks.insert(4, "Sector", sector, True)

def fiftydayavg(ticker):
    fiftydayavg = yf.Ticker(ticker).info.get('fiftyDayAverage')
    list_fiftydayavg.append(fiftydayavg)
    return list_fiftydayavg 
    
result = [fiftydayavg(x) for x in stocks['Symbol']]

stocks.insert(5, "50Day Avg", list_fiftydayavg, True)


filename = "stock_tickers_info.csv"
stocks.to_csv(filename, encoding='utf-8', index=False)


