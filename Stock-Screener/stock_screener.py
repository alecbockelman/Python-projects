# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:53:37 2022

@author: Owner
"""


import yfinance as yf
import pandas as pd


#Determine point of business cycle

csv= "stock_tickers.csv"
stocks = pd.read_csv(csv)

# freefloat = yf.Ticker('A').info
listfloat = []


def floatshs(ticker):
    freefloat = yf.Ticker(ticker).info.get('floatShares')
    listfloat.append(freefloat)
    
result = [floatshs(x) for x in stocks['Symbol']]



