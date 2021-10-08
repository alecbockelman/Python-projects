# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 02:29:23 2021

@author: Owner
"""
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

csv="VIX_hist.csv"
data = pd.read_csv(csv)

def hist_average_close (data):
    hist_avg = data['CLOSE'].mean()
    return hist_avg

def hist_average_std_dev (data):
    hist_std_dev =[]
    for row in data:
        hist_std_dev = (data['CLOSE'].rolling(20).std())
        for i in range(20):
            del hist_std_dev[i]
        avg_hist_std_dev = hist_std_dev.mean()
    return avg_hist_std_dev
 
def vix_current():
    VIX = yf.Ticker('^VIX')
    df_curr = data.iloc[len(data)-20:,:]
    curr_std_dev = df_curr['CLOSE'].std()
    return VIX.info['regularMarketPrice'], curr_std_dev

df_curr = data.iloc[len(data)-20:,:]
df_curr_datetime = df_curr['DATE']


y = df_curr['CLOSE']
x = pd.to_datetime(df_curr_datetime)

fig, ax = plt.subplots(figsize=(12, 12))
plt.plot(x,y)
plt.ylabel('VIX Price')
plt.xlabel('DATE')
plt.title('VIX Price Last 20 Market Days')


plt.show()

print('The historical average vix close is:     ', hist_average_close(data))
print('The average historical 20 day vix std dev is:    ', hist_average_std_dev(data))
print('Current VIX price: ',  vix_current()[0], '       Past 20 day avg std dev: ', vix_current()[1])


