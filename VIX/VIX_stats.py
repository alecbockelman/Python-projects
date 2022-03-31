# -*- coding: utf-8 -*-

"""
Created on Fri Nov  5 00:34:51 2021

Stats of VIX

@author: Alec Bockelman
"""

''''''

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas import Series
from statistics import mean

csv= 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
data = pd.read_csv(csv)
data= data[:len(data):]

VIX = yf.Ticker('^VIX')
close_price_vix = VIX.history(period ='1Y')['Close']



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

def hist_percentile_std_dev (data):
    hist_std_dev =[]
    for row in data:
        hist_std_dev = (data['CLOSE'].rolling(20).std())
        for i in range(20):
            del hist_std_dev[i]
        hist_25_percentile = hist_std_dev.quantile(q=.15)
        hist_85_percentile = hist_std_dev.quantile(q=.85)
    return hist_25_percentile, hist_85_percentile #skewed


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
fig.autofmt_xdate()
plt.plot(close_price_vix[-20::])
plt.ylabel('VIX Price')
plt.xlabel('DATE')
plt.title('VIX Price Last 20 Market Days')

plt.show()

print('The historical average vix close is:     ', hist_average_close(data))
print('\nThe average historical 20 day vix std dev is:    ', hist_average_std_dev(data))
print('\nThe historical 20 day vix std dev 15th percentile is:    ', hist_percentile_std_dev(data)[0])
print('\nThe historical 20 day vix std dev 85th percentile is:    ', hist_percentile_std_dev(data)[1])
print( '\nPast 20 day avg std dev of VIX: ', vix_current()[1])
print('\nCurrent VIX price: ',  vix_current()[0])

#Predicting a VIX spike
x_point=[] #create list for x values, containing date
y_point=[] #create list of y values, containing the historical vix price
vix_std = []
vvix_std = []
for row in data:
    vix_std=(data['CLOSE'].rolling(20).std())
    for i in range(20):
        del vix_std[i]

vix_std = vix_std.iloc[-3780::]  #resize the dataframe to Last 5000 trading day observations or 15 years of historical data

vix_std = vix_std.tolist()

#vix historical prices from 2011-current
data_datetime = data['DATE'].iloc[-3780::] 
y = data['CLOSE'].iloc[-3780::]
x = pd.to_datetime(data_datetime)

#series from 2006-current for autoregression mode
data_ar = data.iloc[-3780::] 
data_date = pd.to_datetime(data_ar["DATE"])
data_close = data_ar["CLOSE"].tolist()
vix_series = Series(data_close, index = data_date )
        
# Function to check the value
def CheckForLess(list1, val): 
      
    # traverse in the list 
    for x in list1: 
  
        # compare with all the  
        # values with value
        if val <= x: 
            return False
    return True

# find low std dev and filter out clusters: objective = identify future vix spikes
for l in range(30, len(vix_std)):
     if vix_std[l] <= 0.75 : #change to var
         if data_close[l] <= 19.47: #check for below historical avg
             if CheckForLess(vix_std[l-10: l:], vix_std[l])==False: #no vix std of below .93 past 10 trading days to screen out clusters of low volatility
                x_point.append(data_date.values[l])
                y_point.append(data_close[l])

     
    
x_peak_point =[]
y_peak_point = []
# find peaks at high std dev and when vix is above the historical average of approx 19,47 and a jump of > 30 % from 5 days ago has occured: objective = identify vix peaks    
for l in range(10,len(vix_std)):
    if vix_std[l] >= 2.77:#change to var
        if data_close[l] >= data_close[l-5]*1.3: 
            if data_close[l] > 19.47: #make a var
                x_peak_point.append(data_date.values[l])
                y_peak_point.append(data_close[l])
     





#create datetime objects for x axis
x_point = pd.to_datetime(x_point)
x_peak_point = pd.to_datetime(x_peak_point)



days_out_return = 63 # quarterly return from a given signal is  = 252/4 = 63 days out
list_low_vol_return =[]  
for i in range((len(y_point))-2):  #could be out of range
    list_low_vol_return.append(((data_close[data_close.index(y_point[i]) + days_out_return   ]-y_point[i])/y_point[i]) *100)
    
avg_low_vol_signal_return = mean(list_low_vol_return)
 
 
list_high_vol_return =[]  
for i in range((len(y_peak_point))-10):  #could be out of range
    list_high_vol_return.append(((data_close[data_close.index(y_peak_point[i]) + days_out_return   ]-y_peak_point[i])/y_peak_point[i]) *100)
    
avg_high_vol_signal_return = mean(list_high_vol_return)
 

print('\nThe average', days_out_return,' day drop in the VIX from a peak signal is', abs(avg_high_vol_signal_return), '%')
print('\nThe average', days_out_return,' day gain in the VIX from a low signal is', abs(avg_low_vol_signal_return), '%')

fig, ax = plt.subplots(figsize=(12, 12))
plt.plot(x,y)
plt.scatter(x_point,y_point,  color='black', marker='o', s=95)
plt.scatter(x_peak_point,y_peak_point,  color='green', marker='o', s=95)
plt.ylabel('VIX Price')
plt.xlabel('DATE')
plt.title('Historical VIX Price')

