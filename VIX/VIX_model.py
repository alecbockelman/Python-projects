# -*- coding: utf-8 -*-

"""
Created on Fri Nov  5 00:34:51 2021

@author: Alec Bockelman
"""


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statistics
from math import sqrt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import Series
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


csv="VIX_History.csv"
csv2= "CBOE Vix Volatility Historical Data.csv"
data = pd.read_csv(csv)
data2= pd.read_csv(csv2)
data2 = data2.iloc[::-1] #reverses the second data set
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

def hist_average_std_dev_vvix (data):
    hist_std_dev_vvix =[]
    for row in data2:
        hist_std_dev_vvix = (data2['Price'].rolling(20).std())
        for i in range(20):
            del hist_std_dev_vvix[i]
        avg_hist_std_dev_vvix = hist_std_dev_vvix.mean()
    return avg_hist_std_dev_vvix
 
def vix_current():
    VIX = yf.Ticker('^VIX')
    df_curr = data.iloc[len(data)-20:,:]
    curr_std_dev = df_curr['CLOSE'].std()
    return VIX.info['regularMarketPrice'], curr_std_dev

 
def vvix_current():
    VVIX = yf.Ticker('^VVIX')
    df_curr2 = data2.iloc[len(data2)-20:,:]
    curr_std_dev2 = df_curr2['Price'].std()
    return VVIX.info['regularMarketPrice'], curr_std_dev2

df_curr = data.iloc[len(data)-20:,:]
df_curr_datetime = df_curr['DATE']


y = df_curr['CLOSE']
x = pd.to_datetime(df_curr_datetime)

fig, ax = plt.subplots(figsize=(12, 12))
plt.plot(x,y)
plt.ylabel('VIX Price')
plt.xlabel('DATE')
plt.title('VIX Price Last 30 Market Days')

plt.show()

print('The historical average vix close is:     ', hist_average_close(data))
print('The average historical 20 day vix std dev is:    ', hist_average_std_dev(data))
print('The average historical 20 day vvix std dev is:    ', hist_average_std_dev_vvix(data2))
print( 'Past 20 day avg std dev of VIX: ', vix_current()[1])
print( "Past 20 day avg std dev of VVIX: ", vvix_current()[1])
print('Current VIX price: ',  vix_current()[0])

#Predicting a VIX spike
x_point=[] #create list for x values, containing date
y_point=[] #create list of y values, containing the historical vix price
vix_std = []
vvix_std = []
for row in data:
    vix_std=(data['CLOSE'].rolling(20).std())
    for i in range(20):
        del vix_std[i]
for row in data2:
    vvix_std=(data2['Price'].rolling(20).std())
    del vvix_std[i]

vix_std = vix_std.iloc[len(vix_std)-2505:]  #resize the dataframe to have same start date as vvix on 10/6/2011
vvix_std = vvix_std.iloc[20:2526]

vix_std = vix_std.tolist()
vvix_std = vvix_std.tolist()

# find low std dev and filter out clusters: objective = identify future vix spikes
for l in range(len(vix_std)):
     if vix_std[len(vix_std)-1-l] <= .86 and vvix_std[len(vix_std)-1-l] <= 3.16:
        if vix_std[len(vix_std)-1-l] <= (vix_std[len(vix_std)-2-l] or vix_std[len(vix_std)-3-l]  or  vix_std[len(vix_std)-4-l]) and vvix_std[len(vix_std)-1-l] <= (vvix_std[len(vix_std)-2-l] or vvix_std[len(vix_std)-3-l] or vvix_std[len(vix_std)-4-l]):
            x_point.append(data2['Date'][l])
            y_point.append(data['CLOSE'][len(data) -1 -l])
 
    
x_peak_point =[]
y_peak_point = []
# find peaks at high std dev and when vix is above the historical average of approx 20 and a jump of > 30 % from 5 days ago has occured: objective = identify vix peaks    
for l in range(len(vix_std)):
    if vix_std[len(vix_std)-1-l] >= 3.16 and data['CLOSE'][len(data) -1 -l] >= 20 and data['CLOSE'][len(data) -2 -l] >= 20  and data['CLOSE'][len(data) -1 -l] >= (data['CLOSE'][len(data) -6 -l]*1.3) :    
        x_peak_point.append(data2['Date'][l])
        y_peak_point.append(data['CLOSE'][len(data) -1 -l])
 
        
#vix historical prices from 2011-current
data_datetime = data['DATE'].iloc[len(data)-2505:]
y = data['CLOSE'].iloc[len(data)-2505:]
x = pd.to_datetime(data_datetime)

#create datetime objects for x axis
x_point = pd.to_datetime(x_point)
x_peak_point = pd.to_datetime(x_peak_point)

#series from 2011-current for autoregression mode, vvix max historical data starts at 2011
data_ar = data.iloc[len(data)-2506:len(data)-1]
data_date = data_ar["DATE"].index.tolist()
data_close = data_ar["CLOSE"].tolist()
vix_series = Series(data_close, index = data_date )





fig, ax = plt.subplots(figsize=(12, 12))
plt.plot(x,y)
plt.scatter(x_point,y_point,  color='black', marker='o', s=95)
plt.scatter(x_peak_point,y_peak_point,  color='green', marker='o', s=95)
plt.ylabel('VIX Price')
plt.xlabel('DATE')
plt.title('Historical VIX Price')




plot_acf(vix_series, lags =900)
plot_pacf(vix_series, lags =10)
plt.show()

# fit model
model = ARIMA(vix_series, order=(1,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

model_fit.summary()



X = vix_series.values
size = int(len(X) * 0.75)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
obs_list =[]
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 

# seasonal difference
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(1,1,0))
model_fit = model.fit()
# multi-step out-of-sample forecast
start_index = len(differenced)
end_index = start_index + 9
forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
x_val_forecast =[]
y_val_forecast=[]
future_std_dev=[]
forecast = forecast.tolist()
for yhat in forecast:
    inverted = inverse_difference(history, yhat, days_in_year)

    x_val_forecast.append(len(test)+day)
    #add values to vix closing price list to allow for past 20 day std dev calc
    data_close = data_close[-20::]
    std_dev = statistics.stdev(data_close)
    future_std_dev.append(std_dev)
    y_val_forecast.append(inverted)
    history.append(inverted)
    data_close.append(inverted)
    print('Day %d: %f' % (day, inverted))
    day += 1
#plot forecasts against actual outcome
fig, ax = plt.subplots(figsize=(12, 12))
plt.plot(test)
plt.plot(x_val_forecast, y_val_forecast, color = 'blue', linestyle='dashed')
plt.plot(predictions, color='red')
plt.show()
