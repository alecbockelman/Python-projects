# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 02:15:39 2021

@author: Owner
"""

import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
import yfinance as yf
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import yahoofinancials

gld = yf.Ticker('GLD')
# get stock info


Df = yf.download('GLD', '2008-01-01', '2021-12-25', auto_adjust=True)
Df_SLV = yf.download('SLV', '2008-01-01', '2021-12-25', auto_adjust=True)
Df.insert(5, 'SLV Closing Price' , Df_SLV['Close'], True)
Df_DRD= yf.download('DRD', '2008-01-01', '2021-12-25', auto_adjust=True)
Df_GDX= yf.download('GDX', '2008-01-01', '2021-12-25', auto_adjust=True)
Df_AEM= yf.download('AEM', '2008-01-01', '2021-12-25', auto_adjust=True)
Df_HMY= yf.download('HMY', '2008-01-01', '2021-12-25', auto_adjust=True)
Df_NEM= yf.download('NEM', '2008-01-01', '2021-12-25', auto_adjust=True)
Df = Df.dropna()
Df['Next Day Closing Price'] = Df['Close']


# Define explanatory variables
Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['S_250_SLV'] = Df['SLV Closing Price'].rolling(window=250).mean()
Df['DRD'] = Df_DRD['Close']
Df['GDX']=Df_GDX['Close']
Df['HMY']=Df_HMY['Close']
Df['NEM']=Df_NEM['Close']
Df['AEM']=Df_AEM['Close']

Df['Next Day Closing Price'] = Df['Close']
Df = Df.dropna()
X = Df[['S_3', 'S_9', 'S_250_SLV', 'DRD', 'NEM', 'AEM']]
y = Df['Next Day Closing Price']


# Split the data into train and test dataset
t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]


retrain = input("Do you want to retrain the model[y/n] ?")
best = 0
i = 0

if retrain == 'y':
    for i in range(100):
        linear = linear_model.LinearRegression()
        linear.fit(X_train, y_train)
        acc = linear.score(X_test, y_test)

        if acc > best:
            best = acc
            with open('GLDmodel.pickle', 'wb') as file:
                pickle.dump(linear, file)

pickle_in = open('GLDmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

# Print linear regression,y=mx+b
print("Linear Regression model")
print('Coefficant: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predicted_price = linear.predict(X_test)
guess=[]
total_guess=0

for x in range(len(predicted_price)):
    if predicted_price[x] < y_test.values[x]:
        guess.append(0)
        print('Closing Price = ', y_test.values[x], 'on ', y_test.index[x] ,'Predicted Next Day Out Closing Price = ', predicted_price[x],  'rating = sell')
    elif predicted_price[x] > y_test.values[x]:
        guess.append(1)
        print('Closing Price = ', y_test.values[x],'on ', y_test.index[x] , 'Predicted Next Day Out Closing Price = ', predicted_price[x], 'rating = buy')
    total_guess+=1



'list of returns per trade'
returns=[]
for x in range(len(predicted_price)-1):
    if guess[x] == 0:
        returns.append(((y_test.values[x+1]-y_test.values[x])/y_test.values[x])*100)
    if guess[x] == 1:
        returns.append(((y_test.values[x] - y_test.values[x+1]) / y_test.values[x])*100)

'average return per trade'
return_sum = 0
avg_returns =0
for i in range(len(returns)):
    return_sum += returns[i]
avg_returns= return_sum /len(returns)
print('Average return per trade', avg_returns, '%')

val_comp=0
val_init=10000
for i in range(5000):
    val_comp = val_init + (avg_returns / 100) * (val_init)
    val_init = val_comp
print('Total 20 year return beginning with 10000', val_comp,'$')


predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 7))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("GLD Price")
plt.show()

