
"""
Created on Fri Nov  5 00:34:51 2021

ARIMA Model of the VIX in sample

@author: Alec Bockelman
"""

''''''

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math 
from statsmodels.tsa.stattools import adfuller
from VIX_stats import vix_series, data_date



X = np.log(vix_series.values)  ## log values to prevent negative forecasts
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]



if adfuller(train)[1] < 0.05:
    print("\np-value:", adfuller(train))
    print('\nWe reject the null hypothesis and the VIX is determined to be stationary')
    
#determine d value
train_diff = np.diff(train)
plot_acf(train, lags =50) # plot autocorrelation
plot_acf(train_diff, lags =10)

if adfuller(train)[1] < 0.05:
    print("\np-value:", adfuller(train_diff))
    print('\nTake the first difference in order to make the series more stationary')
    print('\nSince the difference of VIX is stationary,  we can assume that the VIX is a random walk and the d value = 1')
    d =1

#determine p value
import warnings
warnings.filterwarnings("ignore")

plot_pacf(train, lags =10)
print('\nSince the number of lags is significant to 1 in the pacf, the p value is set to 1')
p =1

#determine best fitting q values from pacf


print('\nThe q value was determined from the acf graph to be 1')
q = 1
print('\nThe ARIMA model choosen = ', p,',',d,',',q)



model = ARIMA(test, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())


# line plot of residuals
residuals = pd.DataFrame((model_fit.resid))
residuals = residuals[1::] #remove intial spike
residuals.plot(title = 'Residuals')
plt.show()
# density plot of residuals
residuals.plot(kind='kde', title = 'Density')
plt.show()




history = train.tolist()

test = test.tolist()
predictions = list()


count=0
arima_predictions_low = []
arima_predictions_high = []
for i in range(len(test)):
    count += 1
    model = ARIMA(history, order=(1,1,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    obs = test[i]
    predictions.append(yhat)

    if count ==3:
         history.append(obs)
         count = 0
    else:
         history.append(yhat)

i=0
for val in test:
    test[i] = math.exp(test[i])
    i+=1
    
i=0
for val in predictions:
    predictions[i] = math.exp(predictions[i])
    i+=1
    
rmse = sqrt(mean_squared_error(test, predictions))
print('\nTest RMSE: %.3f' % rmse)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.autofmt_xdate()
plt.plot(data_date[-50::],test[-50::], color = 'green')
plt.plot(data_date[-50::],predictions[-50::], color = 'red')
plt.show()

