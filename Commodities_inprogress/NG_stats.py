# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:13:20 2022

@author: Owner
"""


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas import Series
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import datetime
import math 
from statistics import mean
from statsmodels.tsa.stattools import adfuller

csv= 'NG_eia_data.csv'
data = pd.read_csv(csv)

