# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:47:11 2022

@author: Alexander Bockelman

Dupont Beta L & WACC
"""
import yfinance as yf
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

ticker = 'DD'
totaldebt =11.26
ann_interest = .525
rf = 2
mktprm = 5.5
taxrate =.25
new_d_to_e = 0.5
increase_rd = 1

mkt = '^GSPC' #GSPC is the S&P 500 Index in yahoo finance
def BetaL(ticker,mkt_ticker):
    eqdata = yf.download(ticker,period = "2Y" )
    eqdata = eqdata['Close']
    mdata = yf.download(mkt_ticker,period = "2Y")
    mdata = mdata['Close']
    offset = pd.offsets.BusinessDay(-1)
    eqdata_week_ret = eqdata.resample('W',loffset=offset).last().pct_change()
    mdata_week_ret = mdata.resample('W',loffset=offset).last().pct_change()
    frames = [eqdata_week_ret,mdata_week_ret]
    df_concat = pd.concat(frames, axis =1)
    covar = df_concat.cov().iloc[1][0]
    var = mdata_week_ret.var()
    BetaL = covar/var
    std_dev_eqdata = eqdata_week_ret.std()
    std_dev_mdata = mdata_week_ret.std()
    r_sqrd = (covar/(std_dev_eqdata * std_dev_mdata))**2
    return BetaL,r_sqrd


def equity(ticker):
    eqdata = yf.download(ticker,period = "2Y" )
    eqdata = eqdata['Close']
    shs_outstding = yf.Ticker(ticker).info['sharesOutstanding']
    price = eqdata.loc[pd.to_datetime('2022-3-03')]
    equity = (shs_outstding *price)/(10**9)
    return equity

def return_equity(beta,rf,mktprm):
    re = rf + beta*(mktprm)
    re_round = round(re,2)
    return re

def return_debt(mkt_debt,ann_interest):
    rd = (ann_interest/mkt_debt) *100
    rd_round = round(rd,2)
    return rd
    

BetaLev = BetaL(ticker,mkt)[0]     
r_2 = BetaL(ticker,mkt)[1]

debt_equity_ratio = totaldebt/equity(ticker)

wd = debt_equity_ratio/(1+debt_equity_ratio)
we = 1- wd
wacc = (return_debt(totaldebt,ann_interest)*(1-taxrate)*wd) + (rf+(BetaLev*(mktprm)) * we)


BetaU = BetaLev/(1+(debt_equity_ratio*(1-taxrate)))
new_BetaLev = BetaU*(1 +(new_d_to_e*(1-taxrate)))
new_wd =new_d_to_e/(1+new_d_to_e)
new_we = 1-wd
new_rd = return_debt(totaldebt,ann_interest)+increase_rd
new_wacc  = (new_rd*(1-taxrate)*wd) + (rf+(new_BetaLev*(mktprm)) * we)

print('Beta ',BetaLev, 'R^2',r_2)
print('Return on Equity: ',round(return_equity(BetaLev,rf,mktprm),3) )
print('Equity =',equity(ticker), 'Billion')
print('Return on debt - pre-tax', round(return_debt(totaldebt,ann_interest),3))
print('D/E = ', debt_equity_ratio)
print('WACC = ', wacc)
print('new WACC =',new_wacc)