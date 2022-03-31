# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:05:55 2022

@author: Owner
"""

BL = 1.2
taxrate = .25
debt_equityratio = .30
new_debt_equityratio = .65


rd = 3.7
rf = 2.0
cohen_mod = (rd/rf)*(debt_equityratio)
new_cohen_mod = (rd/rf)*(new_debt_equityratio)

#Hamada
Beta_U =BL/(1+debt_equityratio*(1-taxrate))

Beta_L = Beta_U*(1+new_debt_equityratio*(1-taxrate))

print('Beta L at new D/E', Beta_L)

#Cohen
Beta_U =BL/(1+cohen_mod*(1-taxrate))

Cohen_Beta_L = Beta_U*(1+new_cohen_mod*(1-taxrate))

print('Beta L at new D/E', Cohen_Beta_L)
