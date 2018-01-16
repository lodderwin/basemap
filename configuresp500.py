#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:25:48 2018

@author: erwinlodder
"""

import pandas as pd
df = pd.read_csv('^GSPC (3).csv', sep=',')
#%%
df = df[['Close']]
df = df[40:]

#%%
df.to_csv('sp500.csv', index=False,header=None)