#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:25:48 2018

@author: erwinlodder
"""
from datetime import datetime
import pandas as pd
df = pd.read_csv('NTAP.csv', sep=',')
#%%
df = df[['Date','Close']]
fmt = '%Y-%m-%d'
df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, fmt))
#df = df[10:]

#%%
df.to_csv('understandntap.csv', index=False, header=True)