# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:31:47 2018

@author: Frank
"""
import pandas as pd
import yahoo_reader
import lstm_utils as utils
import random

#%% usual stuff from create_models.py
user = utils.load_user_from_yml(yml_file='./configs/user_settings.yml')
user_tickers = utils.get_tickers_for_a_user(user=user)
tickers_done = utils.get_tickers_done('./results/')
tickers_to_do = [ticker for ticker in user_tickers if ticker not in tickers_done]
tickers_to_do = tickers_to_do[:1]
#%% 

def get_same_industry_tickers(ticker, samplesize):
    
    #create usefull string for ticker    
    ticker = str(ticker)
    for character in ["[","]","'"]:
        if character in str(ticker):
            ticker = ticker.replace(character,"")
    
    #create list of all tickers in same industry
    df_alltickers = pd.read_csv('./tickers.csv')
    industry = df_alltickers[df_alltickers.ticker == ticker].iloc[0]['industry']
    industrytickers = df_alltickers[df_alltickers.industry == industry]['ticker'].tolist()
    
    #select random sample of same-industry tickers
    sameindustrytickers = random.sample(industrytickers, samplesize)
    
    #create dataframe
    df_sameindustrytickers = pd.DataFrame()

    #dowload and append data for each ticker (one-by-one to avoid Frank's error)
    for sit in sameindustrytickers:
        sit = [sit]
        yr = yahoo_reader.finance_data(tickers=sit)
        df, tickers = yr.get_fix_yahoo_data()
        df_sameindustrytickers = df_sameindustrytickers.append(df, ignore_index=True)
    
    return df_sameindustrytickers

#%%


