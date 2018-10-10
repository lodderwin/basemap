# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:31:47 2018

@author: Frank
"""
import pandas as pd
import yahoo_reader
import lstm_utils as utils
import random
import numpy as np

#%% usual stuff from create_models.py
user = utils.load_user_from_yml(yml_file='./configs/user_settings.yml')
user_tickers = utils.get_tickers_for_a_user(user=user)
tickers_done = utils.get_tickers_done('./results/')
tickers_to_do = [ticker for ticker in user_tickers if ticker not in tickers_done]
#tickers_to_do = tickers_to_do[:1]
#%% 

#function returns a random sample of 'samplesize' tickers for each industry
#accounts for number of tickers already trained in an industry

def get_random_samples_per_industry(samplesize):
        
    #dataframe of all tickers and industries and list of all tickers
    df_tickers = pd.read_csv('./tickers.csv')
    
    #create one dataframe of tickers that are done (df_tickers_done) and one of tickers to do (df_tickers)
    df_tickers_done = pd.DataFrame()
    for ticker in tickers_done:
        df_tickerindustry = df_tickers[df_tickers.ticker == ticker]
        df_tickers_done = df_tickers_done.append(df_tickerindustry)
        df_tickers = df_tickers[df_tickers.ticker != ticker]
    
    industries = list(df_tickers.industry.unique())
    industrysamples = []
    
    for industry in industries:
        
        #check how much tickers still need to be trained for this industry
        stilltotrain = samplesize - len(df_tickers_done[df_tickers_done.industry == industry])  
        print(stilltotrain)
        
        #make dataframe and list of all tickers in the industry
        df_industrytickers = df_tickers[df_tickers.industry == industry]
        industrytickers = list(df_industrytickers.ticker.unique())
        
        #if there are enough tickers left in industry: take random sample, else: take all
        if len(industrytickers)>stilltotrain:
            industrytickers = random.sample(industrytickers, stilltotrain)  
        industrysamples.extend(industrytickers)
        
    return industrysamples

#%%
#get_random_samples_per_industry(samplesize=20)

#%%
def get_same_industry_tickers(ticker, samplesize):
    
    #create usefull string for ticker    
    ticker = str(ticker)
    for character in ["[","]","'"]:
        if character in str(ticker):
            ticker = ticker.replace(character,"")
    print(ticker)
    #create list of all tickers in same industry
    df_alltickers = pd.read_csv('./tickers.csv')
    industry = df_alltickers[df_alltickers.ticker == ticker].iloc[0]['industry']
    industrytickers = df_alltickers[df_alltickers.industry == industry]['ticker'].tolist()
    industrytickers.remove(ticker)

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
#get_same_industry_tickers(tickers_to_do,5)




#%%
def get_large_decreases_in_industry(ticker, percentage):
    
    #create usefull string for ticker    
    ticker = str(ticker)
    for character in ["[","]","'"]:
        if character in str(ticker):
            ticker = ticker.replace(character,"")

    #create list of all tickers in same industry
    df_alltickers = pd.read_csv('./tickers.csv')
    industry = df_alltickers[df_alltickers.ticker == ticker].iloc[0]['industry']
    industrytickers = df_alltickers[df_alltickers.industry == industry]['ticker'].tolist()
    industrytickers.remove(ticker)
    
    #for counter
    number = len(industrytickers)

    #main dataframe
    df_largedecreases = pd.DataFrame()
    
    #dowload and append data for each ticker (one-by-one to avoid Frank's error)
    for sit in industrytickers:
        try:
            print("Downloading data for " + str(number) + " same-industry tickers.")
            sit = [sit]
            yr = yahoo_reader.finance_data(tickers=sit)
            df, tickers = yr.get_fix_yahoo_data()
            
            #calculate decrease and select 100 rows above and 50 rows below rows where this decrease is larger than 'percentage'
            df['perc_change'] = df.close/df.open
            df['arounddecrease'] = 0
            for X in list(range(-100,50)):
                df['arounddecrease'] = np.where(df.perc_change.shift(X)<percentage, 1, df.arounddecrease)
            
            #append ticker data to main dataframe
            df_largedecreases = df.append(df, ignore_index=True)
            number = number-1
            print("Ticker data added") 
        except: 
            number = number-1
            print("No data")  
    
    df_largedecreases = df_largedecreases[df_largedecreases.arounddecrease == 1]
    df_largedecreases = df_largedecreases.drop(columns=['arounddecrease', 'perc_change'])
    
    return df_largedecreases
#%%
#df_largedecreases = get_large_decreases_in_industry(tickers_to_do, 0.95)
