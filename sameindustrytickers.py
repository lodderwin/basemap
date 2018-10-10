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

#function returns the data for a sample of 'samplesize' tickers, from the same industry
#which have had the most similar volatility over the last 'window' number of days

def get_same_industry_similar_volatility_tickers(ticker, window, samplesize):
    
    #download data for ticker A
    yr = yahoo_reader.finance_data(tickers=[ticker])
    dfA, tickersA = yr.get_fix_yahoo_data()    
    
    #calculate price and volatility (average standarddeviation over window) and save it for the last day
    dfA['volatility'] = dfA.close.rolling(window).std()
    volatilityA = dfA.volatility.tail(1)

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
    
    #main dataframe and dataframe for last day of dataframe
    df_industrytickers = pd.DataFrame()
    df_tails = pd.DataFrame()
    
    #dowload and append data for each ticker (one-by-one to avoid Frank's error)
    for sit in industrytickers:
        try:
            print("Downloading data for " + str(number) + " same-industry tickers.")
            sit = [sit]
            yr = yahoo_reader.finance_data(tickers=sit)
            df, tickers = yr.get_fix_yahoo_data()
            
            #calculate volatility
            df['volatility'] = df.close.rolling(window).std()
            
            #take the last day of dataframe and compare volatility with that of ticker A
            df_tail = df.tail(1)
            volatility = df_tail.volatility.tail(1)
            df_tail['vol_diff'] = abs(volatilityA - volatility)
            df_industrytickers = df_industrytickers.append(df, ignore_index=True)
            df_tails = df_tails.append(df_tail, ignore_index=True)
            number = number-1
            print("Ticker data added") 
            
        except: 
            number = number-1
            print("No data")  

    #sort the tickers by the difference in volatility compared to ticker A
    df_tails = df_tails.sort_values('vol_diff', ascending=True) 
    df_tails = df_tails.head(samplesize)
    sample = list(df_tails.ticker.unique())
    
    #make final dataframe with all data from stocks in sample
    df_sample = pd.DataFrame()
    for ticker in sample:
        df = df_industrytickers[df_industrytickers.ticker == ticker]
        df_sample = df_sample.append(df, ignore_index=True)

    return df_sample


#%%
#get_same_industry_similar_volatility_tickers(ticker='AA', window=3, samplesize=2) 




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
            df_largedecreases = df_largedecreases.append(df, ignore_index=True)
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
