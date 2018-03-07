#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 01:37:41 2018

@author: erwinlodder
"""
import matplotlib.pyplot as plt
import pandas as pd
import yahoo_reader as yr
import lstm_functions as lf
import operator
import datetime as dt
from pandas.tseries.offsets import BDay
import preprocessing as pp

# Define the instruments to download
tickers = ['AAPL', 'NTAP', 'MSFT','BIDU','ASML','AMAG'
          ,'QCOM', 'CSCO', 'BA', 'AAOI', 'GPOR', 'AEG'
          , 'TENX', 'VICR', 'WK', 'ANIK', 'FNGN', 'MNGA']
dutch_tickers = ['ASML', 'HEI', 'KPN', 'UN', 'AKZO', 'AF', 'AMG', 'ASM', 'MT', 'BRNL', 'BINCK','BOKA', 'CLB', 'DSM', 'INGA',
                 'HYDRA', 'IMCD', 'NN', 'PNL', 'RAND', 'RDSA','UNCP7', 'VOPAK', 'WDP' ]
fluc_stocks = ['EMMS', 'RAS', 'DTRM', 'INT'
              , 'QUMU', 'TSD', 'OLED', 'ITGR', 'AMAG', 'MOSY',
               'TENX', 'BA', 'BIDU', 'MNGA', 'ADMP', 'VICR', 'DAIO', 'NEON', 'MDCA', 'NTAP','QCOM']
# Create Historic data
#df = yr.finance_data(tickers=fluc_stocks).getData()
#df.to_csv('saved_stock_data.csv')

#df = pp.preProcessData(df)
#%%
df = pd.read_csv('saved_stock_data.csv')
#%% select approppiate stocks
df_volatile_stocks = pd.read_csv('volatile_stocks.csv' , encoding='latin-1')
def choose_stocks(df_volatile_stocks):
    volatile_stocks_list = []
    for length in [2,3,4]:
        for index, row in b.iterrows():
            volatile_stocks_list.append(row['Stocks'][:length])
    return volatile_stocks_list
volatile_stocks_list = choose_stocks(df_volatile_stocks)   
df = yr.finance_data(tickers=volatile_stocks_list).getData()
#%%
volatile_stocks_list_correct = []
number = 1
for stock in volatile_stocks_list:
    print(number)
    number=number+1
    check = df[(df['ticker']==stock)]
    if check.empty:
        df_not_sufficient[stock + '_no_data'] = 'data extraction failed'
        continue    
        
    data = check['close'].tolist()
    
    data = [x for x in data if str(x) != 'nan']
    if len(data)<2000.0:
        df_not_sufficient[stock + '_short_on_data'] = len(data)
        continue
    volatile_stocks_list_correct.append(stock)
    

#%%
# What is this? number of days used as input
seq_len = 5
# What are these for? configuring model
model_layers = [1,5,16,1]
# attempts at what? #attempts at finding the correct model
attempts = 5

#what stock to invest in?
highest_increase_dct = {}
#store insufficient stocks
df_not_sufficient = {}
#store profits
df_profits = {}
#store plots
dct_plots = {} 
#store predictions
dct_predictions = {}
dct_dates = {}
investment_curve = 0
for stock in fluc_stocks:
    #reset for each stock
    best_model = 'shit'
    profit = 0.0
    check = df[(df['ticker']==stock)]
    if check.empty:
        df_not_sufficient[stock + '_no_data'] = 'data extraction failed'
        continue    
        
    data = check['close'].tolist()
    
    data = [x for x in data if str(x) != 'nan']
    if len(data)<2000.0:
        df_not_sufficient[stock + '_short_on_data'] = len(data)
        continue    
    x_train, y_train, x_test, y_test, y_test_correction =  lf.create_sets(data,seq_len,True)
    model = lf.build_model(model_layers)
    for lstm_layer_1 in [10]:
        for lstm_layer_2 in [25]:
            model = lf.build_model([1,lstm_layer_1,lstm_layer_2,1])
         
            for k in range(int(attempts)):
                print(stock)
                model.fit(
                        x_train,
                        y_train,
                        batch_size=64,
                        nb_epoch=1,
                        validation_split=0.05)
                days_ahead = 5
                predicted_test = lf.predict_test(days_ahead, x_test, seq_len, model)
                predictions_in_function = int(x_test.shape[0]/days_ahead)
                corrected_predicted_test = lf.correct_predict_test(days_ahead, predicted_test, y_test_correction, seq_len)
                turnover, investments, investment_dev = lf.invest_sim(corrected_predicted_test, y_test_correction)
                if turnover>profit:
                    investment_curve = investment_dev
                    best_model = model
                    profit = turnover
                    print(turnover)
    if profit<1000.0: 
        df_not_sufficient[stock+'_low_turnover'] = turnover
        continue
    df_profits[stock] = profit
    
        
    predicted_test = lf.predict_test(days_ahead, x_test, seq_len, best_model)
    corrected_predicted_test = lf.correct_predict_test(days_ahead, predicted_test, y_test_correction,seq_len)
    lf.plot_results(y_test_correction, corrected_predicted_test, days_ahead,stock)
    lf.plot_investment(investment_curve, stock)
    ##final and last prediction
    current_prediction = lf.predict_current(seq_len,days_ahead, x_test[-4:], model)
    current_prediction_correction = lf.predict_current_corrected(current_prediction, y_test_correction)
    highest_increase_dct[stock]= current_prediction_correction[-1]-y_test_correction[-1]
    dct_predictions[stock] =  current_prediction
    
    dct_plots[stock] = lf.plot_current(y_test_correction[-40:],current_prediction_correction,stock)

#%%
from pandas.tseries.offsets import BDay

today = pd.datetime.today()
dates_list = []
for i in range(0,5,1):
    dates_list.append(today+BDay(i))

df_not_sufficient = pd.DataFrame(data=df_not_sufficient, index=[0]) 
df_not_sufficient.to_csv('not_good_enough.csv')
df_predictions = pd.DataFrame(data=dct_predictions) 
df_predictions['dates'] = dates_list
df_predictions.to_csv(dt.datetime.now().strftime("%Y-%m-%d") + 'predictions_fluc.csv')

#def compare(date_predictions, stock):
#    df = pd.read_csv('predictions_fluc.csv')
#    date_prediction = pd.datetime.
    
print(max(highest_increase_dct.items(), key=operator.itemgetter(1))[0])

#add multiple curves
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
