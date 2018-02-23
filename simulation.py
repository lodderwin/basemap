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

# Define the instruments to download
tickers = ['AAPL', 'NTAP', 'MSFT','BIDU','TRIP','AMAG','QCOM', 'CSCO', 'BA', 'AAOI', 'GPOR', 'AEG']
# Create Historic data
df = yr.finance_data(tickers=tickers).getData()
# Store data to csv
df.to_csv('./csv/store_data.csv')

#%%
# What is this? number of days used as input
seq_len = 5  
# What are these for? configuring model
model_layers = [1,3,16,1]
# attempts at what? #attempts at finding the correct model
attempts = 10

#what stock to invest in?
highest_increase_dct = {}

df_not_sufficient = {}
for stock in tickers:
    #reset for each stock
    best_model = 'shit'
    profit = 0.0
    check = df[(df['ticker']==stock)]
    if check.empty:
        df_not_sufficient[stock + '_no_data'] = 'data extraction failed'
        continue    
        
    data = check['close'].tolist()
    
    data = [x for x in data if str(x) != 'nan']
    if len(data)<2500.0:
        df_not_sufficient[stock + '_short_on_data'] = len(data)
        continue    
    x_train, y_train, x_test, y_test, y_test_correction =  lf.create_sets(data,seq_len,True)
    model = lf.build_model(model_layers)
    for i in range(int(attempts)):
        model.fit(
                x_train,
                y_train,
                batch_size=128,
                nb_epoch=1,
                validation_split=0.05)
        days_ahead = 5
        predicted_test = lf.predict_test(days_ahead, x_test, seq_len, model)
        predictions_in_function = int(x_test.shape[0]/seq_len)
        corrected_predicted_test = lf.correct_predict_test(seq_len, predicted_test, y_test_correction)
        turnover, investments = lf.invest_sim(corrected_predicted_test, y_test_correction)
        if turnover>profit:
            best_model = model
            profit = turnover
            print(turnover)
    if turnover<1000.0: 
        df_not_sufficient[stock+'_low_turnover'] = turnover
        continue
        
    predicted_test = lf.predict_test(days_ahead, x_test, seq_len, best_model)
    corrected_predicted_test = lf.correct_predict_test(seq_len, predicted_test, y_test_correction)
    lf.plot_results(y_test_correction, corrected_predicted_test, days_ahead)
    ##final and last prediction
    current_prediction = lf.predict_current(seq_len,days_ahead, x_test[-1], model)
    highest_increase_dct[stock]= current_prediction[-1]-current_prediction[0]
    lf.plot_current(x_test[-1],current_prediction,stock)
#%%
df_not_sufficient = pd.DataFrame(data=df_not_sufficient, index=[0]) 
df_not_sufficient.to_csv('not_good_enough.csv')
print(max(highest_increase_dct.items(), key=operator.itemgetter(1))[0])
    
    
    
    
    
