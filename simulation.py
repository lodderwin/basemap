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
import pygmail
import yaml
# Define the instruments to download
tickers = ['AAPL', 'NTAP', 'MSFT','BIDU','ASML','AMAG'
          ,'QCOM', 'CSCO', 'BA', 'AAOI', 'GPOR', 'AEG'
          , 'TENX', 'VICR', 'WK', 'ANIK', 'FNGN', 'MNGA']
dutch_tickers = ['ASML', 'HEI', 'KPN', 'UN', 'AKZO', 'AF', 'AMG', 'ASM', 'MT', 'BRNL', 'BINCK','BOKA', 'CLB', 'DSM', 'INGA',
                 'HYDRA', 'IMCD', 'NN', 'PNL', 'RAND', 'RDSA','UNCP7', 'VOPAK', 'WDP' ]
fluc_stocks = ['EMMS', 'RAS', 'DTRM', 'INT'
              , 'QUMU', 'TSD', 'OLED', 'ITGR', 'AMAG', 'MOSY','TENX', 'BA', 'BIDU', 'MNGA', 'ADMP', 'VICR', 'DAIO', 'NEON', 'MDCA', 'NTAP','QCOM']
# Create Historic data
new_volatile_stocks = ['IFON', 'AUTO', 'DXR', 'CHRS', 'SNMX', 'AMWD', 'SMRT', 'BOOM', 'UUU', 'BRID','SCX', 'VISI', 'PDLI','BKYI', 'GEN', 'GALT','BIG', 'BFLS', 'INFI',
                       'CECE', 'INSY', 'FIZZ', 'MGEN', 'UTSI', 'OMEX', 'IPAR']
#df = pp.preProcessData(df)
promising_stocks = ['AMAG', 'ADMP', 'DAIO', 'MOSY', 'OLED', 
                     'TENX', 'BKYI', 'GALT', 'GEN', 'IFON', 
                    'INFI', 'INSY', 'OMEX', 'SMRT', 'SNMX', 'UTSI',
                    'UUU', 'ABEO', 'AA','ACY', 'ACHV', 'ACLS', 'AEMD, AEZS', 
                    'AEHR', 'AGEN', 'AGM', 'AHPI', 'AIRI', 'AKS' , 'ALQA', 
                     'AMD', 'AMSC', 'ANF', 'AOI', 'AP', 'ARCB', 'ARDM',
                    'ARL', 'ARLZ', 'ARNA', 'ARQL', 'ARRY', 'ARTW', 'ARWR', 'BOOM', 'CRNT', 'AVD',
                    'ASFI', 'ASNA', 'ASTC', 'ASUR' , 'ASYS', 'ATLC', 'AXAS',
                    'ALSK', 'BASI', 'BCRX', 'CAMT', 'CENX', 'CGEN', 'CLWT', 
                      'CRZO', 'DWSN', 'AXTI', 'ELTK', 'ESIO', 'EXAS',
                    'FSTR', 'INOD', 'HBIO', 'EXTR', 'LINK', 'ACH', 'BASI', 'BCRX', 'BELFA', 'CAMT', 'CBI','BBGI',
                      'CGEN', 'CGI', 'CHKE', 'CRR', 'CVTI', 'CYBE', 'CYH', 'DO', 'DRRX', 'ELTK', 'ESIO', 'EXTR', 'EXEL']
                    

#add avd boom crnt
current = ['ASTC', 'ACY']
df_tickers = pd.read_csv('volatilestocks.csv', sep=';')
#200 done
tickers_lst = df_tickers['ticker'].tolist()[150:200]
tickers_lst = list(set(tickers_lst))

df = yr.finance_data(tickers=tickers_lst).get_data()
#df.to_csv('saved_stock_data_1.csv')
#df = pd.read_csv('saved_stock_data_1.csv')

#%%

#%% select approppiate stocks
#df_volatile_stocks = pd.read_csv('volatile_stocks.csv' , encoding='latin-1')
#def choose_stocks(df_volatile_stocks):
#    volatile_stocks_list = []
#    for length in [2,3,4]:
#        for index, row in b.iterrows():
#            volatile_stocks_list.append(row['Stocks'][:length])
#    return volatile_stocks_list
#volatile_stocks_list = choose_stocks(df_volatile_stocks)   
#df = yr.finance_data(tickers=volatile_stocks_list).getData()
##%%
#volatile_stocks_list_correct = []
#number = 1
#for stock in volatile_stocks_list:
#    print(number)
#    number=number+1
#    check = df[(df['ticker']==stock)]
#    if check.empty:
#        continue    
#        
#    data = check['close'].tolist()
#    
#    data = [x for x in data if str(x) != 'nan']
#    if len(data)<2000.0:
#        continue
#    volatile_stocks_list_correct.append(stock)
# #%%
#volatile_stocks_list_correct_correct = []
#for stock in volatile_stocks_list_correct: 
#    if len(stock)>2:
#        volatile_stocks_list_correct_correct.append(stock)

        

#%%
# What is this? number of days used as input
seq_len = 5
# What are these for? configuring model
model_layers = [1,5,16,1]
# attempts at what? #attempts at finding the correct model
attempts = 3

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
dct_promising = {}
for stock in tickers_lst:
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
    for lstm_layer_1 in [10,15]:
        for lstm_layer_2 in [25,40]:
            for batch_size in [32]:
                for epoch in [1]:
                    model = lf.build_model([1,lstm_layer_1,lstm_layer_2,1])
                 
                    for k in range(int(attempts)):
                        print(stock)
                        model.fit(
                                x_train,
                                y_train,
                                batch_size=batch_size,
                                nb_epoch=epoch,
                                validation_split=0.05)
                        days_ahead = 5
                        predicted_test_temp = lf.predict_test(days_ahead, x_test, seq_len, model)
                        predictions_in_function = int(x_test.shape[0]/days_ahead)
                        corrected_predicted_test_temp = lf.correct_predict_test(days_ahead, predicted_test_temp, y_test_correction, seq_len)
                        turnover, investments, investment_dev = lf.invest_sim(corrected_predicted_test_temp, y_test_correction)
                        print(turnover,profit,'model:'+str(lstm_layer_1) + ' ' + str(lstm_layer_2))
                        if turnover>profit:
                            investment_curve = investment_dev
                            best_model = model
                            profit = turnover
                            ##
                            predicted_test = lf.predict_test(days_ahead, x_test, seq_len, model)
                            predicted_test_day = lf.predict_test_day(days_ahead, x_test, seq_len, model)
                            current_prediction,compare_value = lf.predict_current(seq_len,days_ahead, data[-5:], model)
                            df_profits[stock] = profit
    #                        lf.save_model(stock, model)
                        
                    ##

    if profit<1000.0: 
        df_not_sufficient[stock+'_low_turnover'] = turnover
        continue
#    df_profits[stock] = profit
#    predicted_test = lf.predict_test(days_ahead, x_test, seq_len, best_model)
#    predicted_test_day = lf.predict_test_day(days_ahead, x_test, seq_len, best_model)
#    current_prediction = lf.predict_current(seq_len,days_ahead, x_test[-4:], best_model)
#    corrected_predicted_test = lf.correct_predict_test(days_ahead, predicted_test, y_test_correction,seq_len)
#    corrected_predicted_test_day = lf.correct_predict_test_day(days_ahead, predicted_test_day, y_test_correction,seq_len)
#    lf.plot_results(y_test_correction, corrected_predicted_test, days_ahead,stock)
#    lf.plot_results_day(y_test_correction, corrected_predicted_test, days_ahead,stock,corrected_predicted_test_day)
#    lf.plot_investment(investment_curve, stock)
#    ##final and last prediction
#    current_prediction_correction = lf.predict_current_corrected(current_prediction, y_test_correction, seq_len)
#    highest_increase_dct[stock]= current_prediction_correction[-1]-y_test_correction[-1]
#    dct_predictions[stock] =  current_prediction_correction
#    best_model = lf.open_model(stock)
    
                        
    corrected_predicted_test = lf.correct_predict_test(days_ahead, predicted_test, y_test_correction,seq_len)
    corrected_predicted_test_day = lf.correct_predict_test_day(days_ahead, predicted_test_day, y_test_correction,seq_len)
    lf.plot_results(y_test_correction, corrected_predicted_test, days_ahead,stock)
    lf.plot_results_day(y_test_correction, corrected_predicted_test, days_ahead,stock,corrected_predicted_test_day)
    lf.plot_investment(investment_curve, stock)
    ##final and last prediction
    current_prediction_correction = lf.predict_current_corrected(current_prediction, y_test_correction, seq_len)
    highest_increase_dct[stock]= current_prediction[-1]-compare_value
    dct_predictions[stock] =  current_prediction_correction

    #load
    if current_prediction_correction[-1]>current_prediction_correction[0] and current_prediction_correction[-1]>y_test_correction[-1]:
        dct_promising[stock] = current_prediction_correction
    lf.plot_current(y_test_correction[-30:],current_prediction_correction,stock)

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
print(dct_promising.keys())    
#train on multiple stocks for 10-20 day predictions
  # compare predictions compare prediction  
    

#%%
# Create email body
subject, body, attachments = pygmail.compose_email(expected_deltas=highest_increase_dct)

# Send email
pygmail.send_mail(subject=subject,
          attachments=attachments,
          body=body)
#%%
def top_x_of_dict(dictionary, x):
    """Returns the top x items of given dictionary.
    
    Parameters
    --------
    dictionary : Dictionary you would like to reduce to top x only, dict
    x : The number of items you would like to return, int
    
    Returns
    --------
    dictionary : Dictionary reduced to top x items, dict
    """
    # Sort dict keys from highest to lowest
    keys = sorted(dictionary, key=dictionary.get, reverse=True)
    # Select top x keys
    keys = keys[:x]
    # Reduce dict to only those in keys
    dictionary = {key : value for key, value in dictionary.items() if key in keys}
    
    return dictionary

#%%
top_x_of_dict(df_profits, 15)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
