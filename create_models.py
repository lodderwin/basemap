import os

import pandas as pd
import numpy as np
from numpy import newaxis

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import gc
#import pygmail
from keras.models import Sequential, load_model

industry = 'brewers'
results = './'+industry+'/results/'
tickers = './'+industry+'/tickers/'
volatile_tickers = pd.read_csv(tickers+industry+'.csv',sep=',')
#volatile_tickers= volatile_tickers.reset_index()
volatile_tickers_list = volatile_tickers['Ticker'].tolist()


shortterm_models = './'+industry+'/shortterm_models/'

#volatile_tickers_to_complete = [item for item in volatile_tickers_list if item not in volatile_tickers_done_list]
yr = yahoo_reader.finance_data(tickers=volatile_tickers_list)
df_main = yr.get_fix_yahoo_data()
df_main = df_main[0]
days_ahead=1
df_test = {}
compare_investment = 300.
for ticker in volatile_tickers_list:
    for window_length in [16]:
        df = df_main[df_main.ticker == ticker].reset_index(drop=True)
        df['volume'] = df['volume'].replace(0,1.0)
        df_p = pp.pre_process_data(df, window_length=window_length)
        close_nmd_array = utils.series_to_ndarray(df_p,window_length, column='close_nmd')
        volumne_nmd_array = utils.series_to_ndarray(df_p,window_length, column='volume_nmd')
        high_nmd_close_array = utils.series_to_ndarray(df_p,window_length, column='high_nmd_close')
        low_nmd_close_array = utils.series_to_ndarray(df_p,window_length, column='low_nmd_close')
        open_nmd_close_array = utils.series_to_ndarray(df_p,window_length, column='open_nmd_close')
        day_number_array = utils.series_to_ndarray(df_p,window_length, column='day_number')
        dates_array = utils.series_to_ndarray(df_p,window_length, column='date')
        windows_array = utils.series_to_ndarray(df_p,window_length, column='window')
        #closing price must go first 
        combined_input = np.concatenate((close_nmd_array,open_nmd_close_array,low_nmd_close_array,high_nmd_close_array,volumne_nmd_array, day_number_array),axis=2)
        x_train, y_train, x_test, y_test, train_days, test_days, test_windows,train_windows_non_randomized,x_train_sim = utils.train_test_split(combined_input,combined_input.shape[2], dates_array, windows_array)
#%%
        investment, best_investment_dev,params,margin,mcr = lstm_model.randomised_model_config(test_windows,
                                                        df_p,
                                                        test_days,
                                                        train_days,
                                                        train_windows_non_randomized,
                                                        x_train_sim,
                                                        combined_input.shape[2],
                                                        window_length,
                                                        ticker,
                                                        df,
                                                        days_ahead,
                                                        x_train,
                                                        y_train, 
                                                        x_test,
                                                        y_test,
                                                        industry,
                                                        iterations=1)
        #%%
        gc.collect()    
        if (investment/compare_investment)>1.00 :  
            volatile_tickers = pd.read_csv(tickers+industry+'.csv',sep=',')
            volatile_tickers['margin'][volatile_tickers.loc[volatile_tickers['Ticker'] == ticker].index[0]] = margin
            volatile_tickers['window_length'][volatile_tickers.loc[volatile_tickers['Ticker'] == ticker].index[0]] = window_length
            volatile_tickers['mcr'][volatile_tickers.loc[volatile_tickers['Ticker'] == ticker].index[0]] = mcr
            volatile_tickers.to_csv(tickers+industry+'.csv')
        elif (investment/compare_investment)<1.00 :  
            volatile_tickers = pd.read_csv(tickers+industry+'.csv',sep=',')
            volatile_tickers['margin'][volatile_tickers.loc[volatile_tickers['Ticker'] == ticker].index[0]] = np.NaN
            volatile_tickers['window_length'][volatile_tickers.loc[volatile_tickers['Ticker'] == ticker].index[0]] = np.NaN
            volatile_tickers['mcr'][volatile_tickers.loc[volatile_tickers['Ticker'] == ticker].index[0]] = np.NaN
            volatile_tickers.to_csv(tickers+industry+'.csv')
            
    
df.loc[df['favorite_color'] == 'yellow']
#    df_test[ticker] = best_investment_dev
    

    
    
    
    #%%

#    
#    
#    predictions_nmd = lstm_model.predict(model, , days_ahead)
#    predictions = (predictions_nmd + 1) * X_1[0][0]
#    prediction_single_day = lstm_model.predict_single(model, combined_input, days_ahead)
#    # Calculate predicted growth over 5 days
#    growth = np.round(((predictions[0][4] / X_1[0][4]) -1) * 100, 2)[0]
#    
#    # Plot predictions
#    plt = plotting.plot_latest_prediction(days_ahead,df, predictions, ticker, growth, mse,
#                                          model, df_p,days_ahead)
    
    # Add predicted growth to ticker_dict
#    ticker_dict[ticker] = growth

# Compose and send email
#subject, body, attachments = pygmail.compose_email(expected_deltas=ticker_dict)
#pygmail.send_mail(subject=subject,
#                  attachments=attachments, 
#                  body=body)