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

model_folder = './short_term_models/'

#volatile_tickers = pd.read_csv('Tickers.csv',sep=';')
#volatile_tickers_list = volatile_tickers['ticker'].tolist()
#volatile_tickers_done = pd.read_csv('tickers_done_short_correct.csv')
#volatile_tickers_done_list = volatile_tickers_done['tickers'].tolist()
#volatile_tickers_to_complete = [item for item in volatile_tickers_list if item not in volatile_tickers_done_list]
yr = yahoo_reader.finance_data(tickers=['DLTR', 'IFON'])
df_main = yr.get_fix_yahoo_data()
#%%
df_main = df_main[0]
run_model=True
days_ahead=1
df_test = {}
short_term_folder = './short_term_models/'
compare_investment = 300.
for ticker in volatile_tickers_to_complete[:20]:
    initial_investment = 50.0
#    window_size = 12
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
                                                        iterations=10)
        gc.collect()    
    if (investment/compare_investment)>1.00 :        
        volatile_tickers_done = pd.read_csv('tickers_done_short_correct.csv')
        volatile_tickers_done_lst = volatile_tickers_done['tickers'].tolist()
        volatile_tickers_done_lst.append(ticker)
        volatile_tickers_done_lst_margin = volatile_tickers_done['margin'].tolist()
        volatile_tickers_done_lst_margin.append(margin)
        volatile_tickers_done_lst_window_length = volatile_tickers_done['window_length'].tolist()
        volatile_tickers_done_lst_window_length.append(window_length)
        volatile_tickers_done_lst_mcr = volatile_tickers_done['mcr'].tolist()
        volatile_tickers_done_lst_mcr.append(mcr)
        df_temp = pd.DataFrame({'tickers':volatile_tickers_done_lst,'window_length':volatile_tickers_done_lst_window_length,'margin':volatile_tickers_done_lst_margin,'mcr':volatile_tickers_done_lst_mcr})
        df_temp.to_csv('tickers_done_short_correct.csv')
    elif (investment/compare_investment)<1.00 :  
        volatile_tickers_done = pd.read_csv('tickers_done_short_correct.csv')
        volatile_tickers_done_lst = volatile_tickers_done['tickers'].tolist()
        volatile_tickers_done_lst.append(ticker)
        volatile_tickers_done_lst_margin = volatile_tickers_done['margin'].tolist()
        volatile_tickers_done_lst_margin.append(np.NaN)
        volatile_tickers_done_lst_window_length = volatile_tickers_done['window_length'].tolist()
        volatile_tickers_done_lst_mcr = volatile_tickers_done['mcr'].tolist()
        volatile_tickers_done_lst_mcr.append(np.NaN)
        volatile_tickers_done_lst_window_length.append(np.NaN)
        df_temp = pd.DataFrame({'tickers':volatile_tickers_done_lst,'window_length':volatile_tickers_done_lst_window_length,'margin':volatile_tickers_done_lst_margin,'mcr':volatile_tickers_done_lst_mcr})
        df_temp.to_csv('tickers_done_short_correct.csv')
    
    
    
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