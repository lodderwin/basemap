import datetime as dt
import json

import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import gc

user = utils.load_user_from_yml(yml_file='./configs/user_settings.yml')
user_tickers = utils.get_tickers_for_a_user(user=user)
tickers_done = utils.get_tickers_done('./results/')
tickers_to_do = [ticker for ticker in user_tickers if ticker not in tickers_done]

yr = yahoo_reader.finance_data(tickers=tickers_to_do)
df_main, tickers = yr.get_fix_yahoo_data()

days_ahead = 1
#%%
for ticker in tickers:
    df_results = utils.read_a_user_results_csv(directory='./results/', user=user)
    window_length = 16
    df = df_main[df_main.ticker == ticker].reset_index(drop=True)
    df['volume'] = df['volume'].replace(0,1.0)
    df_p = pp.pre_process_data(df, window_length=window_length)
    
    close_nmd_array = utils.series_to_ndarray(df_p, window_length, column='close_nmd')
    volumne_nmd_array = utils.series_to_ndarray(df_p, window_length, column='volume_nmd')
    high_nmd_close_array = utils.series_to_ndarray(df_p, window_length, column='high_nmd_close')
    low_nmd_close_array = utils.series_to_ndarray(df_p, window_length, column='low_nmd_close')
    open_nmd_close_array = utils.series_to_ndarray(df_p, window_length, column='open_nmd_close')
    day_number_array = utils.series_to_ndarray(df_p, window_length, column='day_number')
    dates_array = utils.series_to_ndarray(df_p, window_length, column='date')
    windows_array = utils.series_to_ndarray(df_p, window_length, column='window')
    
    #closing price must go first 
    combined_input = np.concatenate((close_nmd_array,open_nmd_close_array,low_nmd_close_array,high_nmd_close_array,volumne_nmd_array, day_number_array),axis=2)
    x_train, y_train, x_test, y_test, train_days, test_days, test_windows,train_windows_non_randomized,x_train_sim = utils.train_test_split(combined_input,combined_input.shape[2], dates_array, windows_array)
    investment, best_investment_dev, params, margin, mcr = lstm_model.randomised_model_config(
        test_windows,
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
        iterations=20
    )
    
    gc.collect()
    
    # In params batch size is np.int64 and needs to be int in order to be json dumped
    params = {k : int(v) for k, v in params.items()}
    
    if (investment/300) > 1.00:
        df_temp = pd.DataFrame({
            'margin': [np.round(margin, 2)],
            'window_length': [window_length],
            'mcr': [np.round(mcr, 2)],
            'ticker': [ticker],
            'params': [json.dumps(params)],
            'date_created': [dt.datetime.now()]
        })

    else: 
        df_temp = pd.DataFrame({
            'margin': [np.NaN],
            'window_length': [np.NaN],
            'mcr': [np.NaN],
            'ticker': [ticker],
            'params': [json.dumps(params)],
            'date_created': [dt.datetime.now()]
        })  

    
    df_results = pd.concat([df_results, df_temp])
    df_results.to_csv('./results/model_results_{}.csv'.format(user), index=False)