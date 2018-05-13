import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import pygmail
volatile_tickers = pd.read_csv('volatile_complete.csv')
dvolatile_tickers_list = volatile_tickers['tickers'].tolist()
volatile_tickers_done = pd.read_csv('tickers_done_long.csv')
volatile_tickers_to_complete = [item for item in volatile_tickers_list if item not in volatile_tickers_done]
#%%
yr = yahoo_reader.finance_data(tickers=volatile_tickers_to_complete)
df_main = yr.get_data()

ticker_dict = {}
#df_main = pd.read_csv('saved_stock_data.csv')

for ticker in volatile_tickers_to_complete:
    for window_length in [50,100]:
        for days_average in [5,10]:
            df = df_main[df_main.ticker == ticker].reset_index(drop=True)
            df_p = pp.pre_process_data_average(df,days_average, window_length=window_length)
        
        #%%
        #    df_t = pp.pre_process_data(df,window_length=5)
            # Split closing price into test and train
            #define correct length
            close_nmd_array = utils.series_to_ndarray(df_p, window_length+1, column='close_nmd')
        #    day_number_array = utils.series_to_ndarray(df_p,window_length+1, column='day_number')
            windows_array = utils.series_to_ndarray(df_p,window_length+1, column='window')
            high_nmd_array = utils.series_to_ndarray(df_p,window_length+1, column='high_nmd')
        #    volumne_nmd_array = utils.series_to_ndarray(df_p,window_length+1, column='volume_nmd')
            combined_input = np.concatenate((close_nmd_array,high_nmd_array),axis=2)
            dates_array = utils.series_to_ndarray(df_p,window_length+1, column='date')
            x_train, y_train, x_test, y_test,  train_days, test_days, test_windows = utils.train_test_split(combined_input,combined_input.shape[2], dates_array, windows_array)
        
        
        #    x_train, y_train, x_test, y_test, dates_test = utils.train_test_split(combined_input,2, dates_array)
        #%%
            # Build model
            model, investment, best_investment_dev = lstm_model.randomised_model_config_days_average(test_windows,
                                                            df_p,
                                                            test_days,
                                                            combined_input.shape[2],
                                                            window_length,
                                                            ticker,
                                                            df,
                                                            days_average,
                                                            x_train,
                                                            y_train, 
                                                            x_test,
                                                            y_test,
                                                            iterations=3,
                                                            epochs=5)
        
            # Create X based on last window in data (last window is 0)
        #%%    
            df_pred = pp.pre_process_data(df[-window_length:], window_length=window_length)
            df_pred = df_pred.reset_index(drop=True)
            close_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length, column='close_nmd')
            volumne_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length, column='volume_nmd')
            high_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length, column='high_nmd')
            low_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length, column='low_nmd')
            day_number_array_pred = utils.series_to_ndarray(df_pred,window_length, column='day_number')
        
            combined_input_pred = np.concatenate((close_nmd_array_pred,high_nmd_array_pred),axis=2)
            predicted_normalised = (model.predict(combined_input_pred)[0,0])
            predicted = (predicted_normalised + 1.0)*df_pred.loc[0,'normaliser']
            growth = predicted/df_pred.loc[len(df_pred)-1,'close']
            ticker_dict[ticker] = growth
            volatile_tickers_done = pd.read_csv('tickers_done_long.csv')
            volatile_tickers_done_lst = volatile_tickers_done['tickers'].tolist()
            volatile_tickers_done_lst.append(ticker)
            df_temp = pd.DataFrame({'tickers':volatile_tickers_done_lst})
            df_temp.to_csv('tickers_done_long.csv')

# Compose and send email
#subject, body, attachments = pygmail.compose_email(expected_deltas=ticker_dict)
#pygmail.send_mail(subject=subject,
#                  attachments=attachments, 
#                  body=body)