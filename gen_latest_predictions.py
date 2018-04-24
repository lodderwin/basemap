import pandas as pd
import numpy as np

import yahoo_reader
from numpy import newaxis
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import pygmail
#%%
#yr = yahoo_reader.finance_data(tickers=['FSTR', 'INOD', 'HBIO', 'EXTR', 'LINK', 'ACH', 'BASI', 'BCRX', 'BELFA', 'CAMT', 'CBI','BBGI'])
#df_main = yr.get_data()
tickers = ['TENX', 'DAIO','AMAG', 'MOSY']
ticker_dict = {}
df_main = pd.read_csv('saved_stock_data.csv')
# Prep data for LSTM model
window_length = 16
days_ahead=1
df_test = {}
#epochs
for ticker in tickers:
    df = df_main[df_main.ticker == ticker].reset_index(drop=True)
    df['volume'] = df['volume'].replace(0,1.0)
    df_p = pp.pre_process_data(df, window_length=window_length)
    close_nmd_array = utils.series_to_ndarray(df_p,window_length, column='close_nmd')
    volumne_nmd_array = utils.series_to_ndarray(df_p,window_length, column='volume_nmd')
    high_nmd_array = utils.series_to_ndarray(df_p,window_length, column='high_nmd')
    low_nmd_array = utils.series_to_ndarray(df_p,window_length, column='low_nmd')
#    day_number_array = utils.series_to_ndarray(df_p,window_length, column='day_number')
    dates_array = utils.series_to_ndarray(df_p,window_length, column='date')
    windows_array = utils.series_to_ndarray(df_p,window_length, column='window')
    #closing price must go first 
    combined_input = np.concatenate((close_nmd_array,volumne_nmd_array,high_nmd_array,low_nmd_array),axis=2)
    x_train, y_train, x_test, y_test,  train_days, test_days, test_windows = utils.train_test_split(combined_input,combined_input.shape[2], dates_array, windows_array)
#%%
    model, investment, best_investment_dev = lstm_model.randomised_model_config(test_windows,
                                                    df_p,
                                                    test_days,
                                                    combined_input.shape[2],
                                                    window_length,
                                                    ticker,
                                                    df,
                                                    days_ahead,
                                                    x_train,
                                                    y_train, 
                                                    x_test,
                                                    y_test,
                                                    iterations=10,
                                                    epochs=10)
    ### predict tomorrow  
    window_length_pred = window_length - 1
    df_pred = pp.pre_process_data(df[-15:], window_length=window_length_pred)
    df_pred = df_pred.reset_index(drop=True)
    close_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='close_nmd')
    volumne_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='volume_nmd')
    high_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='high_nmd')
    low_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='low_nmd')
    combined_input_pred = np.concatenate((close_nmd_array_pred,volumne_nmd_array_pred,high_nmd_array_pred,low_nmd_array_pred),axis=2)
    predicted_normalised = (model.predict(combined_input_pred)[0,0])
    predicted = (predicted_normalised + 1.0)*df_pred.loc[0,'normaliser']
    growth = predicted/df_pred.loc[len(df_pred)-1,'close']
    ticker_dict[ticker] = growth
    df_test[ticker] = best_investment_dev
    

    
    
    
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