import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import pygmail
#%%
#yr = yahoo_reader.finance_data(tickers=['OMEX', 'TENX'])
#df_main = yr.get_data()
tickers = ['TENX', 'DAIO','AMAG', 'MOSY']
ticker_dict = {}
df_main = pd.read_csv('saved_stock_data.csv')
days_average = 5
ticker_dict = {}
# Prep data for LSTM model
window_length = 20
for ticker in ['TENX']:
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

    combined_input = np.concatenate((close_nmd_array,high_nmd_array),axis=2)
    dates_array = utils.series_to_ndarray(df_p,window_length+1, column='date')
    x_train, y_train, x_test, y_test,  train_days, test_days, test_windows = utils.train_test_split(combined_input,combined_input.shape[2], dates_array, windows_array)


#    x_train, y_train, x_test, y_test, dates_test = utils.train_test_split(combined_input,2, dates_array)
#%%
    # Build model
    model, mse = lstm_model.randomised_model_config_days_average(test_windows,
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
                                                    epochs=3)

    # Create X based on last window in data (last window is 0)
#%%    
    X_1, X_nmd_1 = utils.gen_X(df_p,'close',window_length, window=0)
    X_2, X_nmd_2 = utils.gen_X(df_p,'day_number', window_length, window=0)
    combined_input = np.concatenate((X_nmd_1,X_2),axis=2)
    
    predictions_nmd = lstm_model.predict(model, , days_ahead)
    predictions = (predictions_nmd + 1) * X_1[0][0]
    prediction_single_day = lstm_model.predict_single(model, combined_input, days_ahead)
    # Calculate predicted growth over 5 days
    growth = np.round(((predictions[0][4] / X_1[0][4]) -1) * 100, 2)[0]
    
    # Plot predictions
    plt = plotting.plot_latest_prediction(days_ahead,df, predictions, ticker, growth, mse,
                                          model, df_p,days_ahead)
    
    # Add predicted growth to ticker_dict
    ticker_dict[ticker] = growth

# Compose and send email
#subject, body, attachments = pygmail.compose_email(expected_deltas=ticker_dict)
#pygmail.send_mail(subject=subject,
#                  attachments=attachments, 
#                  body=body)