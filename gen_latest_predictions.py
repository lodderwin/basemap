import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import pygmail

yr = yahoo_reader.finance_data(tickers=['OMEX', 'TENX'])
df_main = yr.get_data()

ticker_dict = {}
# Prep data for LSTM model
window_length = 11
for ticker in yr.tickers:
    df = df_main[df_main.ticker == ticker].reset_index(drop=True)
    df_p = pp.pre_process_data(df, window_length=window_length)
#    df_t = pp.pre_process_data(df,window_length=5)
    # Split closing price into test and train
    close_nmd_array = utils.series_to_ndarray(df_p, column='close_nmd')
### 2D preperation
    day_number_array = utils.series_to_ndarray(df_p, column='day_number')
    combined_input = np.concatenate((close_nmd_array,day_number_array),axis=2)
    x_train, y_train, x_test, y_test = utils.train_test_split(combined_input,2)
    days_ahead=1
    # Build model
    model, mse = lstm_model.randomised_model_config(2,
                                                    ticker,
                                                    df,
                                                    days_ahead,
                                                    x_train,
                                                    y_train, 
                                                    x_test,
                                                    y_test,
                                                    iterations=3,
                                                    epochs=20)

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