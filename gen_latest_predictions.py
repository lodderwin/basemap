import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import pygmail

yr = yahoo_reader.finance_data(tickers=['CRNT','AVD','BOOM'])
df_main = yr.get_data()

ticker_dict = {}

# Prep data for LSTM model
for ticker in yr.tickers:
    df = df_main[df_main.ticker == ticker].reset_index(drop=True)
    df_p = pp.pre_process_data(df)

    # Split closing price into test and train
    close_nmd_array = utils.series_to_ndarray(df_p, column='close_nmd')
    x_train, y_train, x_test, y_test = utils.train_test_split(close_nmd_array)

    # Build model
    model = lstm_model.randomised_model_config(x_train, y_train, x_test, y_test)

    # Create X based on last five days of close data
    X = utils.series_to_ndarray(df_p[-5:], column='close')
    # Normailse X, by dividing all numbers in array but first number
    X_nmd = X / X[0][0]
 
    predictions_nmd = lstm_model.predict(model, X_nmd)
    predictions = predictions_nmd * X[0][0]
    
    # Calculate predicted growth over 5 days
    growth = round((predictions_nmd[0][predictions_nmd.shape[1] - 1]
        - predictions_nmd[0][0])[0] * 100, 2)
    
    # Plot predictions
    plotting.plot_latest_prediction(df, predictions, ticker, growth)
    
    # Add predicted growth to ticker_dict
    ticker_dict[ticker] = growth

# Compose and send email
subject, body, attachments = pygmail.compose_email(expected_deltas=ticker_dict)
pygmail.send_mail(subject=subject,
                  attachments=['./plots/2018-03-20/latest_prediction_CRNT.png',
                               './plots/2018-03-20/latest_prediction_BOOM.png',
                               './plots/2018-03-20/latest_prediction_AVD.png'], 
                  body=body)