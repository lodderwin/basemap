import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import pygmail

yr = yahoo_reader.finance_data(tickers=['AMSC'])#,'GEN','SMRT'])
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
    model, mse = lstm_model.randomised_model_config(x_train,
                                                    y_train, 
                                                    x_test,
                                                    y_test,
                                                    iterations=1,
                                                    epochs=1)

    # Create X based on last window in data (last window is 0)
    X, X_nmd = utils.gen_X(df_p, window=0)
 
    predictions_nmd = lstm_model.predict(model, X_nmd)
    predictions = (predictions_nmd + 1) * X[0][0]
    
    # Calculate predicted growth over 5 days
    growth = np.round(((predictions[0][4] / X[0][4]) -1) * 100, 2)[0]
    
    # Plot predictions
    plt = plotting.plot_latest_prediction(df, predictions, ticker, growth, mse,
                                          model, df_p)
    
    # Add predicted growth to ticker_dict
    ticker_dict[ticker] = growth

# Compose and send email
subject, body, attachments = pygmail.compose_email(expected_deltas=ticker_dict)
pygmail.send_mail(subject=subject,
                  attachments=attachments, 
                  body=body)