import pandas as pd
import numpy as np
from numpy import newaxis

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import plotting
import gc
import pygmail
from keras.models import Sequential, load_model
#%%
model_folder = './short_term_models/'

df_tickers = pd.read_csv('tickers_of_interest.csv',sep=';')[-100:]
tickers = df_tickers['ticker'].tolist()
tickers_done = utils.get_tickers_done(model_folder)
tickers_to_do = [ticker for ticker in tickers if ticker not in tickers_done]

good_stocks = ['ADMP', 'ACHV', 'SNMX', 'CBI', 'ASNA', 'AIRI', 'BCRX']
good_window_lengths = [21,16,16,16,16,16,16]
#True for analyse, False for prediction
##
run_model = True
##
current = [ 'ASUR', 'ATLC', 'AIRI', 'SNMX','ALSK', 'BASI', 'BCRX', 'CAMT', 'CENX', 'CGEN', 'CLWT', 
                      'CRZO', 'DWSN', 'AXTI', 'ELTK', 'ESIO', 'EXAS']
#current = ['ASNA', 'ASUR', 'ATLC', 'AIRI']
current = ['CBI', 'CAMT', 'AMAG', 'ADMP', 'DAIO', 'GALT', 'GEN']
yr = yahoo_reader.finance_data(tickers=tickers_to_do)
df_main, tickers_to_do = yr.main()

#%%
ticker_dict = {}
#df_main = pd.read_csv('saved_stock_data.csv')
# Prep data for LSTM model
window_length = 21
days_ahead=1
df_test = {}
short_term_folder = './short_term_models/'

for ticker in tickers_to_do:
    initial_investment = 100.0
    window_size = 12
    for window_length in [16]:

        df = df_main[df_main.ticker == ticker].reset_index(drop=True)
        df['volume'] = df['volume'].replace(0,1.0)
        
        if run_model:
            df_p = pp.pre_process_data(df, window_length=window_length)
            close_nmd_array = utils.series_to_ndarray(df_p,window_length, column='close_nmd')
            volumne_nmd_array = utils.series_to_ndarray(df_p,window_length, column='volume_nmd')
            high_nmd_array = utils.series_to_ndarray(df_p,window_length, column='high_nmd')
            low_nmd_array = utils.series_to_ndarray(df_p,window_length, column='low_nmd')
            day_number_array = utils.series_to_ndarray(df_p,window_length, column='day_number')
            dates_array = utils.series_to_ndarray(df_p,window_length, column='date')
            windows_array = utils.series_to_ndarray(df_p,window_length, column='window')
            #closing price must go first 
            combined_input = np.concatenate((close_nmd_array,volumne_nmd_array,high_nmd_array,low_nmd_array, day_number_array),axis=2)
            x_train, y_train, x_test, y_test,  train_days, test_days, test_windows = utils.train_test_split(combined_input,combined_input.shape[2], dates_array, windows_array)
       
            investment, best_investment_dev = lstm_model.randomised_model_config(test_windows,
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
                                                            iterations=6,
                                                            epochs=8)
            gc.collect()    

            if investment>initial_investment :
                initial_investment = investment
                window_size = window_length
#            del model

                
            
            
    #    date_today = dt.datetime.now().strftime("%Y-%m-%d")
#        date = '2018-04-25'
#        model = load_model(date+'_'+ticker+'_model.h5')
        ### predict tomorrow  
        ##train data do something with it
        
        if not run_model:
            window_length_pred = window_length - 1
            df_pred = pp.pre_process_data(df[-window_length_pred:], window_length=window_length_pred)
            df_pred = df_pred.reset_index(drop=True)
            close_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='close_nmd')
            volumne_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='volume_nmd')
            high_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='high_nmd')
            low_nmd_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='low_nmd')
            day_number_array_pred = utils.series_to_ndarray(df_pred,window_length_pred, column='day_number')
        
            combined_input_pred = np.concatenate((close_nmd_array_pred,volumne_nmd_array_pred,high_nmd_array_pred,low_nmd_array_pred,day_number_array_pred),axis=2)
            predicted_normalised = (model.predict(combined_input_pred)[0,0])
            predicted = (predicted_normalised + 1.0)*df_pred.loc[0,'normaliser']
            growth = predicted/df_pred.loc[len(df_pred)-1,'close']
            ticker_dict[ticker] = growth
#        #%%
#    volatile_tickers_done = pd.read_csv('tickers_done_short.csv')
#    volatile_tickers_done_lst = volatile_tickers_done['tickers'].tolist()
#        
#    volatile_tickers_done_lst.append(ticker)
#    volatile_tickers_done_lst_window_length = volatile_tickers_done['window_length'].tolist()
#    volatile_tickers_done_lst_window_length.append(window_size)
#    df_temp = pd.DataFrame({'tickers':volatile_tickers_done_lst,'window_length':volatile_tickers_done_lst_window_length})
#    df_temp.to_csv('tickers_done_short.csv')
    
    
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