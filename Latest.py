import pandas as pd
import numpy as np

import YahooReader
from numpy import newaxis
import PreProcessing as pp
import LstmUtils as utils
import LstmModel
import Plotting
import gc
import PyGmail
from keras.models import Sequential, load_model
model_folder = './shortterm_models/'

df_stock_data = pd.read_csv('tickers_done_short_correct.csv')
df_stock_data = df_stock_data.dropna() 
df_stock_data = df_stock_data.reset_index(drop=True)
stocks = df_stock_data['tickers'].tolist()
stock_window_lengths = df_stock_data['window_length'].tolist()
stock_margins = df_stock_data['margin'].tolist()

good_stocks = ['ADMP', 'ACHV', 'SNMX', 'CBI', 'ASNA', 'AIRI', 'BCRX', 'VSAT','WBS','WMGI', 'GEN', 'GALT', 'ANTM', 'ANF','ARLP','ARNA', 'ASTE','BBGI', 'BDC', 'AKS', 'ALXN', 'AMKR','CECO', 'CETV','UNFI', 'UNP','CLH', 'CLB','CHS', 'COLB', 'CNXN','CHRW','CTAS','EEI','CHRW','CTXS','CVTI','DDE','DGICA','DLTR','DMRC','DO','EA']
#%%
yr = YahooReader.finance_data(tickers=stocks)
df_main = yr.get_fix_yahoo_data()
#%%
ticker_dict = {}
#df_main = pd.read_csv('saved_stock_data.csv')
# Prep data for LSTM model
good_window_lengths = [21,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]

df_test = {}
short_term_folder = './short_term_models/'
margin_clear={}

for i in range(len(stocks)):
    
    ticker = stocks[i]
    window_length = stock_window_lengths[i]
    margin = stock_margins[i]
    model = load_model(model_folder+ticker+'_'+str(window_length)+'_model.h5')
    df = df_main[df_main.ticker == ticker].reset_index(drop=True)
    df['volume'] = df['volume'].replace(0,1.0)   
    window_length_pred = window_length - 1
    df_pred = pp.pre_process_data(df[-window_length_pred:], window_length=window_length_pred)
    df_pred = df_pred.reset_index(drop=True)
    close_nmd_array = utils.series_to_ndarray(df_pred,window_length, column='close_nmd')
    volumne_nmd_array = utils.series_to_ndarray(df_pred,window_length, column='volume_nmd')
    high_nmd_close_array = utils.series_to_ndarray(df_pred,window_length, column='high_nmd_close')
    low_nmd_close_array = utils.series_to_ndarray(df_pred,window_length, column='low_nmd_close')
    open_nmd_close_array = utils.series_to_ndarray(df_pred,window_length, column='open_nmd_close')
    day_number_array = utils.series_to_ndarray(df_pred,window_length, column='day_number')
    dates_array = utils.series_to_ndarray(df_pred,window_length, column='date')
    windows_array = utils.series_to_ndarray(df_pred,window_length, column='window')
    #closing price must go first 
    combined_input = np.concatenate((close_nmd_array,open_nmd_close_array,low_nmd_close_array,high_nmd_close_array,volumne_nmd_array, day_number_array),axis=2)
    predicted_normalised = (model.predict(combined_input)[0,0])
    predicted = (predicted_normalised + 1.0)*df_pred.loc[0,'normaliser']
    growth = predicted/df_pred.loc[len(df_pred)-1,'close']
    
    ticker_dict[ticker] = growth
    if growth>margin:
        margin_clear[ticker]=growth
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
print(margin_clear)

    
    
    
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