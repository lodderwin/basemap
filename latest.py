import datetime as dt

from keras.models import load_model
import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
from slack import PyntBot

today = dt.datetime.today().strftime('%d_%m_%Y')

user = utils.load_user_from_yml(yml_file='./configs/user_settings.yml')

df_results = utils.read_all_results_csv(directory='./results/')

tickers_to_do = list(df_results.ticker.unique())
#tickers_to_do = ['ASRV', ]
yr = yahoo_reader.finance_data(tickers=tickers_to_do[:10])
df_main, tickers = yr.get_fix_yahoo_data()

df_results = df_results[df_results.ticker.isin(tickers)]

predictions = pd.DataFrame({})
#%%
for idx, row in df_results.iterrows():
    model = load_model('./shortterm_models/{}_{}_model.h5'.format(row.ticker, int(row.window_length)))
    df = df_main[df_main.ticker == row.ticker].reset_index(drop=True)
    df['volume'] = df['volume'].replace(0,1.0)  
    
    window_length_pred = int(row.window_length - 1)
    
    #new
#    df = df_main[df_main.ticker == ticker].reset_index(drop=True)
    df['volume'] = df['volume'].replace(0,1.0)
    df = pp.ichimoku_cloud(df)
    
    df = pp.crsi(df)
    df.drop(['delta_close3', 'dUp3', 'dDown3',
       'RolUp_3', 'RolDown_3', 'rsi_3', 'delta_close2', 'dUp2', 'dDown2',
       'RolUp_2', 'RolDown_2', 'rsi_2', 'close_pct_change', 'rank'], axis=1, inplace=True)
    df = df[200:]
    #new
    df_pred = (pp
        .pre_process_data(df[-window_length_pred:].reset_index(drop=True),
                          window_length=window_length_pred)
        .reset_index(drop=True)
    )
        
    close_nmd_array = utils.series_to_ndarray(df_pred, row.window_length, column='close_nmd')
    volumne_nmd_array = utils.series_to_ndarray(df_pred, row.window_length, column='volume_nmd')
    high_nmd_close_array = utils.series_to_ndarray(df_pred, row.window_length, column='high_nmd_close')
    low_nmd_close_array = utils.series_to_ndarray(df_pred, row.window_length, column='low_nmd_close')
    open_nmd_close_array = utils.series_to_ndarray(df_pred, row.window_length, column='open_nmd_close')
    day_number_array = utils.series_to_ndarray(df_pred, row.window_length, column='day_number')
    dates_array = utils.series_to_ndarray(df_pred, row.window_length, column='date')
    windows_array = utils.series_to_ndarray(df_pred, row.window_length, column='window')
    #new
    crsi_array = utils.series_to_ndarray(df_pred, row.window_length, column='csri')
    a_b_diff_array = utils.series_to_ndarray(df_pred, row.window_length, column='diff_span_a_b')
    span_a_array = utils.series_to_ndarray(df_pred, row.window_length, column='span_a')
    conversion_line_array = utils.series_to_ndarray(df_pred, row.window_length, column='conversion_line')
    base_line_array = utils.series_to_ndarray(df_pred, row.window_length, column='base_line')
    #new
    
    #closing price must go first
    print('making prediction for {}'.format(row.ticker))
    combined_input = np.concatenate((close_nmd_array,open_nmd_close_array,low_nmd_close_array,high_nmd_close_array,volumne_nmd_array, day_number_array,crsi_array,a_b_diff_array,span_a_array,conversion_line_array,base_line_array),axis=2)
    predicted_normalised = (model.predict(combined_input)[0,0])
    predicted = (predicted_normalised + 1.0)*df_pred.loc[0,'normaliser']
    growth = predicted/df_pred.loc[len(df_pred)-1,'close']
    
    df_temp = pd.DataFrame({
        'ticker': [row.ticker],
        'coefficient': [row.angle_coefficient],
        'start_line': [row.start_coefficient],
        'date_created': [dt.datetime.now()],
        'growth': [growth],
        'growth_mt_margin': [np.where(growth > row.margin, 1, 0)],
        'user': [user]
    })
    
    predictions = pd.concat([predictions, df_temp])
    
    
predictions.to_csv('./predictions/{}_{}.csv'.format(today, user), index=False)

pyntbot = PyntBot(df=predictions)

pyntbot.send_top_tickers()