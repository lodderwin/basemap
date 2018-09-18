import datetime as dt
import json

import pandas as pd
import numpy as np

import yahoo_reader
import preprocessing as pp
import lstm_utils as utils
import lstm_model
import gc
import matplotlib.pyplot as plt

user = utils.load_user_from_yml(yml_file='./configs/user_settings.yml')
user_tickers = utils.get_tickers_for_a_user(user=user)
tickers_done = utils.get_tickers_done('./results/')
tickers_to_do = [ticker for ticker in user_tickers if ticker not in tickers_done]
tickers_to_do = tickers_to_do[:10]
yr = yahoo_reader.finance_data(tickers=tickers_to_do)
#df = pd.read_csv('forflight.csv', sep=',')

df, tickers = yr.get_fix_yahoo_data()
#ichimoku cloud
#tickers = df.ticker.unique().tolist()
#df = df_main[df_main.ticker == 'AMAG'].reset_index(drop=True)
#%%
#df = df[df.ticker == 'ASRV'].reset_index(drop=True)
#new_df_main = pd.DataFrame([])
#split = 100
#rest = len(df)%split
#
#df = df[rest:]
#print(len(df))
#for i in range(int(len(df)/split)-2):
#    df_temp = df[i*split:((i+1)*split)]
#    l = [i for i in range(len(df_temp))]
#    se = pd.Series(l)
#    df_temp['x_fit'] = se.values
#    df_temp['y_fit'] = 0
#    df_temp['section'] = i
#    df_temp['polyfit'] = np.poly1d(np.polyfit(df_temp['x_fit'], df_temp['close'], 6))(df_temp['x_fit'])
#    df_temp['min'] = df_temp.polyfit[(df_temp.polyfit.shift(1) > df_temp.polyfit) & (df_temp.polyfit.shift(-1) > df_temp.polyfit)]
#    df_temp['max'] = df_temp.polyfit[(df_temp.polyfit.shift(1) < df_temp.polyfit) & (df_temp.polyfit.shift(-1) < df_temp.polyfit)]
#    df_temp['min'][i*split] = df_temp.iloc[0]['polyfit']
#    df_temp['min'][((i+1)*split)-1]= df_temp.iloc[-1]['polyfit']
#    
#    new_df_main = pd.concat([new_df_main, df_temp])
#new_df_main['min'].fillna(value=np.nan, inplace=True)
#new_df_main['max'].fillna(value=np.nan, inplace=True)
##%%
#df_temp = new_df_main[-800:-500]
##plt.plot(df_temp['x_fit'].values.tolist(), df_temp['polyfit'].values.tolist() )
##
##plt.show()
#
#fig, ax1 = plt.subplots()
#ax1.plot(df_temp.index.values.tolist(), df_temp['polyfit'].tolist(), 'b-')
#ax1.scatter(df_temp.index.values.tolist(), df_temp['min'].tolist(), c='r',s=100)
#ax1.scatter(df_temp.index.values.tolist(), df_temp['max'].tolist(), c='g',s=100)
##ax1.tick_params('y', colors='b')
#
#
#ax2 = ax1.twinx()
#ax2.plot(df_temp.index.values.tolist(), df_temp['close'].values.tolist(), 'b.')
##ax2.tick_params('y', colors='r')
#
#fig.tight_layout()
#plt.show()



    
#%%
#connor relative strength index
#df_main.to_csv('forflight.csv')
days_ahead = 1
#df_main=df
for ticker in tickers:
    new_df_main = pd.DataFrame([])
    df = df[df.ticker == ticker].reset_index(drop=True)
    df['volume'] = df['volume'].replace(0,1.0)
    split = 100
    rest = len(df)%split
    
    df = df[rest:]
    print(len(df))
    for i in range(int(len(df)/split)-2):
        df_temp = df[i*split:((i+1)*split)]
        l = [i for i in range(len(df_temp))]
        se = pd.Series(l)
        df_temp['x_fit'] = se.values
        df_temp['y_fit'] = 0
        df_temp['section'] = i
        df_temp['polyfit'] = np.poly1d(np.polyfit(df_temp['x_fit'], df_temp['close'], 6))(df_temp['x_fit'])
        df_temp['min'] = df_temp.polyfit[(df_temp.polyfit.shift(1) > df_temp.polyfit) & (df_temp.polyfit.shift(-1) > df_temp.polyfit)]
        df_temp['max'] = df_temp.polyfit[(df_temp.polyfit.shift(1) < df_temp.polyfit) & (df_temp.polyfit.shift(-1) < df_temp.polyfit)]
        df_temp['min'][i*split] = df_temp.iloc[0]['polyfit']
        df_temp['min'][((i+1)*split)-1]= df_temp.iloc[-1]['polyfit']
        
        new_df_main = pd.concat([new_df_main, df_temp])
        
    new_df_main = new_df_main[int(0.25*len(df)):]
    new_df_main['min'].fillna(value=np.nan, inplace=True)
    new_df_main['max'].fillna(value=np.nan, inplace=True)
    
#    df_temp = new_df_main[-800:-500]
#    #plt.plot(df_temp['x_fit'].values.tolist(), df_temp['polyfit'].values.tolist() )
#    #
#    #plt.show()
#    
#    fig, ax1 = plt.subplots()
#    ax1.plot(df_temp.index.values.tolist(), df_temp['polyfit'].tolist(), 'b-')
#    ax1.scatter(df_temp.index.values.tolist(), df_temp['min'].tolist(), c='r',s=100)
#    ax1.scatter(df_temp.index.values.tolist(), df_temp['max'].tolist(), c='g',s=100)
#    #ax1.tick_params('y', colors='b')
#    
#    
#    ax2 = ax1.twinx()
#    ax2.plot(df_temp.index.values.tolist(), df_temp['close'].values.tolist(), 'b.')
#    #ax2.tick_params('y', colors='r')
#    
#    fig.tight_layout()
#    plt.show()
    
    
    
    
    
    df_results = utils.read_a_user_results_csv(directory='./results/', user=user)
    window_length = 31
    
    df = pp.ichimoku_cloud(df)
    
    df = pp.crsi(df)
    df.drop(['delta_close3', 'dUp3', 'dDown3',
       'RolUp_3', 'RolDown_3', 'rsi_3', 'delta_close2', 'dUp2', 'dDown2',
       'RolUp_2', 'RolDown_2', 'rsi_2', 'close_pct_change', 'rank'], axis=1, inplace=True)
    df = df[200:]
#%%
    df_p = pp.pre_process_data(df, window_length=window_length)
#    print(df.tail())

    
    close_nmd_array = utils.series_to_ndarray(df_p, window_length, column='close_nmd')
    volumne_nmd_array = utils.series_to_ndarray(df_p, window_length, column='volume_nmd')
    high_nmd_close_array = utils.series_to_ndarray(df_p, window_length, column='high_nmd_close')
    low_nmd_close_array = utils.series_to_ndarray(df_p, window_length, column='low_nmd_close')
    open_nmd_close_array = utils.series_to_ndarray(df_p, window_length, column='open_nmd_close')
    day_number_array = utils.series_to_ndarray(df_p, window_length, column='day_number')
    dates_array = utils.series_to_ndarray(df_p, window_length, column='date')
    windows_array = utils.series_to_ndarray(df_p, window_length, column='window')
    crsi_array = utils.series_to_ndarray(df_p, window_length, column='csri')
    a_b_diff_array = utils.series_to_ndarray(df_p, window_length, column='diff_span_a_b')
    span_a_array = utils.series_to_ndarray(df_p, window_length, column='span_a')
    conversion_line_array = utils.series_to_ndarray(df_p, window_length, column='conversion_line')
    base_line_array = utils.series_to_ndarray(df_p, window_length, column='base_line')
    
   
    #closing price must go first 
    combined_input = np.concatenate((close_nmd_array,open_nmd_close_array,low_nmd_close_array,high_nmd_close_array,volumne_nmd_array, day_number_array,crsi_array,a_b_diff_array,span_a_array,conversion_line_array,base_line_array),axis=2)    
    x_train, y_train, x_test, y_test, train_days_sim_non_normal, test_days, test_windows,train_windows_non_randomized,x_train_sim = utils.train_test_split(combined_input,combined_input.shape[2], dates_array, windows_array)
    x_train_sim = utils.train_test_split_for_x_train(combined_input,combined_input.shape[2], dates_array, windows_array)
#%%
    investment, best_investment_dev, params, margin, mcr, new_df, angle_coefficient, start_coefficient = lstm_model.randomised_model_config(
        test_windows,
        df_p,
        test_days,
        train_days_sim_non_normal,
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
        new_df_main,
        iterations=25
    )
    #%%
    
    final_indicator, df_selection_title_positive, df_selection_title_negative, df_selection = lstm_model.selection_mcr(new_df_main, new_df, ticker)
#%%
    new_df_main['date'] = new_df_main['date'].astype('datetime64[ns]')
    new_new_df = lstm_model.merge_df_predict(new_df,new_df_main,ticker )
    new_new_df['min'].fillna(value=np.nan, inplace=True)
    new_new_df['max'].fillna(value=np.nan, inplace=True)
    new_new_df['min'].fillna(value=0, inplace=True)
    new_new_df['max'].fillna(value=0, inplace=True)
    new_new_df['minmax'] = new_new_df['min']+new_new_df['max']
    new_new_df['check'] = 0
    mask = (new_new_df['minmax'] == 0)
    new_new_df_valid = new_new_df[mask]
#    new_new_df['c'] = 0
    new_new_df.loc[mask, 'check'] = 1
    new_new_df['value_grp'] = (new_new_df.check.diff(1) != 0).astype('int').cumsum()
    df_selection = pd.DataFrame({'begindate' : new_new_df.groupby('value_grp').date.first(), 
              'enddate' : new_new_df.groupby('value_grp').date.last(),
              'closebegin' : new_new_df.groupby('value_grp').close.first(),
              'closeend' : new_new_df.groupby('value_grp').close.last(),
              'investmentbegin' : new_new_df.groupby('value_grp').investment.first(),
              'investmentend' : new_new_df.groupby('value_grp').investment.last(),
              'consecutive' : new_new_df.groupby('value_grp').size()}).reset_index(drop=True)
    df_selection['closeratio'] = df_selection['closeend']/df_selection['closebegin']
    df_selection['investmentratio'] = df_selection['investmentend']/df_selection['investmentbegin']
    df_selection = df_selection[df_selection.consecutive >= 10]
    df_selection = df_selection.reset_index(drop=True)
    df_selection['indicator'] = df_selection['investmentratio'] - df_selection['closeratio']
    df_selection['closeratio_plot'] = df_selection['closeratio'] - 1
    df_selection['investmentratio_plot'] = df_selection['investmentratio'] - 1
    df_selection['cirkelsize_plot'] = df_selection['consecutive']/(df_selection['consecutive'].sum())
    df_selection_title_positive = df_selection[df_selection.investmentratio>df_selection.closeratio]
    df_selection_title_negative = df_selection[df_selection.investmentratio<=df_selection.closeratio]
    df_selection_decrease = df_selection[df_selection.closeratio <= 1.]
    df_selection_increase = df_selection[df_selection.closeratio > 1.]
    df_selection_decrease['indicator_negative_negative'] = df_selection_decrease['investmentratio'] - 1.0
    df_selection_decrease['indicator_final'] = ((df_selection_decrease['indicator']+df_selection_decrease['indicator_negative_negative'])*df_selection_decrease['consecutive'])/df_selection_decrease['consecutive'].sum()
    final_indicator_decrease = df_selection_decrease['indicator_final'].sum()
    df_selection_increase['indicator_positive_negative'] = df_selection_increase['investmentratio'] - 1.0
    df_selection_increase['indicator_final'] = ((df_selection_increase['indicator']+df_selection_increase['indicator_positive_negative'])*df_selection_increase['consecutive'])/df_selection_increase['consecutive'].sum()
    final_indicator_increase = df_selection_increase['indicator_final'].sum()
    final_indicator = final_indicator_decrease + final_indicator_increase
    print(final_indicator)
#%%    
    plt.scatter(np.asarray(df_selection['closeratio_plot'].tolist()), np.asarray(df_selection['investmentratio_plot'].tolist()), s=np.asarray(df_selection['cirkelsize_plot'].tolist())*1000)
    line = np.polyfit(np.asarray(df_selection['closeratio_plot'].tolist()), np.asarray(df_selection['investmentratio_plot'].tolist()), 1, w=np.asarray(df_selection['cirkelsize_plot'].tolist())*1000)
    x = np.arange(-1,1.01,0.01)
    y1 = line[0]*x + line[1]
    y = x
    plt.plot(x,y)
    plt.plot(x,y1)
    plt.xlabel('close ratio')
    plt.ylabel('investment ratio')
    plt.title('Above : '+str(df_selection_title_positive['cirkelsize_plot'].sum())+'   Below : '+ str(df_selection_title_negative['cirkelsize_plot'].sum()))
    #plt.show()
    
    
    #%%
    # In params batch size is np.int64 and needs to be int in order to be json dumped
    params = {k : int(v) for k, v in params.items()} 
    if (investment/300) > 1.00:
        df_temp = pd.DataFrame({
            'coefficient': [angle_coefficient],
            'start_line': [start_coefficient],
            'window_length': [window_length],
            'mcr': [mcr],
            'ticker': [ticker],
            'params': [json.dumps(params)],
            'date_created': [dt.datetime.now()]
        })
    else: 
        df_temp = pd.DataFrame({
            'coefficient': [angle_coefficient],
            'start_line': [start_coefficient],
            'window_length': [np.NaN],
            'mcr': [np.NaN],
            'ticker': [ticker],
            'params': [json.dumps(params)],
            'date_created': [dt.datetime.now()]
        })  

    
    df_results = pd.concat([df_results, df_temp])
    df_results.to_csv('./results/model_results_{}.csv'.format(user), index=False)