import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, DateOffset
from pandas.tseries.offsets import *
from datetime import datetime
import time
import numpy as np

import matplotlib.pyplot as plt




def ichimoku_cloud(df):
#    df = df[['date','close', 'high', 'low']]
#    weekmask = 'Mon Tue Wed Thu Fri'
#    holidays = [datetime(2016, 3, 30), datetime(2016, 5, 28), datetime(2016, 7, 4), datetime(2016, 5, 28),
#                datetime(2016, 7, 4), datetime(2016, 9, 3), datetime(2016, 11, 22), datetime(2016, 12, 25),
#                datetime(2017, 3, 30), datetime(2017, 5, 28), datetime(2017, 7, 4), datetime(2017, 5, 28),
#                datetime(2017, 7, 4), datetime(2017, 9, 3), datetime(2017, 11, 22), datetime(2017, 12, 25),
#                datetime(2018, 3, 30), datetime(2018, 5, 28), datetime(2018, 7, 4), datetime(2018, 5, 28),
#                datetime(2018, 7, 4), datetime(2018, 9, 3), datetime(2018, 11, 22), datetime(2018, 12, 25)]
#    bday_cust = CustomBusinessDay(holidays=holidays, weekmask=weekmask) 
#    
##FIX FREQ
##    df = df[[ 'close', 'high', 'low']]
#    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
#    period9_high = pd.rolling_max(df['high'], window=9)
#    period9_low = pd.rolling_min(df['low'], window=9)
#    df['tenkan_sen'] = (period9_high + period9_low) / 2
#    tenkan_sen = (period9_high + period9_low) / 2
#    
#    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
#    period26_high = pd.rolling_max(df['high'], window=26)
#    period26_low = pd.rolling_min(df['low'], window=26)
#    df['kijun_sen'] = (period26_high + period26_low) / 2
#    kijun_sen = (period26_high + period26_low) / 2
#    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
#    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
#    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
#    period52_high = pd.rolling_max(df['high'], window=52)
#    period52_low = pd.rolling_min(df['low'], window=52)
#    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
#    # The most current closing price plotted 22 time periods behind (optional)
#    df['chikou_span'] = df['close'].shift(-22) # 22 according to investopedia
#    return df
    df['date'] =  pd.to_datetime(df['date'], format='%Y-%m-%d')
    from datetime import timedelta

    df['high_9'] = df['high'].rolling(window= 9).max()
    df['low_9'] = df['low'].rolling(window= 9).min()
    df['tenkan_sen'] = (df['high_9'] + df['low_9']) /2
    df['tenkan_sen_average'] = pd.rolling_mean(df['tenkan_sen'],window=2)
    
    df['high_26'] = df['high'].rolling(window= 26).max()
    df['low_26'] = df['low'].rolling(window= 26).min()
    df['kijun_sen'] = (df['high_26'] + df['low_26']) /2
    df['kijun_sen_average'] = pd.rolling_mean(df['kijun_sen'],window=2)
    
    # this is to extend the 'df' in future for 26 days
    # the 'df' here is numerical indexed df
#    last_index = df.iloc[-1:].index[0]
#    last_date = df['date'].iloc[-1].date()
#    for i in range(26):
#        df.loc[last_index+1 +i, 'date'] = last_date + timedelta(days=i)
    
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2)
#    df['senkou_span_a_average'] = pd.rolling_mean(df['senkou_span_a'],window=2)
    
    df['high_52'] = df['high'].rolling(window= 52).max()
    df['low_52'] = df['low'].rolling(window= 52).min()
    df['senkou_span_b'] = ((df['high_52'] + df['low_52']) /2)
#    df['senkou_span_b_average'] = pd.rolling_mean(df['senkou_span_b'],window=2) 
    # most charting softwares dont plot this line
    df['chikou_span'] = df['close'].shift(-22) #sometimes -26 
#    df['chikou_span_average'] = pd.rolling_mean(df['chikou_span'],window=2) 
    
    tmp = df[['close','senkou_span_a','senkou_span_b','kijun_sen','tenkan_sen']].tail(50)
    a1 = tmp.plot(figsize=(15,10))
    a1.fill_between(tmp.index, tmp.senkou_span_a, tmp.senkou_span_b)
    
    return df

def crsi(df):
    #3 day RSI
    df=RSIfun(df,n=3)
    #2 day rsi
    df=RSIfun(df,n=2)
    rollrank = lambda data: data.size - data.argsort().argsort()[-1]
    df['close_pct_change'] = df['close'].pct_change()

    df['rank'] = (pd.rolling_apply(df['close_pct_change'], 100, rollrank))
    df['csri'] = ((df['rsi_3']+df['rsi_2']+df['rank'])/300)**2
#    tmp = df[['csri']].tail(100)
#    plt.plot(np.arange(1,101,1),tmp)
#    plt.show()
    
    
    
    
#    
#    #relative magnitude price change
#    df['higher_streak'] = df['close'].index.to_series().map(lambda i: consecutive_run(gt, df['close'], i))
#    df['lower_streak'] = df['close'].index.to_series().map(lambda i: consecutive_run(lt, df['close'], i))
#    
#    
#    
#    df['diff_close'] = df['close'].pct_change()
#    df['count_roc'] = df.count[[df['diff_close']>pd.rolling_window(df['diff_close'], window=100)]]
# #   condition smaller or larger than 0--> higher (for ROC) -->average (for RSI)
#    df['positive'] = df['close'].groupby((df['close'] < df.close.shift()).cumsum()).cumcount()
#    df['negative'] = df['close'].groupby((df['close'] > df.close.shift()).cumsum()).cumcount()*-1.0
#    df['updown'] = df['positive'] + df['negative']
#    
#    #rsi 3-10 periods
#    #Updownlength if <-2 == 2, rsi 2 days
#    #ROC 50-100 days
##    pd.rolling_count((df['diff']<df['diff']),window=100)
    return df

#create rsi for every column, multiply by the streak value
def RSIfun(df, n=3):
    df['delta_close'+str(n)] = df['close'].diff()
    #-----------
    df['dUp'+str(n)]= df['delta_close'+str(n)][df['delta_close'+str(n)] > 0]
    df['dDown'+str(n)]= df['delta_close'+str(n)][df['delta_close'+str(n)] < 0]
    df[['dUp'+str(n), 'dDown'+str(n)]] = df[['dUp'+str(n),'dDown'+str(n)]].fillna(value=0)

    df['RolUp_'+str(n)]=pd.rolling_mean(df['dUp'+str(n)], n)
    df['RolDown_'+str(n)]=pd.rolling_mean(df['dDown'+str(n)], n).abs()
    #if rollupof roldown = 0 rsi 100 or 0
    conditions = [
            (df['RolUp_'+str(n)] == 0),
            (df['RolDown_'+str(n)] == 0),
            (df['RolUp_'+str(n)] == 0) & (df['RolDown_'+str(n)] == 0)]
    choices = [0,100,0]
    df['rsi_'+str(n)] = np.select(conditions, choices, default=100.0 - (100.0 / (1.0 +(df['RolUp_'+str(n)]/df['RolDown_'+str(n)]))))
#    df['RS_'+str(n)] = df['RolUp_'+str(n)]/df['RolDown_'+str(n)]
#    df['rsi_'+str(n)]= 100.0 - (100.0 / (1.0 + df['RS_'+str(n)]))
    return df
import pandas as pd
from operator import gt, lt

a = pd.Series([30, 10, 20, 25, 35, 15])

def consecutive_run(op, ser, i):
    """
    Sum the uninterrupted consecutive runs at index i in the series where the previous data
    was true according to the operator.
    """
    thresh_all = op(ser[:i], ser[i])
    # find any data where the operator was not passing.  set the previous data to all falses
    non_passing = thresh_all[~thresh_all]
    start_idx = 0
    if not non_passing.empty:
        # if there was a failure, there was a break in the consecutive truth values,
        # so get the final False position. Starting index will be False, but it
        # will either be at the end of the series selection and will sum to zero
        # or will be followed by all successive True values afterwards
        start_idx = non_passing.index[-1]
    # count the consecutive runs by summing from the start index onwards
    return thresh_all[start_idx:].sum()


#res = pd.concat([a.index.to_series().map(lambda i: consecutive_run(gt, a, i)),
#                 a.index.to_series().map(lambda i: consecutive_run(lt, a, i))],
#       axis=1)
#res.columns = ['Higher than streak', 'Lower than streak']
#print(res)


    
    
    
def normalise_windows(df, window_length=6):
    """
    Takes a DataFrame as input and returns a much larger DataFrame with
    normalised windows. A window of data is a run of dates of size window
    length. The number of windows created is dependant on the length of
    the DataFrame and the window size: Windows = Length - Window size.
    For each window the data is normalised by dividing by the first row of
    each window.
    
    Parameters
    --------
    df : pd.DataFrame incl columns: ticker, date, close
    window_length : size of Window = train + test
    
    Returns
    --------
    df : df with extra columns: window (window id), normaliser (for 
    de-normalisation), close_nmd (nomrlaised close price)
    """
    # Minus 5 instead of 6 due to range function used in loop
    windows = len(df) - (window_length-1)
    # Create empty dataframe to be filled with windows
    df_final = pd.DataFrame([])
    
    for i in range(0, windows):
        # Print progress counter
        print('\r {} of {}'.format(i + 1, windows), end='\r', flush=False)
        # Create a dataframe for every 6 rows
        df_temp = df[i:i + window_length]
        # Reset index
        df_temp = df_temp.reset_index(drop=True)
        # Create Window id
        df_temp['window'] = i + 1
        # Normailse close price column
        df_temp['normaliser'] = df_temp.close[0]
        df_temp['normaliserv'] = df_temp.volume[0]
        df_temp['close_nmd'] = (df_temp.close / df_temp.normaliser) - 1
        df_temp['volume_nmd'] = (df_temp.volume / df_temp.normaliserv) - 1
        df_temp['normaliserh'] = df_temp.high[0]
        df_temp['normaliserl'] = df_temp.low[0]
        df_temp['high_nmd'] = (df_temp.high / df_temp.normaliser) - 1
        df_temp['low_nmd'] = (df_temp.low / df_temp.normaliser) - 1
        df_temp['high_nmd_close'] = (df_temp.high / df_temp.normaliser) - 1
#        print(df_temp['high_nmd_close'].values)
        df_temp['low_nmd_close'] = (df_temp.low / df_temp.normaliser) - 1
        df_temp['open_nmd_close'] = (df_temp.open / df_temp.normaliser) - 1
#        df_temp['diff_span_a_b'] = (df_temp['senkou_span_a']-df_temp['senkou_span_b'])/df_temp.normaliser
#        df_temp['span_a'] = (df_temp['senkou_span_a']/df_temp.normaliser)-1
#        df_temp['conversion_line'] = (df_temp['tenkan_sen']/df_temp.normaliser)-1
#        df_temp['base_line'] = (df_temp['kijun_sen']/df_temp.normaliser)-1
        
        # Concat df_temp to df_final
        df_final = pd.concat([df_final, df_temp])
        
    # Reset Index
    df_final = df_final.reset_index(drop=True)
    
    return df_final

def pre_process_data(df,window_length=6):
    """
    Processes data for LSTM model
    
    Parameters
    --------
    df : pd.DataFrame with cols: ticker, date, close
    
    Returns
    --------
    df : pd.DataFrame
    """
    # pandas datetime
    df['date'] = pd.to_datetime(df.date)
    # Sort dataFrame
    df = df.sort_values(['ticker','date'])
    # Create empty df to be filled with windows normalised for each ticker
    df_final = pd.DataFrame([])
    # Create list of unique tickers
    tickers = list(df.ticker.unique())
    # For each ticker in df create normalised Windows
    for idx, ticker in enumerate(tickers):
        print('\ncreating normalised windows for {} ({}/{})'.format(ticker, 
                                                           idx+1,
                                                           len(tickers)), sep='')
        # Create new df with one ticker only
        df_temp = df[df.ticker == ticker]
        # Create normalised windows
        df_temp = normalise_windows(df_temp, window_length=window_length)
        # Concat df_temp to df_final
        df_final = pd.concat([df_final, df_temp])
        
    # Reset index
    df_final = df_final.reset_index(drop=True)
    
    return df_final

def pre_process_data_average(df,days_average,window_length=6):
    """
    Processes data for LSTM model
    
    Parameters
    --------
    df : pd.DataFrame with cols: ticker, date, close
    
    Returns
    --------
    df : pd.DataFrame
    """
    # pandas datetime
    df['date'] = pd.to_datetime(df.date)
    # Sort dataFrame
    df = df.sort_values(['ticker','date'])
    # Create empty df to be filled with windows normalised for each ticker
    df_final = pd.DataFrame([])
    # Create list of unique tickers
    tickers = list(df.ticker.unique())
    # For each ticker in df create normalised Windows
    for idx, ticker in enumerate(tickers):
        print('\ncreating normalised windows for {} ({}/{})'.format(ticker, 
                                                           idx+1,
                                                           len(tickers)), sep='')
        # Create new df with one ticker only
        df_temp = df[df.ticker == ticker]
        # Create normalised windows
        df_temp = normalise_windows_average(df_temp, days_average, window_length=window_length)
        # Concat df_temp to df_final
        df_final = pd.concat([df_final, df_temp])
        
    # Reset index
    df_final = df_final.reset_index(drop=True)
    
    return df_final

def normalise_windows_average(df, days_average, window_length=6):
    """
    Takes a DataFrame as input and returns a much larger DataFrame with
    normalised windows. A window of data is a run of dates of size window
    length. The number of windows created is dependant on the length of
    the DataFrame and the window size: Windows = Length - Window size.
    For each window the data is normalised by dividing by the first row of
    each window.
    
    Parameters
    --------
    df : pd.DataFrame incl columns: ticker, date, close
    window_length : size of Window = train + test
    
    Returns
    --------
    df : df with extra columns: window (window id), normaliser (for 
    de-normalisation), close_nmd (nomrlaised close price)
    """
    # Minus 5 instead of 6 due to range function used in loop
    windows = len(df) - (window_length-1)-days_average
    # Create empty dataframe to be filled with windows
    df_final = pd.DataFrame([])
    
    for i in range(0, windows):
        # Print progress counter
        print('\r {} of {}'.format(i + 1, windows), end='\r', flush=False)
        # Create a dataframe for every 6 rows
        df_temp = df[i:i + window_length+days_average]
        # Reset index
        df_temp = df_temp.reset_index(drop=True)
        # Create Window id
        df_temp['window'] = i + 1
        # Normailse close price column
        df_temp['normaliser'] = df_temp.close[0]
        df_temp['normaliserv'] = df_temp.volume[0]
        df_temp['normaliserh'] = df_temp.high[0]
        df_temp['normaliserl'] = df_temp.low[0]

        
        df_temp['close_nmd'] = (df_temp.close / df_temp.normaliser) - 1
        df_temp['volume_nmd'] = (df_temp.volume / df_temp.normaliserv) - 1
        df_temp['high_nmd'] = (df_temp.high / df_temp.normaliser) - 1
        df_temp['low_nmd'] = (df_temp.low / df_temp.normaliser) - 1
        df_temp['high_nmd_close'] = (df_temp.high / df_temp.normaliser) - 1
        df_temp['low_nmd_close'] = (df_temp.low / df_temp.normaliser) - 1
        df_temp['open_nmd_close'] = (df_temp.open / df_temp.normaliser) - 1

        
        average_value = sum(df_temp['close_nmd'].tolist()[-days_average:])/days_average
        df_temp = df_temp.drop(df_temp.index[-days_average+1:])
        df_temp.loc[len(df_temp)-1, 'close_nmd'] = average_value
        # Concat df_temp to df_final
        df_final = pd.concat([df_final, df_temp])
        
    # Reset Index
    df_final = df_final.reset_index(drop=True)
    
    return df_final

