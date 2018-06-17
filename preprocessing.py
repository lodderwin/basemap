import pandas as pd

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
        
        #ichikomu calculation
#        df_temp['lowichigh'] = max(df_temp.close[-9:])
#        df_temp['lowiclow'] = min(df_temp.close[-9:])
#        df_temp['middleichigh'] = max(df_temp.close[-26:])
#        df_temp['middleiclow'] = min(df_temp.close[-26:])
#        df_temp['highichigh'] = max(df_temp.close[-52:])
#        df_temp['highiclow'] = min(df_temp.close[-52:])  #or whole list
#        df_temp['conversionline'] = (df_temp.lowichigh+df_temp.lowiclow)/2.
#        df_temp['baseline'] = (df_temp.middleichigh+df_temp.middleiclow)/2.
#        df_temp['leadingspana'] = (df_temp.conversionline+df_temp.baseline)/2.
#        df_temp['leadingspanb'] = (df_temp.)
        
        
        
        
        
        
        
        
        #
        # Normailse close price column
        df_temp['normaliser'] = df_temp.close[0]
        df_temp['normaliserv'] = df_temp.volume[0]
        df_temp['close_nmd'] = (df_temp.close / df_temp.normaliser) - 1
        df_temp['volume_nmd'] = (df_temp.volume / df_temp.normaliserv) - 1
        df_temp['normaliserh'] = df_temp.high[0]
        df_temp['normaliserl'] = df_temp.low[0]
        df_temp['high_nmd'] = (df_temp.high / df_temp.normaliserh) - 1
        df_temp['low_nmd'] = (df_temp.low / df_temp.normaliserl) - 1
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
    #ichimoku
    #
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
        df_temp['high_nmd'] = (df_temp.high / df_temp.normaliserh) - 1
        df_temp['low_nmd'] = (df_temp.low / df_temp.normaliserl) - 1

        
        average_value = sum(df_temp['close_nmd'].tolist()[-days_average:])/days_average
        df_temp = df_temp.drop(df_temp.index[-days_average+1:])
        df_temp.loc[len(df_temp)-1, 'close_nmd'] = average_value
        # Concat df_temp to df_final
        df_final = pd.concat([df_final, df_temp])
        
    # Reset Index
    df_final = df_final.reset_index(drop=True)
    
    return df_final
