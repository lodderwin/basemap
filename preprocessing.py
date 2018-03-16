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
    windows = len(df) - 5
    # Create empty dataframe to be filled with windows
    df_final = pd.DataFrame([])
    
    for i in range(0, windows):
        # Print progress counter
        print('\r' + str(i + 1) + ' of ' + str(windows), end='\r', flush=False)
        # Create a dataframe for every 6 rows
        df_temp = df[i:i + window_length]
        # Reset index
        df_temp = df_temp.reset_index(drop=True)
        # Create Window id
        df_temp['window'] = i + 1
        # Normailse close price column
        df_temp['normaliser'] = df_temp.close[0]
        df_temp['close_nmd'] = df_temp.close / df_temp.normaliser
        # Concat df_temp to df_final
        df_final = pd.concat([df_final, df_temp])
        
    # Reset Index
    df_final = df_final.reset_index(drop=True)
    
    return df_final

def pre_process_data(df):
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
        print('creating normalised windows for ', ticker, 
              ' (', idx+1, '/',len(tickers), ')', sep='')
        # Create new df with one ticker only
        df_temp = df[df.ticker == ticker]
        # Create normalised windows
        df_temp = normalise_windows(df_temp)
        # Concat df_temp to df_final
        df_final = pd.concat([df_final, df_temp])
        
    # Reset index
    df_final = df_final.reset_index(drop=True)
    
    return df_final
