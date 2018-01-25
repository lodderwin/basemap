import pandas as pd
import numpy as np

def _cagr(start_value, end_value, n):
    """
    calculate CAGR
    """
    return ((start_value/end_value)**(1/n)) - 1

    
def genFeatures(df):
    # Date Related Features
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['dayofweek'] = df.date.dt.dayofweek
    df['dayofmonth'] = df.date.dt.day
    df['weekday'] = df.date.dt.weekday
    df['dayofyear'] = df.date.dt.dayofyear
    
    # Difference between low and high
    df['high_low'] = df['high'] - df['low']
    
    # Define range 
    _range = (1,5)
    # Open & close features
    for i in _range:
        # lagged features for open
        df['open_min' + str(i)] = df['open'].shift(i)
        # lagged features for close
        df['close_min' + str(i)] = df['close'].shift(i)
        # lagged features for volumne
        df['volume_min' + str(i)] = df['volume'].shift(i)
        # lagged features for high low difference
        df['high_low_min' + str(i)] = df['volume'].shift(i)
        
    # Drop shift variables are null & reset index
    df = df[~pd.isnull(df['close_min' + str(i)])]
    df = df.reset_index(drop=True)    
    # Calculate open CAGRs
    df['open_cagr'] = df.apply(lambda row: _cagr(row['open'],
                                                 row['open_min' + str(max(_range))],
                                                 max(_range)),
                               axis=1)
    # Calculate close CAGRs
    df['close_cagr'] = df.apply(lambda row: _cagr(row['close'],
                                                  row['close_min' + str(max(_range))],
                                                  max(_range)),
                               axis=1)
    
    return df

def featureProcessing(df):
    # Scale numeric columns
    num_cols = df1.select_dtypes(include=['float64','int']).columns
    for col in num_cols:
        df[col + 'sc'] = 
    
    # Encode string columns
    str_cols = df1.select_dtypes(exclude=['float64','int']).columns
    for col in str_cols:
        df[col + 'en'] = 


def genTarget(df):
    # Generate the target variable 
    df['y'] = df.close.shift(-1)
    # drop where target is null & reset index
    df = df[~pd.isnull(df.y)]
    df = df.reset_index(drop=True)    
    
    return df

#%%
import finance_data as fd
fd = fd.finance_data()
df = fd.getData()

#%%
df1 = genFeatures(df)
df1 = genTarget(df1)

#%%
df1.head()