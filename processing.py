import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def _cagr(start_value, end_value, n):
    """
    calculate CAGR
    """
    return ((start_value/end_value)**(1/n)) - 1

def _cyclicalFeature(feature_column, feature_max):
    """
    Cnnverts repeating Linear features into cyclical features using point on a
    circle's circumference given by x y coordinates
    """
    # Scale features
    f_scaled = feature_column / feature_max * (2 * np.pi)
    # Retrieve corordinates
    x = np.cos(f_scaled)
    y = np.sin(f_scaled)
    
    return x, y

def prepData(df):
    """
    Preps data for ML predictions
    """
    # Ensure each row is the day after the row before
    df['next_day'] = df.date.shift(-1)
    df['day_diff'] = (df['next_day'] - df['date'])
    df['day_diff']  = df.apply(lambda row: row.day_diff.days, axis=1)
    # Select only data where the next row is the following day
    df = df[df.day_diff == 1]
    
    df = df.reset_index(drop=True)
    
    return df

    
def genFeatures(df):
    print('\nGenerating Features')
    # Empty df
    df_with_features = pd.DataFrame([])
    # For each stock in data gen features
    for ticker in df.ticker.unique():
        df_t = df[df.ticker == ticker]
        df_t = df_t.reset_index(drop=True)
        # Date Related Features
        df_t['year'] = df_t.date.dt.year
        df_t['month'] = df_t.date.dt.month
        df_t['dow'] = df_t.date.dt.dayofweek
        df_t['dom'] = df_t.date.dt.day
        df_t['weekday'] = df_t.date.dt.weekday
        df_t['doy'] = df_t.date.dt.dayofyear
        
        # Generate cyclical features
        df_t['ce_dow_x'], df_t['ce_dow_y'] = _cyclicalFeature(df_t.dow, 7)
        df_t['ce_doy_x'], df_t['ce_doy_y'] = _cyclicalFeature(df_t.doy, 365)
        df_t['ce_month_x'], df_t['ce_month_y'] = _cyclicalFeature(df_t.dow, 12)
        
        # Difference between low and high
        df_t['high_low'] = df_t['high'] - df_t['low']
        
        # Define range 
        _range = (1,5)
        # Open & close features
        for i in _range:
            # lagged features for open
            df_t['open_min' + str(i)] = df_t['open'].shift(i)
            # lagged features for close
            df_t['close_min' + str(i)] = df_t['close'].shift(i)
            # lagged features for volumne
            df_t['volume_min' + str(i)] = df_t['volume'].shift(i)
            # lagged features for high low difference
            df_t['high_low_min' + str(i)] = df_t['volume'].shift(i)
            
        # Drop shift variables where null & reset index
        df_t = df_t[~pd.isnull(df_t['close_min' + str(i)])]
        df_t = df_t.reset_index(drop=True)    
        # Calculate open CAGRs
        df_t['open_cagr'] = df_t.apply(lambda row: _cagr(row['open'],
                                                     row['open_min' + str(max(_range))],
                                                     max(_range)),
                                   axis=1)
        # Calculate close CAGRs
        df_t['close_cagr'] = df_t.apply(lambda row: _cagr(row['close'],
                                                      row['close_min' + str(max(_range))],
                                                      max(_range)),
                                   axis=1)
        
        # Concat to df_with_feature
        df_with_features = pd.concat([df_with_features, df_t])
    
    
    return df_with_features.reset_index(drop=True)

def featureProcessing(df):
    print('\nProcessing Features')
    # Scale numeric columns
    num_cols = list(df.select_dtypes(include=['float64','int']).columns)
    # Standardise numerical columns
    df_sc = pd.DataFrame(scaler.fit_transform(df[num_cols]), 
                         columns=[column + '_sc' for column in num_cols])
    # Concat standardise columns to df
    df = pd.concat([df, df_sc], axis=1)
    
    # Factorise string columns
    str_cols = list(df.select_dtypes(exclude=['float64','int']).columns)
    for col in str_cols:
        df[col + '_en'] = pd.factorize(df[col])[0]
    
    return df

def genTargets(df):
    print('\nGenerating targets')
    # Empty df
    df_with_target = pd.DataFrame([])
    # Split dataframe into sections based on ticker
    for ticker in df.ticker.unique():
        # Select data and sort
        df_t = df[df.ticker == ticker]
        df_t = df_t.sort_values(['ticker','date'])
        df_t = df_t.reset_index(drop=True)
        # Generate the target variable 
        df_t['regressor_y'] = df_t.close.shift(-1)
        df_t['classifier_y'] = np.where(df_t.close.shift(-1) > 0, 1, 0)
        # drop where target is null & reset index
        df_t = df_t[~pd.isnull(df_t.regressor_y)]
        df_t = df_t.reset_index(drop=True)
        # Concat to df_with_target
        df_with_target = pd.concat([df_with_target, df_t])
        
    # Reset index od df_with_target
    return df_with_target.reset_index(drop=True)