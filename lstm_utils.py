import os

import yaml
import numpy as np
import pandas as pd

def series_to_ndarray(df, window_len, column : str, dates=False):
    """Returns numpy array of shape (1, length of window, 1)
    using pd.DataFrame as input
    """
    # Create empty list of arrays
    arrs_list = []
#    not_correct = 0
    # Stack array for each window vertically
#    if dates:
#        for window in df.window.unique():
#            # Create array and reshape
#            arr = df[df.window == window][column].values
#            if len(arr)<window_len or len(arr)>window_len:
#                not_correct = not_correct + 1 
#                
#                continue
#            
#            # Reshape: (, number of days per array, number of columns)
#            arr = arr.reshape(1, len(df[df.window == window][column]), 1)
#            # append arr to arrs_list
#            arrs_list.append(arr)
#            
#        # Use numpy vstack to create array of arrays
#        arr = np.vstack(arrs_list)
#        return arr
#        
    for window in df.window.unique():
        # Create array and reshape
        arr = df[df.window == window][column].values
#        if len(arr)<window_len or len(arr)>window_len:
#            not_correct = not_correct + 1 
            
#            continue
        
        # Reshape: (, number of days per array, number of columns)
        arr = arr.reshape(1, len(df[df.window == window][column]), 1)
        # append arr to arrs_list
        arrs_list.append(arr)
        
    # Use numpy vstack to create array of arrays
    arr = np.vstack(arrs_list)
        
    return arr

def x_y_array_split(array):
    # First five values in each window as X
    X = array[:, :-1, :]
    # Last value in each window as Y
    y = array[:, -1, :]
    
    return X, y

def train_test_split_for_x_train(array, input_dim,dates_array, window_array, ratio=0.95):
    """Takes multi-dimensional array as input and returns arrays for:
    x_train, y_train, x_test and y_test. x_test and y_test are suffled using a 
    fixed random state.
    """
    # Create copy of np array to avoid shuffling ary
    ary = np.copy(array)
    # Define where to split arr based on length
    split_row = int(ary.shape[0] * ratio)
    
    # Take first fice days of each window as x
    X = ary[:, :-1, :]
    # Split X into train and test
    x_train, x_test = np.split(X, [split_row])
    return x_train

def train_test_split(array, input_dim,dates_array, window_array, ratio=0.95):
    """Takes multi-dimensional array as input and returns arrays for:
    x_train, y_train, x_test and y_test. x_test and y_test are suffled using a 
    fixed random state.
    """
    # Create copy of np array to avoid shuffling ary
    ary = np.copy(array)
    # Define where to split arr based on length
    split_row = int(ary.shape[0] * ratio)
    
    # Take first fice days of each window as x
    X = ary[:, :-1, :]
    # Split X into train and test
    x_train, x_test = np.split(X, [split_row])
    x_train_sim = x_train
    np.random.RandomState(1).shuffle(x_train)
    # Shuffle train dataset using fixed random state
    
    
    
    # Take last day of each window as y
    y = ary[:, -1, :]
    # Split X into train and test
    y_train, y_test = np.split(y, [split_row])
    # Shuffle train dataset using fixed random state
    np.random.RandomState(1).shuffle(y_train)
    
    #dates
    
    train_days, test_days = np.split(dates_array,[split_row])
    train_days_sim_non_normal = train_days
#    np.random.RandomState(1).shuffle(train_days)
    train_windows_non_randomized, test_windows = np.split(window_array,[split_row])
    if input_dim>1.0:
        y_train = y_train[:,0]
        y_test = y_test[:,0]
    
    
    return x_train, y_train, x_test, y_test, train_days, test_days, test_windows, train_windows_non_randomized,x_train_sim

def lstm_ary_splits(df, cols=None):
    """This function makes use of train_test_split to split metric given in 
    cols. 
                                                                   
    Returns
    --------
    arys : dictionary indexed by metric, for each metric a list of arrays is 
    given
    dict_df : Handy dictionary to use when calling a specific metric in arys
    example, to get x_train for col 'close_nmd' arys
    """
    if cols == None:
        array_cols = ['window','date','close','close_nmd','normaliser']
    else:
        array_cols = cols
    
    dict_df = {'x_train':0, 'y_train':1, 'x_test':2, 'y_test':3}
    
    # Empty array dict, index will be column name
    arys = {}
    
    for i, col in enumerate(array_cols):
        print('\rsplitting column {} of {}'.format(i + 1, len(array_cols)), 
              end='\r', flush=False)
        # Use df to create multidimensional array for column
        ary = series_to_ndarray(df, column=col)
        # Split into x, y, train and test
        x_train , y_train, x_test, y_test = train_test_split(ary)
        # Append dfs to arys
        arys[col] = [x_train, y_train, x_test, y_test]
        
    return arys, dict_df

def top_x_of_dict(dictionary, x):
    """Returns the top x items of given dictionary.
    
    Parameters
    --------
    dictionary : Dictionary you would like to reduce to top x only, dict
    x : The number of items you would like to return, int
    
    Returns
    --------
    dictionary : Dictionary reduced to top x items, dict
    """
    # Sort dict keys from highest to lowest
    keys = sorted(dictionary, key=dictionary.get, reverse=True)
    # Select top x keys
    keys = keys[:x]
    # Reduce dict to only those in keys
    dictionary = {key : value for key, value in dictionary.items() if key in keys}
    
    return dictionary

def gen_X(df,column,window_length, window=0):
    # Define which window to use as X
    df = df[df.window == df.window.max() - window][-window_length:]
    # Create X based on last five days of close data
    X = series_to_ndarray(df, column=column)
    # Normailse X, by dividing all numbers in array but first number
    X_nmd = (X / X[0][0]) - 1
    
    return X, X_nmd

def get_tickers_done(directory : str):
    results_files =  os.listdir(directory)
    
    if not results_files:
        tickers = []
    else:
        df = pd.DataFrame({})
        for file in results_files:
            df = pd.concat([df, pd.read_csv(directory + file)])
    
        tickers = list(df.ticker.unique())
            
    return tickers

def load_user_from_yml(yml_file : str):
    user_settings = yaml.load(open(yml_file))
    
    return user_settings['user']

def get_tickers_for_a_user(user : str):
    owners = pd.read_csv('./industry_owners.csv')
    
    user_industries = list(owners[owners.owner == user]['industry'].unique())
    
    tickers = pd.read_csv('./tickers.csv')
    
    user_tickers = list(tickers[tickers.industry.isin(user_industries)]['ticker'].unique())
    
    return user_tickers

def read_a_user_results_csv(directory : str, user : str):
    results_csv = directory + 'model_results_{}.csv'.format(user)
    if os.path.isfile(results_csv):
        df = pd.read_csv(results_csv)
    else:
        df = pd.DataFrame({})
    
    return df

def read_all_results_csv(directory : str):
    files = os.listdir(directory)
    df = pd.DataFrame({})
    
    for file in files:
        df_temp = pd.read_csv(directory + file)
        df = pd.concat([df, df_temp])
        
    df = df[df.window_length > 0]
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        
    return df.reset_index(drop=True)