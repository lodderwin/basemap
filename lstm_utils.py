""" A number of functions built to prep data for lstm predictions using Keras.
"""
import numpy as np

def series_to_ndarray(df, column : str):
    """Returns numpy array of shape (1, length of window,1)
    using pd.DataFrame as input
    """
    # Create empty list of arrays
    arrs_list = []
      
    # Stack array for each window vertically
    for window in df.window.unique():
        # Create array and reshape
        arr = df[df.window == window][column].values
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

def train_test_split(array, ratio=0.9):
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
    # Shuffle train dataset using fixed random state
    np.random.RandomState(0).shuffle(x_train)
    
    # Take last day of each window as y
    y = ary[:, -1, :]
    # Split X into train and test
    y_train, y_test = np.split(y, [split_row])
    # Shuffle train dataset using fixed random state
    np.random.RandomState(0).shuffle(y_train)
    
    return x_train, y_train, x_test, y_test

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
        print('\rsplitting column ' + str(i + 1) + ' of ' + str(len(array_cols)), 
              end='\r', flush=False)
        # Use df to create multidimensional array for column
        ary = series_to_ndarray(df, column=col)
        # Split into x, y, train and test
        x_train , y_train, x_test, y_test = train_test_split(ary)
        # Append dfs to arys
        arys[col] = [x_train, y_train, x_test, y_test]
        
    return arys, dict_df