import yahoo_reader as yr
import preprocessing as pp
import pandas as pd
import numpy as np
<<<<<<< HEAD
=======
from scipy.stats import randint

>>>>>>> 4a5df7853c2e8d5df1633214f38dac34aec74368

#%%
yr_data = yr.finance_data()
df = yr_data.get_data()
#df = pd.read_csv('csv/store_data.csv')
#df = df.drop('Unnamed: 0', axis=1)

#%%
# Prep data for LSTM model
df = df[df.ticker == 'AAPL'].reset_index(drop=True)
df = pp.pre_process_data(df)

#%%

def series_to_ndarray(df, column : str):
<<<<<<< HEAD
    """Returns numpar array using pd.DataFrame as input
=======
    """Returns numpy array of shape (1,6,1) using pd.DataFrame as input
>>>>>>> 4a5df7853c2e8d5df1633214f38dac34aec74368
    """
    # Create empty list of arrays
    arrs_list = []
      
    # Stack array for each window vertically
    for window in df.window.unique():
        # Create array and reshape
        arr = df[df.window == window][column].values
        # Reshape: (, number of days per array, number of columns)
        arr = arr.reshape(1, 6, 1)
        # append arr to arrs_list
        arrs_list.append(arr)
        
    # Use numpy vstack to create array of arrays
    arr = np.vstack(arrs_list)
        
    return arr

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
         
    
def lstm_ary_splits(cols=None):
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

from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, model_from_yaml
import lstm
import time


<<<<<<< HEAD
def build_model(layers):
=======
def build_model(params):
>>>>>>> 4a5df7853c2e8d5df1633214f38dac34aec74368
    """
    """
    #
    model = Sequential()
    #
    model.add(LSTM(
<<<<<<< HEAD
        input_dim=layers[0],
        output_dim=layers[1],
=======
        input_dim = params['input_dim'],
        output_dim = params['node1'],
>>>>>>> 4a5df7853c2e8d5df1633214f38dac34aec74368
        return_sequences=True))
    #
    model.add(Dropout(0.2))
    #
    model.add(LSTM(
<<<<<<< HEAD
        layers[2],
=======
        params['node2'],
>>>>>>> 4a5df7853c2e8d5df1633214f38dac34aec74368
        return_sequences=False))
    #
    model.add(Dropout(0.2))
    #
    model.add(Dense(
<<<<<<< HEAD
        output_dim=layers[3]))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
=======
        output_dim = params['output_dim']))
    model.add(Activation("linear"))
    
    # Compile model
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mse'])
>>>>>>> 4a5df7853c2e8d5df1633214f38dac34aec74368
    
    return model

#%%
# Split into x, y, train and test
arys, dict_df = lstm_ary_splits()

#%%
<<<<<<< HEAD
# Build Model and fit
model = build_model([1,15,40,1])
batch_size=32
model.fit(arys['close_nmd'][0],
          arys['close_nmd'][1],
          batch_size=batch_size,
          nb_epoch=1,
          validation_split=0.05)
=======
#
def randomised_model_config(mse=10, iterations=10):
    """
    """
    for iteration in range(0, iterations):
        print('iteration: ', iteration + 1, 'of', iterations)
        params = {'input_dim':1,
                  'node1':np.random.randint(10,20),
                  'node2':np.random.randint(35,45),
                  'output_dim':1,
                  'batch_size':np.random.randint(10,40)}
        
        # Build Model
        model = build_model(params)
        # Fit using x and y test and validate on x and y test
        model.fit(arys['close_nmd'][0],
                  arys['close_nmd'][1],
                  validation_data = (arys['close_nmd'][2], arys['close_nmd'][3]),
                  batch_size = params['batch_size'],
                  nb_epoch=10)
        # Get models MSE 
        score = model.evaluate(arys['close_nmd'][2], arys['close_nmd'][3], verbose=0)[1]
        
        if score < mse:
            mse  = score
            best_model = model
            
    return best_model
    
model = randomised_model_config()
>>>>>>> 4a5df7853c2e8d5df1633214f38dac34aec74368

#%%
# How many days should the model predict ahead for?
days_ahead = 5

predictions = []

for idx, window in enumerate(arys['close_nmd'][0]):
    print('\rpredicting window ' + str(idx + 1) + ' of ' + str(arys['close_nmd'][0].shape[0]),
          end='\r', flush=False)
    # Reshape window as lstm predict needs np array of shape (1,5,1)
    window = np.reshape(window, (1, window.shape[0], window.shape[1]))
    for day in range(days_ahead):
        # Predict & extract prediction from resulting array
        prediction = model.predict(window)[0][0]
        # Reshape prediction so it can be appened to window
        prediction = np.reshape([prediction], (1, 1, 1))
        # Add new value to window and remove first value
        window = np.append(window, prediction)[1:]
        # Above appendage flattens np array so needs to be rehaped
        window = np.reshape(window, (1, window.shape[0], 1))

    # Result of inner loop is a window of five days no longer containing any original days   
    predictions.append(window)


#%%
# The first set of predictions are
predictions[0]
# The normaliser for this set of predictions is
arys['normaliser'][0][0]
# That means the actual prices for those predictions are
arys['normaliser'][0][0] * predictions[0]
# We can also see what window these predictions were made for
arys['window'][0][0]
# And we can check our original data for this window
df[df.window == 3452][['date','ticker','close','window','normaliser']]
# The next five days are predicted to do the following
arys['normaliser'][0][0] * predictions[0]



