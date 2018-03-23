""" A collection of functions that build an lstm model and use said model to 
make predictions.
"""
import numpy as np
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

def build_model(params):
    """
    """
    #
    model = Sequential()
    #
    model.add(LSTM(
        input_dim = params['input_dim'],
        output_dim = params['node1'],
        return_sequences=True))
    #
    model.add(Dropout(0.2))
    #
    model.add(LSTM(
        params['node2'],
        return_sequences=False))
    #
    model.add(Dropout(0.2))
    #
    model.add(Dense(
        output_dim = params['output_dim']))
    model.add(Activation("linear"))
    
    # Compile model
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mse'])
    
    return model

def randomised_model_config(x_train, y_train, x_test, y_test, 
                            mse=10, iterations=1):
    """
    """
    for iteration in range(0, iterations):
        print('iteration: ', iteration + 1, 'of', iterations)
        # Define params randomly
        params = {'input_dim':1,
                  'node1':np.random.randint(10,20),
                  'node2':np.random.randint(35,45),
                  'output_dim':1,
                  'batch_size':np.random.randint(10,40)}
        
        # Build model
        model = build_model(params)
        
        # Fit using x and y test and validate on 10% of those arrays
        model.fit(x_train,
                  y_train,
                  validation_split = 0.1,
                  batch_size = params['batch_size'],
                  epochs=1)
    
        # Get models MSE 
        score = model.evaluate(x_test, y_test, verbose=0)[1]
        
        if score < mse:
            mse = score
            best_model = model
            
    return best_model, mse

def predict(model, X):
    """X being x_train or x_test
    """
    # How many days should the model predict ahead for?
    days_ahead = 5
    
    predictions = []
    
    for idx, window in enumerate(X):
        print('\rpredicting window ' + str(idx + 1) + ' of ' + str(X.shape[0]),
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
        
    # predictions is a list, create np array with shape matching X
    predictions = np.reshape(predictions, X.shape)
        
    return predictions