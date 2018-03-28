""" A collection of functions that build an lstm model and use said model to 
make predictions.
"""
import numpy as np
from numpy import newaxis
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
import datetime as dt

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

def randomised_model_config(ticker,x_train, y_train, x_test, y_test,
                            mse=10, iterations=20, epochs=10):
    for iteration in range(0, iterations):
        print('iteration: {} of {}'.format(iteration + 1, iterations))
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
                  epochs = epochs)
    
        # Get models MSE 
        score = model.evaluate(x_test, y_test, verbose=0)[1]
        ## add simulator
        date_today = dt.datetime.now().strftime("%Y-%m-%d")
        if score < mse:
            mse = score
            model.save(date_today+'_'+ticker+'_model.h5', overwrite=True)
    best_model = load_model(date_today+'_'+ticker+'_model.h5')
       
    return best_model, mse


def predict(model, X):
    """X being x_train or x_test
    """
    # How many days should the model predict ahead for?
    days_ahead = 5
    
    predictions = []
    
    for idx, window in enumerate(X):
        print('\rpredicting window {} of {}'.format(idx + 1, X.shape[0]),
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

def predict_test(days_ahead, x_test, model,df):
    predicted_test = []
    predictions_in_function = int(x_test.shape[0]/days_ahead)
    for i in range(predictions_in_function):
        current_frame = x_test[i*days_ahead]
        predicted = []
        for j in range(days_ahead):
    #4 days predicted ahead with predicted values!
    #model.predict only accepts 3 dimension numpy matrices therefore newaxis
            predicted.append((model.predict(current_frame[newaxis,:,:])[0,0]))
            current_frame = current_frame[1:]
    # use predicted values for future predictions
            current_frame = np.insert(current_frame, [len(x_test[0])-1], predicted[-1], axis=0)
        predicted_test.append(predicted)
    corrected_predicted_test = []
    for i in range(predictions_in_function):
        correct_predicted = []
        for j in range(len(predicted_test[0])):
            value = predicted_test[i][j]*df.loc[-len(x_test)-days_ahead+i*days_ahead,'close']
            correct_predicted.append(value)
        corrected_predicted_test.append(correct_predicted)
        
    return corrected_predicted_test