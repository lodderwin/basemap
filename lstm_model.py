""" A collection of functions that build an lstm model and use said model to 
make predictions.
"""
import numpy as np
from numpy import newaxis
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
import datetime as dt
import plotting
import random
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import CustomBusinessDay
import time
import statistics

long_term_folder = './long_term_models/'

def build_model(params):
    """
    """
    if len(params.keys())==5:
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
    if len(params.keys())==6:
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
            return_sequences=True))
        model.add(LSTM(
            params['node3'],
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

def randomised_model_config(test_windows,df_p,test_days,train_days,train_windows_non_randomized,x_train_sim,input_dim,window_length,ticker,df,days_ahead,x_train, y_train, x_test, y_test,industry,
                            initial_investment=100, iterations=20, mcr=0.00000001,best_investment_dev=100 ,beginparams={},new_test=-1000000):
    for iteration in range(0, iterations):
        print('iteration: {} of {}'.format(iteration + 1, iterations))
        # Define params randomly
        node = np.random.randint(2,3)
        if node==2:
            params = {'input_dim':input_dim,
                      'node1':np.random.randint(20,100),
                      'node2':np.random.randint(20,100),
                      'output_dim':1,
                      'batch_size':random.choice(np.asarray([4,8,16]))}
        elif node==3:
            params = {'input_dim':input_dim,
                      'node1':np.random.randint(20,100),
                      'node2':np.random.randint(20,100),
                      'node3':np.random.randint(20,100),
                      'output_dim':1,
                      'batch_size':random.choice(np.asarray([4,8,16]))}
        # Build model
        model = build_model(params)   
        # Fit using x and y test and validate on 10% of those arrays
        # take out dates as input
        model.fit(x_train,
                  y_train,
                  validation_split = 0.1,
                  batch_size = params['batch_size'],
                  epochs = random.choice(np.asarray([2,3,4,5])))
        time.sleep(7.1) 
        # Get models MSE 
#        score = model.evaluate(x_test, y_test, verbose=0)[1]
        
        ## add simulator
#        date_today = dt.datetime.now().strftime("%Y-%m-%d")
#        real_prices = df.loc[len(df)-len(x_test):,'close'].tolist()
        df_predict = predict_test(test_windows, df_p, test_days, days_ahead,window_length, x_test, model,df)
#        df_predict_train = (train_windows_non_randomized[-400:], df_p, train_days[-400:], days_ahead,window_length, x_train_sim[-400:], model,df)
        
        margins = list(np.linspace(1.0,1.1,100))
        best_margin = 0.0
        shortterm_models = './'+industry+'/shortterm_models/'
        for margin in margins:
            investment, investment_dev,investment_dev_df, increase_correct, increase_false,mean_test,std_test,len_points = invest_sim(df_predict,df,margin,ticker)
#            investment_train, investment_dev_train,investment_dev_df_train, increase_correct_train, increase_false_train,mean_train,std_train = invest_sim(df_predict_train,df,margin,ticker)   
#            print(investment_dev_train)
            print(mean_test,std_test)
            if  ((1+(mean_test-std_test))**len_points)>new_test and  investment>300.0:
                new_test = ((1+(mean_test-std_test))**len_points)
                mcr=(investment/300.0)*(df_p['close'].tolist()[0]/df_p['close'].tolist()[-1])
                beginparams = params
                initial_investment = investment
                best_margin = margin
#                ratio = increase_correct/increase_false
                best_investment_dev = investment_dev_df
                print(investment)
                plotting.plot_investment(investment_dev,ticker,params,margin, window_length,node,industry)
#                plotting.plot_investment_train(investment_dev_train,ticker,params,margin, window_length,node)

#                plotting.plot_results(real_prices,corrected_predicted_test, days_ahead, ticker)
                model.save(shortterm_models+ticker+'_'+str(window_length)+'_model.h5', overwrite=True)

            elif (increase_correct+increase_false)==0.0:
                continue
                            
    del model   
    return initial_investment, best_investment_dev, beginparams, best_margin,mcr

def randomised_model_config_days_average(test_windows,df_p,test_days,input_dim ,window_length,ticker,df,days_average,x_train, y_train, x_test, y_test,
                            initial_investment=100., iterations=20, epochs=10):
    for iteration in range(0, iterations):
        print('iteration: {} of {}'.format(iteration + 1, iterations))
        # Define params randomly
        params = {'input_dim':2,
                  'node1':np.random.randint(15,80),
                  'node2':np.random.randint(15,80),
                  'output_dim':1,
                  'batch_size':random.choice(np.asarray([16,32]))}
#        params = {'input_dim':1,
#                  'node1':15,
#                  'node2':45,
#                  'output_dim':1,
#                  'batch_size':32}
        
        # Build model
        model = build_model(params)
        
        # Fit using x and y test and validate on 10% of those arrays
        model.fit(x_train,
                  y_train,
                  validation_split = 0.1,
                  batch_size = params['batch_size'],
                  epochs = random.choice(np.asarray([6,12,18])))
    
        # Get models MSE 
#        score = model.evaluate(x_test, y_test, verbose=0)[1]
        
#        ## add simulator
#        date_today = dt.datetime.now().strftime("%Y-%m-%d")
#        real_prices = df.loc[len(df)-len(x_test):,'close'].tolist()
        df_predict = predict_test_days_average(test_windows,test_days, df_p,x_test,window_length, days_average, model, df)
        print(df_predict)
        margins = list(np.linspace(1.0,1.1,40))
        
        for margin in margins:
            investment, investment_dev,investment_dev_df = invest_sim(df_predict,df,margin,ticker)
            
            if initial_investment < investment:
                initial_investment = investment
                best_investment_dev = investment_dev_df
                print(investment)
                plotting.plot_investment(investment_dev,ticker,params,margin)
    #            plotting.plot_results(real_prices,corrected_predicted_test, days_ahead, ticker)
                model.save(long_term_folder+ticker+'_'+str(window_length)+'_'+str(days_average)+'_model.h5', overwrite=True)            
    del model        
    print('Loading model')
    best_model = load_model(long_term_folder+ticker+'_'+str(window_length)+'_'+str(days_average)+'_model.h5')
       
    return best_model, investment, best_investment_dev


def predict(model, X, days_ahead):
    """X being x_train or x_test
    """
    # How many days should the model predict ahead for?
    days_ahead = days_ahead
    
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
def predict_single(model, X, days_ahead):
    days_ahead = days_ahead
    prediction = []
    current_frame = X
    for day in range(days_ahead):
        prediction.append((model.predict(current_frame[newaxis,:,:])[0,0]))
    return prediction



def predict_test(test_windows,df_p,test_days, days_ahead,window_length, x_test, model,df):
    df_predict = pd.DataFrame([])
#    y_date_first = pd.to_datetime(test_days[0][-1])[0]
    predictions_in_function = int(x_test.shape[0]/days_ahead)
    #take out unique values in df_p for window--> normalizer
    df_p_window_index = df_p.set_index(df_p['window'])    
    
    
    ###custom calendar
    weekmask = 'Mon Tue Wed Thu Fri'
    holidays = [datetime(2016, 3, 30), datetime(2016, 5, 28), datetime(2016, 7, 4), datetime(2016, 5, 28),
                datetime(2016, 7, 4), datetime(2016, 9, 3), datetime(2016, 11, 22), datetime(2016, 12, 25),
                datetime(2017, 3, 30), datetime(2017, 5, 28), datetime(2017, 7, 4), datetime(2017, 5, 28),
                datetime(2017, 7, 4), datetime(2017, 9, 3), datetime(2017, 11, 22), datetime(2017, 12, 25),
                datetime(2018, 3, 30), datetime(2018, 5, 28), datetime(2018, 7, 4), datetime(2018, 5, 28),
                datetime(2018, 7, 4), datetime(2018, 9, 3), datetime(2018, 11, 22), datetime(2018, 12, 25)]
    bday_cust = CustomBusinessDay(holidays=holidays, weekmask=weekmask) 
    ###
    for i in range(predictions_in_function):
        current_frame = x_test[i*days_ahead]
        y_date = pd.to_datetime(test_days[i*days_ahead][-1])[0]
        window = test_windows[i*days_ahead][0][0]
        normaliser = df_p_window_index.loc[window,'normaliser']
        predicted = []
        for j in range(days_ahead):
    #4 days predicted ahead with predicted values!
    #model.predict only accepts 3 dimension numpy matrices therefore newaxis
            predicted.append( (model.predict(current_frame[newaxis,:,:])[0,0]))
            predicted_value = predicted[-1]
            df_predict = df_predict.append({'date': pd.to_datetime(y_date+bday_cust*j), 'y_predict': (predicted_value+1)*normaliser.unique()[0] },ignore_index=True)
            current_frame = current_frame[1:]
            current_frame = np.insert(current_frame, [len(x_test[0])-1], predicted[-1], axis=0)
            
    return df_predict


def predict_test_days_average(test_windows,test_days, df_p,x_test,window_length, days_average, model, df):
    df_predict = pd.DataFrame([])
#    y_date_first = pd.to_datetime(test_days[0][-1])[0]
    predictions_in_function = int(x_test.shape[0]/days_average)
    #take out unique values in df_p for window--> normalizer
    df_p_window_index = df_p.set_index(df_p['window'])    
    
    
    ###custom calendar
    weekmask = 'Mon Tue Wed Thu Fri'
    holidays = [datetime(2017, 3, 30), datetime(2017, 5, 28), datetime(2017, 7, 4), datetime(2017, 5, 28),
                datetime(2017, 7, 4), datetime(2017, 9, 3), datetime(2017, 11, 22), datetime(2017, 12, 25),
                datetime(2018, 3, 30), datetime(2018, 5, 28), datetime(2018, 7, 4), datetime(2018, 5, 28),
                datetime(2018, 7, 4), datetime(2018, 9, 3), datetime(2018, 11, 22), datetime(2018, 12, 25)]
    bday_cust = CustomBusinessDay(holidays=holidays, weekmask=weekmask) 
    ###
    for i in range(predictions_in_function):
        current_frame = x_test[i*days_average]
        y_date = pd.to_datetime(test_days[i*days_average][-1])[0]
        window = test_windows[i*days_average][0][0]
        normaliser = df_p_window_index.loc[window,'normaliser']
        predicted = []
        for j in range(5):
    #4 days predicted ahead with predicted values!
    #model.predict only accepts 3 dimension numpy matrices therefore newaxis
            predicted.append( (model.predict(current_frame[newaxis,:,:])[0,0]))
            predicted_value = predicted[-1]
            df_predict = df_predict.append({'date': pd.to_datetime(y_date+bday_cust*j), 'y_predict': (predicted_value+1)*normaliser.unique()[0] },ignore_index=True)            
    return df_predict
    
    
    
    
    
        
def invest_sim(df_predict, df,margin,ticker):
    df_index = df.set_index(df['date'])
    df_predict = df_predict.set_index(df_predict['date'])
    df_merge = pd.merge(df_index, df_predict, how='left', left_on=[np.asarray(df_index.index.year).astype(np.str), np.asarray(df_index.index.month).astype(np.str),np.asarray(df_index.index.day).astype(np.str)],
                        right_on=[np.asarray(df_predict.index.year).astype(np.str),np.asarray(df_predict.index.month).astype(np.str),np.asarray(df_predict.index.day).astype(np.str)])
    df_merge = df_merge.dropna(subset=['y_predict'])
    df_merge = df_merge.dropna(subset=['close'])
    df_merge = df_merge.reset_index(drop=True)
    investment = 300.0
    fee_per_stock = 0.005
    fee = 0.60
    dummy = 0
    investment_dev = []
    end_index = len(df_merge)-1
    dct_df = {}
    increase_correct = 0
    increase_false = 0
    decrease_correct = 0
    decrease_false = 0
    distribution = []
    dct_df['dates'] = [df_merge.loc[0,'date_x']]
    dct_df['y_predict_'+ticker] = [df_merge.loc[0,'close']]
    dct_df['close_real_'+ticker] = [df_merge.loc[0,'close']]
    for index, row in df_merge.iterrows():
        if index<end_index:
            dct_df['dates'].append(df_merge.loc[index+1, 'date_x'])
            dct_df['y_predict_'+ticker].append(df_merge.loc[index+1, 'y_predict'])
            dct_df['close_real_'+ticker].append(df_merge.loc[index+1, 'close'])

            if (df_merge.loc[index+1, 'y_predict']/df_merge.loc[index, 'close'])>margin and dummy==0 :
                investment = investment - (int(investment/df_merge.loc[index, 'close'])*fee_per_stock+fee)
                distribution.append(((df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close']))-1)
                investment = investment*(df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])
                dummy = 1
                if (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])>1.0 :
                    increase_correct = increase_correct+1
                elif (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])<1.0 :
                    increase_false = increase_false+1
#                    print(increase_false)
            elif df_merge.loc[index+1, 'y_predict']>df_merge.loc[index, 'close']>margin and dummy==1:
                distribution.append((df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])-1)
                investment = investment*(df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])
                if (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])>1.0 :
                    increase_correct = increase_correct+1
                elif (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])<1.0 :
                    increase_false = increase_false+1
            elif (df_merge.loc[index+1, 'y_predict']<df_merge.loc[index, 'close'])<margin and dummy==1:
                investment = investment-(int(investment/df_merge.loc[index, 'close'])*fee_per_stock+fee)
                dummy = 0
                if (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])<1.0 :
                    decrease_correct = decrease_correct+1
                elif (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])>1.0 :
                    decrease_false = decrease_false+1
            elif (df_merge.loc[index+1, 'y_predict']<df_merge.loc[index, 'close'])<margin and dummy==0:
                investment = investment
                if (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])<1.0 :
                    decrease_correct = decrease_correct+1
                elif (df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])>1.0 :
                    decrease_false = decrease_false+1
                #### dict to datafram, output dataframe!
        investment_dev.append(investment)      
    
    investment_dev_df = pd.DataFrame(dct_df)
    if len(distribution)>5:
        return investment, investment_dev, investment_dev_df, increase_correct, increase_false,statistics.mean(distribution),statistics.stdev(distribution),len(distribution)
    else:
        return investment, investment_dev, investment_dev_df, increase_correct, increase_false,-100,100000,len(distribution)
            
        



def invest_sim_days_average(df_predict, df,margin,ticker):
    df_index = df.set_index(df['date'])
    df_predict = df_predict.set_index(df_predict['date'])
    df_merge = pd.merge(df_index, df_predict, how='left', left_on=[np.asarray(df_index.index.year).astype(np.str), np.asarray(df_index.index.month).astype(np.str),np.asarray(df_index.index.day).astype(np.str)],
                        right_on=[np.asarray(df_predict.index.year).astype(np.str),np.asarray(df_predict.index.month).astype(np.str),np.asarray(df_predict.index.day).astype(np.str)])
    df_merge = df_merge.dropna(subset=['y_predict'])
    df_merge = df_merge.dropna(subset=['close'])
    df_merge = df_merge.reset_index(drop=True)
    investment = 300.0
    fee_per_stock = 0.005
    fee = 0.60
    dummy = 0
    investment_dev = []
    end_index = len(df_merge)-1
    dct_df = {}
    
    dct_df['dates'] = [df_merge.loc[0,'date_x']]
    dct_df['y_predict_'+ticker] = [df_merge.loc[0,'close']]
    dct_df['close_real_'+ticker] = [df_merge.loc[0,'close']]
    for index, row in df_merge.iterrows():
        if index<end_index:
            dct_df['dates'].append(df_merge.loc[index+1, 'date_x'])
            dct_df['y_predict_'+ticker].append(df_merge.loc[index+1, 'y_predict'])
            dct_df['close_real_'+ticker].append(df_merge.loc[index+1, 'close'])

            if (df_merge.loc[index+1, 'y_predict']/df_merge.loc[index, 'close'])>margin and dummy==0 :
                investment = investment - (int(investment/df_merge.loc[index, 'close'])*fee_per_stock+fee)
                investment = investment*(df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])
                dummy = 1
            elif df_merge.loc[index+1, 'y_predict']>df_merge.loc[index, 'close']>margin and dummy==1:
                investment = investment*(df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])   
            elif (df_merge.loc[index+1, 'y_predict']<df_merge.loc[index, 'close'])<margin and dummy==1:
                investment = investment-(int(investment/df_merge.loc[index, 'close'])*fee_per_stock+fee)
                dummy = 0
            elif (df_merge.loc[index+1, 'y_predict']<df_merge.loc[index, 'close'])<margin and dummy==0:
                investment = investment
                #### dict to datafram, output dataframe!
        investment_dev.append(investment)      
    
    investment_dev_df = pd.DataFrame(dct_df)
    return investment, investment_dev, investment_dev_df

def fluc_check(graph):
    total_log = 0
    for i in range(len(graph)-1):
        total_log = (np.log10(graph[i+1])-np.log10(graph[i]))**2 + total_log
    return total_log
        
        
        