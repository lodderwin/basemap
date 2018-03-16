#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:00:12 2018

@author: erwinlodder
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_yaml
#import remaining necessary modules
import lstm, time   #back up functions
import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from datetime import datetime
import matplotlib
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import CustomBusinessDay
import yaml
import os

# Define plot folder directory as today's date
today = datetime.now().strftime("%Y-%m-%d")
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# If the folder with today's date doesnt exist then make it
plot_folder = './plots/' + today + '/'
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

# Configure plot size 
figsize = (11,11*9/16)

def multiply_lst(lst):
    a = 1
    for i in range(len(lst)):
        a = lst[i] * a
    return a
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data
def normalise_data(data):
    normalised_data = []
    for dat in data:
        normalised_data.append(dat/data[0])
    return normalised_data
#data in list format
def create_sets(data, seq_len, normalise_window):
    sequence_length = seq_len + 1 #the +1 is needed to select 'test day' which will become y
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    if normalise_window:
        result = normalise_windows(result)
    
    
    #
    result = np.array(result)
    #here 90% of the data is chosen to train on, the remaining 10% will be used to test
    row = round(0.95 * result.shape[0])
    train = result[:int(row)]
    #randomize and shuffle data to get rid of patterns staying the same through time
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
    #correction list for denormalizing data
    y_test_correction = normalise_data(data[int(row):])
    #for future plots, exact dates might come in handy
    return x_train, y_train, x_test, y_test, y_test_correction

def build_model(layers):
    
    model = Sequential()
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
#    print("Compilation Time : ", time.time() - start)
    return model
def save_model(stock, model):
    model_yaml = model.to_yaml()
    with open('model.yaml', 'w') as outfile:
        yaml.dump(model_yaml, outfile)
#    model.save_weights('model.h5')
    print('model saved')

def open_model(stock):
    with open('model.yaml', 'r') as inpfile:
        yamlRec4=yaml.load(inpfile)
    model4 = model_from_yaml(yamlRec4)
    model4.summary() # will print
    print('model loaded')
    return model4
    
    
def predict_test(days_ahead, x_test, seq_len, model):
    predicted_test = []
    predictions_in_function = int(x_test.shape[0]/days_ahead)
#    remaining_predictions = x_test.shape[0]%days_ahead
    for i in range(predictions_in_function):
    #current_frame is x values used to predict y
        current_frame = x_test[i*days_ahead]
        predicted = []
        for j in range(days_ahead):
    #4 days predicted ahead with predicted values!
    #model.predict only accepts 3 dimension numpy matrices therefore newaxis
            predicted.append((model.predict(current_frame[newaxis,:,:])[0,0]))
            current_frame = current_frame[1:]
    # use predicted values for future predictions
            current_frame = np.insert(current_frame, [seq_len-1], predicted[-1], axis=0)
        predicted_test.append(predicted)
    return predicted_test

def predict_test_day(days_ahead, x_test, seq_len, model):
    predicted_test = []
    predictions_in_function = int(x_test.shape[0]/days_ahead)
#    remaining_predictions = x_test.shape[0]%days_ahead
    for i in range(len(x_test)):
    #current_frame is x values used to predict y
        current_frame = x_test[i]
        predicted = []
        for j in range(days_ahead):
    #4 days predicted ahead with predicted values!
    #model.predict only accepts 3 dimension numpy matrices therefore newaxis
            predicted.append((model.predict(current_frame[newaxis,:,:])[0,0]))
            current_frame = current_frame[1:]
    # use predicted values for future predictions
            current_frame = np.insert(current_frame, [seq_len-1], predicted[-1], axis=0)
        predicted_test.append(predicted)
    return predicted_test

def predict_current(seq_len,days_ahead, data, model):
    predicted = []
    current_frame = []
    for d in data:
        current_frame.append([d/data[0]-1])
#    current_frame = normalise_data(data)
#    current_frame[:] = [x - 1 for x in current_frame]
    current_frame  = np.asarray(current_frame)
    for j in range(days_ahead):
        predicted.append((model.predict(current_frame[newaxis,:,:])[0,0]))
        current_frame = current_frame[1:]
        current_frame = np.insert(current_frame, [seq_len-1], predicted[-1], axis=0)
    return predicted
def predict_current_corrected(current_prediction, y_test_correction, seq_len):
    current_prediction_corrected = []
    for j in range(len(current_prediction)):
#        temp_pred = [x+1 for x in current_prediction[:j+1]]
#        multiply = multiply_lst(temp_pred)
#        if j<1.0:
        current_prediction_corrected.append((current_prediction[j]+1)*y_test_correction[-seq_len])
    return current_prediction_corrected
        
def plot_current(y_test_correction,predicted,stock):
#    fig = plt.figure(facecolor='white')
#    ax = fig.add_subplot(111)
#    ax.plot(y_test_correction, label='True Data', color='b')
#    #padding is used to set the new predictions to an appropiate distance from 0 days
#    padding = [None for p in list(range(len(y_test_correction)))]
#    if (predicted[-1]-predicted[0])>0.0:
#        plt.plot(padding+predicted, label='Prediction', alpha=0.6, color='g')
#    elif (predicted[-1]-predicted[0])<0.0:
#        plt.plot(padding+predicted, label='Prediction', alpha=0.6, color='r')
#        
#    plt.xlabel('days')
#    plt.savefig(stock+'_current_prediction.png',dpi=400)
#    plt.show()
    # Use seaborn styling
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(y_test_correction, label='True Data', color='b')
    #padding is used to set the new predictions to an appropiate distance from 0 days
    padding = [None for p in list(range(len(y_test_correction)))]
    if (predicted[-1]-predicted[0])>0.0:
        plt.plot(padding+predicted, label='Prediction', alpha=0.6, color='g')
    elif (predicted[-1]-predicted[0])<0.0:
        plt.plot(padding+predicted, label='Prediction', alpha=0.6, color='r')
    
    plt.legend(fontsize=13)
    plt.title('Latest Prediction for '+ stock, size=16)
    plt.xlabel('days', size=13)
    plt.ylabel('share price', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    plt.savefig(plot_folder + stock + '_current_prediction.png',dpi=400)
    plt.show()
    plt.close()
        
#def predict_corrected(predicted_test,y_test_correction):
#    corrected_predicted_test = []
#    for i in range(len(predicted_test)):
#        corrected_predicted = []
#        for j in range(len(predicted_test[0])):
#            temp_pred = [x+1 for x in predicted_test[i][:j+1]]
#            multiply = multiply_lst(temp_pred)
#            corrected_predicted.append(multiply*y_test_correction[i*seq_len+j])
#        corrected_predicted_test.append(corrected_predicted)
#    return corrected_predicted_test
def correct_predict_test(days_ahead, predicted_test, y_test_correction, seq_len):
    corrected_predicted_test = []
    for i in range(len(predicted_test)):
        corrected_predicted = []
    ##include first point
        for j in range(days_ahead):
#            temp_pred = [x+1 for x in predicted_test[i][:j+1]]
#            multiply = multiply_lst(temp_pred)
    #        if j<1.0:
            corrected_predicted.append((predicted_test[i][j]+1)*y_test_correction[(i*days_ahead-seq_len)])
    #        else :
    #            corrected_predicted.append(multiply*y_test_correction[i*seq_len])
        corrected_predicted_test.append(corrected_predicted)   
    return corrected_predicted_test
def correct_predict_test_day(days_ahead, predicted_test, y_test_correction,seq_len):
    corrected_predicted_test = []
    for i in range(len(predicted_test)):
        corrected_predicted = []
    ##include first point
        for j in range(days_ahead):
#            temp_pred = [x+1 for x in predicted_test[i][:j+1]]
#            multiply = multiply_lst(temp_pred)
    #        if j<1.0:
            corrected_predicted.append((predicted_test[i][j]+1)*y_test_correction[(i-seq_len)])
    #        else :
    #            corrected_predicted.append(multiply*y_test_correction[i*seq_len])
        corrected_predicted_test.append(corrected_predicted)   
    return corrected_predicted_test
def plot_results(y_test_correction, corrected_predicted_test, prediction_len,stock):
    # Use seaborn styling
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(y_test_correction, label='True Data')
    #padding is used to set the new predictions to an appropiate distance from 0 days
    for i, data in enumerate(corrected_predicted_test):
            padding = [None for p in list(range(int(((i) * prediction_len)/1.0)))]
            plt.plot(padding+data, label='Prediction', alpha=0.6)

    plt.title('Predictions for ' + stock, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Stock Price', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    plt.savefig(plot_folder + stock + '_predictions.png',dpi=400)
    plt.show()
    plt.close()
def plot_results_day(y_test_correction, corrected_predicted_test, prediction_len,stock,corrected_predicted_test_day):
    # Use seaborn styling
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(y_test_correction, label='True Data', color='b')
    #padding is used to set the new predictions to an appropiate distance from 0 days
    for i, data in enumerate(corrected_predicted_test):
            padding = [None for p in list(range(int(((i) * prediction_len)/1.0)))]
            plt.plot(padding+data, label='Prediction', alpha=0.95, color='#FF9933')
    for i, data in enumerate(corrected_predicted_test_day):
            padding = [None for p in list(range(int(((i))/1.0)))]
            plt.plot(padding+data, label='Prediction', alpha=0.4)
    plt.title('Predictions for ' + stock, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Stock Price', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    plt.savefig(plot_folder + stock + '_compare_predictions.png',dpi=400)
    plt.show()
    plt.close()
def invest_sim(corrected_predicted_test, y_test_correction):
    flat_predictions = np.asanyarray([item for sublist in corrected_predicted_test for item in sublist])
    compare_num_days = len(flat_predictions)
    compare_test = np.asarray(y_test_correction)[:compare_num_days]
    compare_test = compare_test
    mse = np.mean((flat_predictions-compare_test)**2)
    print(np.sqrt(mse))
    investment = 1000.0
    dummy = 0
    fee = 5.0
    bs = 0
    check_lst = []
    b=4
    for i in range(60):
        
        check_lst.append(b)
        b = b+5
    correct_guesses = 0
    investment_dev = [None]
    for i in range(len(flat_predictions)-1):
#        if i not in check_lst:
        if flat_predictions[i+1]>flat_predictions[i] and dummy==0 :
            investment = investment - fee
            investment = investment*(compare_test[i+1]/compare_test[i])
            dummy = 1
            bs = bs+1
#            if compare_test[i+1]>compare_test[i]:
#                correct = correct+1
        elif flat_predictions[i+1]>flat_predictions[i] and dummy==1:
            investment = investment*(compare_test[i+1]/compare_test[i])   
        elif flat_predictions[i+1]<flat_predictions[i] and dummy==1:
            investment = investment-fee
            dummy = 0
            bs = bs+1

        elif flat_predictions[i+1]<flat_predictions[i] and dummy==0:
            investment = investment
        investment_dev.append(investment)
        
    return investment, bs, investment_dev
def plot_investment(investment_dev, stock):
    matplotlib.style.use('seaborn-darkgrid')
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(investment_dev, label='Investment of '+stock)
    plt.title('Investment over time of ' +stock, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Investment', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    plt.savefig(plot_folder + stock + '_investment_development.png',dpi=400)
    plt.show()
    plt.close()

def check_stocks(stock, df, prediction_date, y_test_correction):
    weekmask = 'Mon Tue Wed Thu Fri'
    holidays = [datetime(2018, 3, 30), datetime(2018, 5, 28), datetime(2018, 7, 4), datetime(2018, 5, 28),
                datetime(2018, 7, 4), datetime(2018, 9, 3), datetime(2018, 11, 22), datetime(2018, 12, 25)]
    bday_cust = CustomBusinessDay(holidays=holidays, weekmask=weekmask) 
    diff = np.busday_count(datetime(df.loc[len(df)-1, 'date'].year, df.loc[len(df)-1, 'date'].month, df.loc[len(df)-1, 'date'].day), datetime(int(prediction_date[:4]),int(prediction_date[5:7]) , int(prediction_date[8:])),
                        weekmask=bday_cust.weekmask, 
                        holidays=bday_cust.holidays)
    df_prediction = pd.read_csv('2018-03-14'+'predictions_fluc.csv')
    check = df_prediction[stock]
    data_prediction = check.tolist()  
    data_current = y_test_correction[-10:]
    padding = [None for p in list(range(len(data_current)-diff+1))]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)#, figsize=figsize)
    ax.plot(data_current, label='True Data')
    ax.plot(padding + data_prediction, label='True Data')
    plt.title('Compare ' + stock, size=16)
    plt.xlabel('Days', size=13)
    plt.ylabel('Stock Price', size=13)
    plt.figtext(0.5, 0.01, 'date created: ' + now, 
                horizontalalignment='center', size=10)
    plt.savefig(plot_folder + stock + '_compare.png',dpi=400)
    plt.show()
    plt.close()
#give loss/profit
#    do it to real data not y_test_correction

def distribution(data):
    d = []
    for i in range(len(data)-1):
        a = data[i+1]/data[i]
        d.append(round(a-1,3))
    #    boundary = max()
    counts, bins = np.histogram(d, bins=list(np.arange(-0.2,0.22,0.02)))
    decrements = list(reversed(abs(np.asarray(np.sqrt(counts[:int(len(counts)/2.0)])))))
    increments = list(np.sqrt(counts[int((len(counts)/2.0)):]))
    tags_list = []
    for i in list(np.arange(0.02,0.22,0.02)):
        tags_list.append(str(round(i,2)))
    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(tags_list, increments, align='center', color='g')
    axes[0].set(title='Increments (squared)')
    axes[1].barh(tags_list, decrements,align='center', color='r')
    axes[1].set(title='decrements (squared)')
    axes[0].invert_xaxis()
    axes[0].set( yticklabels=tags_list)
    axes[0].yaxis.tick_right()
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.28)
    extremes = len(d)-sum(counts)
    if abs(max(d))>0.2:
        axes[0].text(37.0,8.,str(extremes)+' times more than 20%', fontsize=15, zorder=12)
    
    plt.show()
    

      
    

    
    
        
        
    
#%%
    

    
    
    
    

    
    
    
    

    
    
    
