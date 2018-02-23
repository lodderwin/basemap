#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:00:12 2018

@author: erwinlodder
"""
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#import remaining necessary modules
import lstm, time   #back up functions
import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from datetime import datetime

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
    result = np.array(result)
    #here 90% of the data is chosen to train on, the remaining 10% will be used to test
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
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
def predict_test(days_ahead, x_test, seq_len, model):
    predicted_test = []
    days_ahead = 5
    predictions_in_function = int(x_test.shape[0]/days_ahead)
    remaining_predictions = x_test.shape[0]%days_ahead
    for i in range(predictions_in_function):
    #current_frame is x values used to predict y
        current_frame = x_test[i*5]
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
def predict_current(seq_len,days_ahead, x_test, model):
    predicted = []
    current_frame = x_test
    for j in range(days_ahead):
        predicted.append((model.predict(current_frame[newaxis,:,:])[0,0]))
        current_frame = current_frame[1:]
        current_frame = np.insert(current_frame, [seq_len-1], predicted[-1], axis=0)
    return predicted
def plot_current(x_test,predicted,stock):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(x_test, label='True Data', color='b')
    #padding is used to set the new predictions to an appropiate distance from 0 days
    padding = [None for p in list(range(len(x_test)))]
    if (predicted[-1]-predicted[0])>0.0:
        plt.plot(padding+predicted, label='Prediction', alpha=0.6, color='g')
    elif (predicted[-1]-predicted[0])<0.0:
        plt.plot(padding+predicted, label='Prediction', alpha=0.6, color='r')
        
    plt.xlabel('days')
    plt.savefig(stock+'_current_prediction.png',dpi=400)
    plt.show()
    
        
def predict_corrected(predicted_test,y_test_correction):
    corrected_predicted_test = []
    for i in range(len(predicted_test)):
        corrected_predicted = []
        for j in range(len(predicted_test[0])):
            temp_pred = [x+1 for x in predicted_test[i][:j+1]]
            multiply = multiply_lst(temp_pred)
            corrected_predicted.append(multiply*y_test_correction[i*seq_len+j])
        corrected_predicted_test.append(corrected_predicted)
    return corrected_predicted_test
def correct_predict_test(seq_len, predicted_test, y_test_correction):
    prediction_len=5
    corrected_predicted_test = []
    for i in range(len(predicted_test)):
        corrected_predicted = []
    ##include first point
        for j in range(len(predicted_test[0])):
            temp_pred = [x+1 for x in predicted_test[i][:j+1]]
            multiply = multiply_lst(temp_pred)
    #        if j<1.0:
            corrected_predicted.append(multiply*y_test_correction[i*seq_len-5])
    #        else :
    #            corrected_predicted.append(multiply*y_test_correction[i*seq_len])
        corrected_predicted_test.append(corrected_predicted)   
    return corrected_predicted_test
def plot_results(y_test_correction, corrected_predicted_test, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(y_test_correction, label='True Data')
    #padding is used to set the new predictions to an appropiate distance from 0 days
    for i, data in enumerate(corrected_predicted_test):
            padding = [None for p in list(range(int(((i) * prediction_len)/1.0)))]
            plt.plot(padding+data, label='Prediction', alpha=0.6)

    plt.xlabel('days')
    plt.savefig('newpredictionamag.png',dpi=400)
    plt.show()
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
    for i in range(len(flat_predictions)-1):
        if i%5 != 0:
            if flat_predictions[i+1]>flat_predictions[i] and dummy==0 :
                investment = investment - fee
                investment = investment*(compare_test[i+1]/compare_test[i])
                dummy = 1
                bs = bs+1
            elif flat_predictions[i+1]>flat_predictions[i] and dummy==1:
                investment = investment*(compare_test[i+1]/compare_test[i])   
            elif flat_predictions[i+1]<flat_predictions[i] and dummy==1:
                investment = investment-fee
                dummy = 0
                bs = bs+1
    
            elif flat_predictions[i+1]<flat_predictions[i] and dummy==0:
                investment = investment
    return investment, bs