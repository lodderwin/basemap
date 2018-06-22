#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:18:19 2018

@author: erwinlodder
"""

import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
def plot_results(true_y_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_y_data)
    plt.show()


#def plot_results_multiple(predicted_data, true_data, prediction_len):
#    fig = plt.figure(facecolor='white')
#    ax = fig.add_subplot(111)
#    ax.plot(true_data, label='True Data')
#    #Pad the list of predictions to shift it in the graph to it's correct start
#    for i, data in enumerate(predicted_data):
#        padding = [None for p in list(range(int((i * prediction_len)/5.0)))]
#        plt.plot(padding+data, label='Prediction', alpha=0.6)
##        plt.legend()
#    plt.savefig('IAG 3days200=700',dpi=400)
#
#    plt.show()
def plot_results_multiple(predictions, predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in list(range(int(((i) * prediction_len)/1.0)))]
        plt.plot(padding+data, label='Prediction', alpha=0.6)
##        plt.legend()
#    matrix = np.zeros(((len(predictions)+4),prediction_len))
#    for k in range(len(predictions)):
#        for l in range(prediction_len):
#            matrix[k,l] = predictions[k][l]
##    matrix = matrix.transpose()
#    lst = np.true_divide(matrix.sum(1),(matrix!=0).sum(1))
#    plt.plot(lst)

    plt.savefig('LETSGOO',dpi=400)

            
            
    plt.show()
def plot_new(model,x_test,true_data,n,extra_days):
    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    curr_frame = x_test[-n]
    predicted = []
    zeros = np.zeros((1,len(x_test)))
    for day in range(extra_days):
        predict = model.predict(curr_frame[newaxis,:,:])[0,0]
        predicted.append(predict)
        curr_frame.append(predict)
        curr_frame = curr_frame[1:]
    predicted = np.array(predicted)
    prediction = np.concatenate((zeros,predicted),axis=1)
    

        

    
    
    
    
    
    
    
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')
#    normalised_data = normalise_data(data)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test,]



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
def build_best_model(X_test, y_test,X_train, y_train, input_dim,iter_output_dim,iter_nodes,output_dim, iter_batch_size):
    error = 100.0
    best_fit_model = 0.0
    for i in range(1,40,4):
        for j in range(1,40,4):
            for k in range(2,iter_batch_size, 1):
                model = build_model([input_dim,i,j,output_dim])
                model.fit(
                    X_train,
                    y_train,
                    batch_size=int(k),
                    nb_epoch=1,
                    validation_split=0.05)
                loss = model.evaluate(X_test,y_test)
                if loss<error :
                    best_fit_model = model
                    error = loss
                    print(i,j,k,loss)
                    
                
                
                
    
    return best_fit_model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(int(len(data))):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
#    data = 
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        for k in [0]:
            curr_frame = data[i*prediction_len+k]
            predicted = []
            for j in range(int(prediction_len)):
                predicted.append((model.predict(curr_frame[newaxis,:,:])[0,0]))
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
    m = 0
#    for l in range(int(len(data)-(i*prediction_len+k))):
#        curr_frame = data[i*prediction_len+k+m]
#        m=m+1
#        predicted = []
#        for j in range(int(prediction_len)):
#            predicted.append((model.predict(curr_frame[newaxis,:,:])[0,0]))
#            curr_frame = curr_frame[1:]
#            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
#        prediction_seqs.append(predicted)
    return prediction_seqs