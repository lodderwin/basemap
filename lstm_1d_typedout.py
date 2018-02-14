#import keras modules
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
filename = 'understandBA.csv'
seq_len=5
normalise_window = True
def multiply_lst(lst):
    a = 1
    for i in range(len(lst)):
        a = lst[i] * a
    return a
#%% call data, closing price of days will be used to train and forecast
df = pd.read_csv(filename)
df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, fmt))
data = df['Close'].tolist()
dates = df['Date'].tolist()
#%%
sequence_length = seq_len + 1 #the +1 is needed to select 'test day' which will become y
high = []
result = []
#to recognise patterns in data, the data has to be normalised and divided up in to smaller training sets
normalise_window = True
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
if normalise_window:
    result = lstm.normalise_windows(result)
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
y_test_correction = lstm.normalise_data(data[int(row):])
#for future plots, exact dates might come in handy
y_test_begin = data[row]
dates = dates[int(row):] 
#%%
#setting the actual model, don't completely understand everything yet
#here keras is used to use tensorflow
#notes: best for bidu output_dim = 10, lstm = 18, batch_size = 128, epoch = 2
#notes: best for boeing output_dim = 10, lstm = 16, batch_size = 128, epoch = 1
n = 5
model = Sequential()
model.add(LSTM(input_dim=1, output_dim=10,
               return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16,
               return_sequences=False))
model.add(Dropout(0.2))
#potential second lstm layer
#model.add(LSTM(10,
#               return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))
start = time.time()
model.compile(loss='mse', optimizer = 'rmsprop')
print('Compilation time: ', time.time() - start)
#machine learning part
model.fit(
        x_train,
        y_train,
        batch_size=128,
        nb_epoch=1,
        validation_split=0.05)
#%%
predicted = []
predicted_test = []
days_ahead = 5
predictions_in_function = int(x_test.shape[0]/seq_len)
remaining_predictions = x_test.shape[0]%5
for i in range(predictions_in_function):
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
#last_prediction = predicted
#denormalize again for a clear and understandable graph
corrected_predicted_test = []
for i in range(predictions_in_function):
    corrected_predicted = []
    for j in range(len(predicted_test[0])):
        temp_pred = [x+1 for x in predicted_test[i][:j+1]]
        multiply = multiply_lst(temp_pred)
        corrected_predicted.append(multiply*y_test_correction[i*seq_len+j])
    corrected_predicted_test.append(corrected_predicted)          
prediction_len=5
#plot predictions
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test_correction, label='True Data')
#padding is used to set the new predictions to an appropiate distance from 0 days
for i, data in enumerate(corrected_predicted_test):
        padding = [None for p in list(range(int(((i) * prediction_len)/1.0)))]
        plt.plot(padding+data, label='Prediction', alpha=0.6)
#dates = dts.date2num(dates)
#plt.plot_date(list(dates), y_test_correction)
plt.xlabel('days')
plt.savefig('understandBA.png',dpi=400)
plt.show()
#%%calculate mse which is quite large, however not corrected for the difference between closing day price day 1 and opening day price day 2.
flat_predictions = np.asanyarray([item for sublist in corrected_predicted_test for item in sublist])
compare_num_days = len(flat_predictions)
compare_test = np.asarray(y_test_correction)[:compare_num_days]
compare_test = compare_test
mse = np.mean((flat_predictions-compare_test)**2)
print(np.sqrt(mse))
#%% build investment simulator
investment = 1000
for i in range(len(flat_predictions)-1):
    if i%5 != 0:
        if flat_predictions[i+1]>flat_predictions[i]:
            investment = investment*(compare_test[i+1]/compare_test[i])

made_money_get_bitches  = investment - 1000
print(made_money_get_bitches)




