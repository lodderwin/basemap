from datetime import datetime
import time
import statistics
import random
import logging
import os

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout


class LstmModel:
    def __init__(self, df: pd.DataFrame, ticker: str):
        self.df = df
        self.ticker = ticker

        self.short_term_models_dir = './data/short_term_models/'
        if not os.path.exists(self.short_term_models_dir):
            os.mkdir(self.self.short_term_models_dir)

        self.window_length = 16
        self.initial_investment = 100
        self.iterations = 20
        self.mcr = 0.00000001
        self.best_investment_dev = 100
        self.new_test = -1000000
        self.validation_split = 0.1
        self.margins = list(np.linspace(1.0,1.1,40))
        self.days_ahead = 1
        self.investment = 300.0
        self.fee_per_stock = 0.005
        self.fee = 0.60
        self.margins = list(np.arange(1.0,1.101,0.001))

        nmd_array_fields = [
            'close_nmd',
            'volume_nmd',
            'high_nmd_close',
            'low_nmd_close',
            'open_nmd_close',
            'day_number',
            'date',
            'window'
        ]

        nmd_array_dict = {}
        for field in nmd_array_fields:
            nmd_array_dict[field] = self.series_to_ndarray(column=field)

        self.nmd_array_dict = nmd_array_dict

        self.model_input = np.concatenate((
            nmd_array_dict['close_nmd'],
            nmd_array_dict['open_nmd_close'],
            nmd_array_dict['low_nmd_close'],
            nmd_array_dict['high_nmd_close'],
            nmd_array_dict['volume_nmd'],
            nmd_array_dict['day_number'],
            ), axis=2
        )

        self.input_dim = self.model_input.shape[2]

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_days_sim_non_shuffle = None
        self.train_windows_non_shuffle = None
        self.test_days = None
        self.test_windows = None
        self.X_train_sim = None
        self.investment = None
        self.investment_dev = None
        self.investment_dev_df = None
        self.increase_correct = None
        self.increase_false = None
        self.mean_test = None
        self.std_test = None
        self.len_points = None


    def series_to_ndarray(self, column: str):
        """Returns numpy array of shape (1, length of window, 1)
        using pd.DataFrame as input
        """
        # Create empty list of arrays
        arrs_list = []

        for window in self.df.window.unique():
            # Create array and reshape
            arr = self.df[self.df.window == window][column].values

            # Reshape: (, number of days per array, number of columns)
            arr = arr.reshape(1, len(self.df[self.df.window == window][column]), 1)
            # append arr to arrs_list
            arrs_list.append(arr)

        # Use numpy vstack to create array of arrays
        arr = np.vstack(arrs_list)

        return arr

    def train_test_split(self, ratio=0.95):
        """Takes multi-dimensional array as input and returns arrays for:
        x_train, y_train, x_test and y_test. x_test and y_test are shuffled using a
        fixed random state.
        """
        # Create copy of np array to avoid shuffling ary
        ary = np.copy(self.model_input)
        # Define where to split arr based on length
        split_row = int(ary.shape[0] * ratio)

        # Take first five days of each window as x_train
        X = ary[:, :-1, :]
        # Split X into train and test and shuffle train
        X_train, X_test = np.split(X, [split_row])
        X_train_sim = X_train
        np.random.RandomState(1).shuffle(X_train)

        # Take last day of each window as y
        y = ary[:, -1, :]
        # Split X into train and test and shuffle y_train
        y_train, y_test = np.split(y, [split_row])
        np.random.RandomState(1).shuffle(y_train)

        train_days, test_days = np.split(self.nmd_array_dict['date'], [split_row])
        train_days_sim_non_shuffle = train_days
        np.random.RandomState(1).shuffle(train_days)
        train_windows_non_shuffle, test_windows = np.split(self.nmd_array_dict['window'], [split_row])

        if self.input_dim > 1.0:
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_days_sim_non_shuffle = train_days_sim_non_shuffle
        self.train_windows_non_shuffle = train_windows_non_shuffle
        self.test_days = test_days
        self.test_windows = test_windows
        self.X_train_sim = X_train_sim


    def build_model(self, params: dict):
        """
        """
        if len(params.keys()) == 5:
            model = Sequential()
            model.add(LSTM(
                input_dim=params['input_dim'],
                output_dim=params['node1'],
                return_sequences=True
            ))
            model.add(Dropout(0.2))
            model.add(LSTM(params['node2'], return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(output_dim=params['output_dim']))
            model.add(Activation("linear"))
            model.compile(loss="mse", optimizer="rmsprop", metrics=['mse'])

        if len(params.keys()) == 6:
            model = Sequential()
            model.add(LSTM(
                input_dim=params['input_dim'],
                output_dim=params['node1'],
                return_sequences=True
            ))
            model.add(Dropout(0.2))
            model.add(LSTM(params['node2'], return_sequences=True))
            model.add(LSTM(params['node3'], return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(output_dim=params['output_dim']))
            model.add(Activation("linear"))
            model.compile(loss="mse", optimizer="rmsprop", metrics=['mse'])

        return model


    def randomised_model_config(self):
        for iteration in range(0, self.iterations):
            logging.info('iteration: {} of {}'.format(iteration + 1, self.iterations))

            self.params = {
                'input_dim': self.input_dim,
                'node1': np.random.randint(15,130),
                'node2': np.random.randint(15,130),
                'output_dim': 1,
                'batch_size': random.choice(np.asarray([8,16,32]))
            }

            model = self.build_model(self.params)

            # Fit using x and y test and validate on 10% of those arrays
            self.epochs = random.choice(np.asarray([3,5,8]))
            model.fit(self.x_train, self.y_train, validation_split = 0.1, batch_size = self.params['batch_size'],
                      epochs = self.epochs)

            self.params['epochs'] = self.epochs
            time.sleep(5.1)

            df_predict = self.predict_test(model)

            for margin in self.margins:
                self.invest_sim(df_predict, margin)

                if  (( 1 + (self.mean_test - self.std_test)) ** self.len_points) > self.new_test
                    and self.investment > 300.0 and self.len_points > 10:
                    self.new_test = ((1 + (self.mean_test - self.std_test)) ** self.len_points)
                    self.mcr = (self.investment / 300.0) * (self.df['close'].tolist()[0] / self.df['close'].tolist()[-1])
                    self.beginparams = self.params
                    self.initial_investment = self.investment
                    self.best_margin = margin
                    self.best_investment_dev = self.investment_dev_df
                    Plotting.plot_investment(investment_dev,ticker,params,margin, window_length)

                    model.save('{}{}_{}.h5'
                        .format(self.short_term_models_dir, self.ticker, self.window_length),
                        overwrite=True
                    )

                elif (self.increase_correct + self.increase_false) == 0.0:
                    continue

        del model


    def predict_test(self, model):
        df_predict = pd.DataFrame([])
        predictions_in_function = int(self.X_test.shape[0]/self.days_ahead)
        #take out unique values in df_p for window--> normalizer
        df_p_window_index = self.df.set_index(self.df['window'])

        # Custom calendar
        weekmask = 'Mon Tue Wed Thu Fri'
        holidays = [datetime(2016, 3, 30), datetime(2016, 5, 28), datetime(2016, 7, 4), datetime(2016, 5, 28),
                    datetime(2016, 7, 4), datetime(2016, 9, 3), datetime(2016, 11, 22), datetime(2016, 12, 25),
                    datetime(2017, 3, 30), datetime(2017, 5, 28), datetime(2017, 7, 4), datetime(2017, 5, 28),
                    datetime(2017, 7, 4), datetime(2017, 9, 3), datetime(2017, 11, 22), datetime(2017, 12, 25),
                    datetime(2018, 3, 30), datetime(2018, 5, 28), datetime(2018, 7, 4), datetime(2018, 5, 28),
                    datetime(2018, 7, 4), datetime(2018, 9, 3), datetime(2018, 11, 22), datetime(2018, 12, 25)]
        bday_cust = CustomBusinessDay(holidays=holidays, weekmask=weekmask)

        for i in range(predictions_in_function):
            current_frame = self.X_test[i * self.days_ahead]
            y_date = pd.to_datetime(self.test_days[i * self.days_ahead][-1])[0]
            window = self.test_windows[i * self.days_ahead][0][0]
            normaliser = df_p_window_index.loc[window, 'normaliser']
            predicted = []
            for j in range(self.days_ahead):
            #4 days predicted ahead with predicted values!
            #model.predict only accepts 3 dimension numpy matrices therefore newaxis
                predicted.append((model.predict(current_frame[np.newaxis,:,:])[0,0]))
                predicted_value = predicted[-1]
                df_predict = df_predict.append(
                        {
                        'date': pd.to_datetime(y_date + bday_cust * j),
                        'y_predict': (predicted_value + 1) * normaliser.unique()[0]
                    }, ignore_index=True
                )
                current_frame = current_frame[1:]
                current_frame = np.insert(current_frame, [len(self.X_test[0])-1], predicted[-1], axis=0)

        return df_predict

    def invest_sim(self, df_predict, margin):
        df_index = self.df.set_index(self.df['date'])
        df_predict = df_predict.set_index(df_predict['date'])
        df_merge = pd.merge(
            df_index,
            df_predict,
            how='left',
            left_on = [
                np.asarray(df_index.index.year).astype(np.str),
                np.asarray(df_index.index.month).astype(np.str),
                np.asarray(df_index.index.day).astype(np.str)
            ],
            right_on = [
                np.asarray(df_predict.index.year).astype(np.str),
                np.asarray(df_predict.index.month).astype(np.str),
                np.asarray(df_predict.index.day).astype(np.str)
            ]
        )
        df_merge = (df_merge
            .dropna(subset=['y_predict'])
            .dropna(subset=['close'])
            .reset_index(drop=True)
        )

        dummy = 0
        investment_dev = []
        end_index = len(df_merge)-1
        increase_correct = 0
        increase_false = 0
        decrease_correct = 0
        decrease_false = 0
        distribution = []
        dct_df = {
            'dates': [df_merge.loc[0,'date_x']],
            'y_predict_{}'.format(self.ticker): [df_merge.loc[0,'close']],
            'close_real_{}'.format(self.ticker): [df_merge.loc[0,'close']]
        }

        for index, row in df_merge.iterrows():
            if index<end_index:
                dct_df['dates'].append(df_merge.loc[index+1, 'date_x'])
                dct_df['y_predict_{}'.format(self.ticker)].append(df_merge.loc[index+1, 'y_predict'])
                dct_df['close_real_{}'.format(self.ticker)].append(df_merge.loc[index+1, 'close'])

                if (df_merge.loc[index + 1, 'y_predict'] / df_merge.loc[index, 'close']) > margin and dummy == 0:
                    investment = self.investment - (int(self.investment / df_merge.loc[index, 'close'])
                                               * self.fee_per_stock + self.fee)
                    distribution.append((df_merge.loc[index+1, 'close'] / df_merge.loc[index, 'close']) - 1)
                    investment = investment * (df_merge.loc[index+1, 'close'] / df_merge.loc[index, 'close'])
                    dummy = 1
                    if (df_merge.loc[index+1, 'close'] / df_merge.loc[index, 'close']) > 1.0:
                        increase_correct = increase_correct + 1
                    elif (df_merge.loc[index + 1, 'close'] / df_merge.loc[index, 'close']) < 1.0:
                        increase_false = increase_false + 1
                elif df_merge.loc[index+1, 'y_predict'] > df_merge.loc[index, 'close'] > margin and dummy == 1:
                    distribution.append((df_merge.loc[index + 1, 'close'] / df_merge.loc[index, 'close']) -1)
                    investment = investment * (df_merge.loc[index + 1, 'close'] / df_merge.loc[index, 'close'])
                    if (df_merge.loc[index + 1, 'close'] / df_merge.loc[index, 'close']) > 1.0:
                        increase_correct = increase_correct + 1
                    elif (df_merge.loc[index + 1, 'close'] / df_merge.loc[index, 'close']) < 1.0:
                        increase_false = increase_false + 1
                elif (df_merge.loc[index + 1, 'y_predict'] < df_merge.loc[index, 'close']) < margin and dummy == 1:
                    investment = investment - (int(investment / df_merge.loc[index, 'close']) * self.fee_per_stock + self.fee)
                    dummy = 0
                    if (df_merge.loc[index + 1, 'close'] / df_merge.loc[index, 'close']) < 1.0:
                        decrease_correct = decrease_correct + 1
                    elif (df_merge.loc[index+1, 'close'] / df_merge.loc[index, 'close']) > 1.0:
                        decrease_false = decrease_false + 1
                elif (df_merge.loc[index + 1, 'y_predict'] < df_merge.loc[index, 'close']) < margin and dummy == 0:
                    investment = investment
                    if (df_merge.loc[index + 1, 'close'] / df_merge.loc[index, 'close']) < 1.0:
                        decrease_correct = decrease_correct + 1
                    elif (df_merge.loc[index+1, 'close'] / df_merge.loc[index, 'close']) > 1.0:
                        decrease_false = decrease_false + 1
            investment_dev.append(investment)

        investment_dev_df = pd.DataFrame(dct_df)

        if len(distribution) > 5:
            self.investment = investment
            self.investment_dev = investment_dev
            self.investment_dev_df = investment_dev_df
            self.increase_correct = increase_correct
            self.increase_false = increase_false
            self.mean_test = statistics.mean(distribution)
            self.std_test = statistics.stdev(distribution)
            self.len_points = len(distribution)
        else:
            self.investment = investment
            self.investment_dev = investment_dev
            self.investment_dev_df = investment_dev_df
            self.increase_correct = increase_correct
            self.increase_false = increase_false
            self.mean_test = -100
            self.std_test = 100000
            self.len_points = len(distribution)


    #
    # def randomised_model_config_days_average(test_windows,df_p,test_days,input_dim ,window_length,ticker,df,days_average,x_train, y_train, x_test, y_test,
    #                             initial_investment=100., iterations=20, epochs=10):
    #     for iteration in range(0, iterations):
    #         print('iteration: {} of {}'.format(iteration + 1, iterations))
    #         # Define params randomly
    #         params = {'input_dim':2,
    #                   'node1':np.random.randint(15,80),
    #                   'node2':np.random.randint(15,80),
    #                   'output_dim':1,
    #                   'batch_size':random.choice(np.asarray([16,32]))}
    #
    #         model = build_model(params)
    #
    #         # Fit using x and y test and validate on 10% of those arrays
    #         model.fit(x_train,
    #                   y_train,
    #                   validation_split = 0.1,
    #                   batch_size = params['batch_size'],
    #                   epochs = random.choice(np.asarray([6,12,18])))
    #
    #         df_predict = predict_test_days_average(test_windows,test_days, df_p,x_test,window_length, days_average, model, df)
    #         print(df_predict)
    #
    #
    #         for margin in margins:
    #             investment, investment_dev,investment_dev_df = invest_sim(df_predict,df,margin,ticker)
    #
    #             if initial_investment < investment:
    #                 initial_investment = investment
    #                 best_investment_dev = investment_dev_df
    #                 print(investment)
    #                 Plotting.plot_investment(investment_dev,ticker,params,margin)
    #     #            Plotting.plot_results(real_prices,corrected_predicted_test, days_ahead, ticker)
    #                 model.save(long_term_folder+ticker+'_'+str(window_length)+'_'+str(days_average)+'_model.h5', overwrite=True)
    #     del model
    #     print('Loading model')
    #     best_model = load_model(long_term_folder+ticker+'_'+str(window_length)+'_'+str(days_average)+'_model.h5')
    #
    #     return best_model, investment, best_investment_dev
    #
    #
    # def predict(model, X, days_ahead):
    #     """X being x_train or x_test
    #     """
    #     # How many days should the model predict ahead for?
    #     days_ahead = days_ahead
    #
    #     predictions = []
    #
    #     for idx, window in enumerate(X):
    #         print('\rpredicting window {} of {}'.format(idx + 1, X.shape[0]),
    #               end='\r', flush=False)
    #         # Reshape window as lstm predict needs np array of shape (1,5,1)
    #         window = np.reshape(window, (1, window.shape[0], window.shape[1]))
    #         for day in range(days_ahead):
    #             # Predict & extract prediction from resulting array
    #             prediction = model.predict(window)[0][0]
    #             # Reshape prediction so it can be appened to window
    #             prediction = np.reshape([prediction], (1, 1, 1))
    #             # Add new value to window and remove first value
    #             window = np.append(window, prediction)[1:]
    #             # Above appendage flattens np array so needs to be rehaped
    #             window = np.reshape(window, (1, window.shape[0], 1))
    #
    #         # Result of inner loop is a window of five days no longer containing any original days
    #         predictions.append(window)
    #
    #     # predictions is a list, create np array with shape matching X
    #     predictions = np.reshape(predictions, X.shape)
    #
    #     return predictions
    #
    # def predict_single(model, X, days_ahead):
    #     days_ahead = days_ahead
    #     prediction = []
    #     current_frame = X
    #     for day in range(days_ahead):
    #         prediction.append((model.predict(current_frame[newaxis,:,:])[0,0]))
    #     return prediction
    #
    #
    # def predict_test_days_average(test_windows,test_days, df_p,x_test,window_length, days_average, model, df):
    #     df_predict = pd.DataFrame([])
    # #    y_date_first = pd.to_datetime(test_days[0][-1])[0]
    #     predictions_in_function = int(x_test.shape[0]/days_average)
    #     #take out unique values in df_p for window--> normalizer
    #     df_p_window_index = df_p.set_index(df_p['window'])
    #
    #
    #     ###custom calendar
    #     weekmask = 'Mon Tue Wed Thu Fri'
    #     holidays = [datetime(2017, 3, 30), datetime(2017, 5, 28), datetime(2017, 7, 4), datetime(2017, 5, 28),
    #                 datetime(2017, 7, 4), datetime(2017, 9, 3), datetime(2017, 11, 22), datetime(2017, 12, 25),
    #                 datetime(2018, 3, 30), datetime(2018, 5, 28), datetime(2018, 7, 4), datetime(2018, 5, 28),
    #                 datetime(2018, 7, 4), datetime(2018, 9, 3), datetime(2018, 11, 22), datetime(2018, 12, 25)]
    #     bday_cust = CustomBusinessDay(holidays=holidays, weekmask=weekmask)
    #     ###
    #     for i in range(predictions_in_function):
    #         current_frame = x_test[i*days_average]
    #         y_date = pd.to_datetime(test_days[i*days_average][-1])[0]
    #         window = test_windows[i*days_average][0][0]
    #         normaliser = df_p_window_index.loc[window,'normaliser']
    #         predicted = []
    #         for j in range(5):
    #     #4 days predicted ahead with predicted values!
    #     #model.predict only accepts 3 dimension numpy matrices therefore newaxis
    #             predicted.append( (model.predict(current_frame[newaxis,:,:])[0,0]))
    #             predicted_value = predicted[-1]
    #             df_predict = df_predict.append({'date': pd.to_datetime(y_date+bday_cust*j), 'y_predict': (predicted_value+1)*normaliser.unique()[0] },ignore_index=True)
    #     return df_predict
    #
    #
    # def invest_sim_days_average(df_predict, df,margin,ticker):
    #     df_index = df.set_index(df['date'])
    #     df_predict = df_predict.set_index(df_predict['date'])
    #     df_merge = pd.merge(df_index, df_predict, how='left', left_on=[np.asarray(df_index.index.year).astype(np.str), np.asarray(df_index.index.month).astype(np.str),np.asarray(df_index.index.day).astype(np.str)],
    #                         right_on=[np.asarray(df_predict.index.year).astype(np.str),np.asarray(df_predict.index.month).astype(np.str),np.asarray(df_predict.index.day).astype(np.str)])
    #     df_merge = df_merge.dropna(subset=['y_predict'])
    #     df_merge = df_merge.dropna(subset=['close'])
    #     df_merge = df_merge.reset_index(drop=True)
    #     investment = 300.0
    #     fee_per_stock = 0.005
    #     fee = 0.60
    #     dummy = 0
    #     investment_dev = []
    #     end_index = len(df_merge)-1
    #     dct_df = {}
    #
    #     dct_df['dates'] = [df_merge.loc[0,'date_x']]
    #     dct_df['y_predict_'+ticker] = [df_merge.loc[0,'close']]
    #     dct_df['close_real_'+ticker] = [df_merge.loc[0,'close']]
    #     for index, row in df_merge.iterrows():
    #         if index<end_index:
    #             dct_df['dates'].append(df_merge.loc[index+1, 'date_x'])
    #             dct_df['y_predict_'+ticker].append(df_merge.loc[index+1, 'y_predict'])
    #             dct_df['close_real_'+ticker].append(df_merge.loc[index+1, 'close'])
    #
    #             if (df_merge.loc[index+1, 'y_predict']/df_merge.loc[index, 'close'])>margin and dummy==0 :
    #                 investment = investment - (int(investment/df_merge.loc[index, 'close'])*fee_per_stock+fee)
    #                 investment = investment*(df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])
    #                 dummy = 1
    #             elif df_merge.loc[index+1, 'y_predict']>df_merge.loc[index, 'close']>margin and dummy==1:
    #                 investment = investment*(df_merge.loc[index+1, 'close']/df_merge.loc[index, 'close'])
    #             elif (df_merge.loc[index+1, 'y_predict']<df_merge.loc[index, 'close'])<margin and dummy==1:
    #                 investment = investment-(int(investment/df_merge.loc[index, 'close'])*fee_per_stock+fee)
    #                 dummy = 0
    #             elif (df_merge.loc[index+1, 'y_predict']<df_merge.loc[index, 'close'])<margin and dummy==0:
    #                 investment = investment
    #                 #### dict to datafram, output dataframe!
    #         investment_dev.append(investment)
    #
    #     investment_dev_df = pd.DataFrame(dct_df)
    #     return investment, investment_dev, investment_dev_df
    #
    # def fluc_check(graph):
    #     total_log = 0
    #     for i in range(len(graph)-1):
    #         total_log = (np.log10(graph[i+1])-np.log10(graph[i]))**2 + total_log
    #     return total_log
    #
    #
        