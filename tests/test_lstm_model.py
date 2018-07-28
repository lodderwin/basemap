import pytest
import numpy as np

from pynt.yahoo_reader import FinanceData
from pynt.lstm_model import LstmModel


# def test_window_length():
#     fd = FinanceData(start_date='2001-01-01', end_date='2001-02-01',
#                      tickers=['AAPL'], data_dir='./data/')
#
#     lstm = LstmModel(df=fd.df_processed, ticker='AAPL')
#
#     assert type(lstm.window_length) == int

# def test_nmd_array_dict():
#     fd = FinanceData(start_date='2001-01-01', end_date='2001-02-01',
#                      tickers=['AAPL'], data_dir='./data/')
#
#     lstm = LstmModel(df=fd.df_processed, ticker='AAPL')
#
#     assert type(lstm.nmd_array_dict) == dict
#     for array in lstm.nmd_array_dict:
#         type(lstm.nmd_array_dict[array]) == np.array
#         len(lstm.nmd_array_dict[array]) > 0


def test_train_test_split():
    fd = FinanceData(start_date='2001-01-01', end_date='2001-02-01',
                         tickers=['AAPL'], data_dir='./data/')

    lstm = LstmModel(df=fd.df_processed, ticker='AAPL')
    lstm.train_test_split()

    # print(lstm.X_train)
    assert len(lstm.X_train) > 0
    assert len(lstm.y_train) > 0
