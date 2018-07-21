import os

from pynt.yahoo_reader import FinanceData
from pynt.lstm_model import LstmModel


class Utils():
    @staticmethod
    def curr_path():
        """Return path this file resides in."""
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def init_financedata(start_date='2001-01-01', end_date='2001-02-01',
                         tickers=['AAPL'], data_dir='./data/'):
        return FinanceData(start_date=start_date, end_date=end_date,
                           tickers=tickers, data_dir=data_dir)

    @staticmethod
    def init_lstmmodel():
        return LstmModel()
