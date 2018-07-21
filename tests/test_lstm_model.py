import pytest

from pynt.lstm_model import LstmModel
from pynt.preprocessing import PreProcessing as pp
from tests.test_utils import Utils as U


def test_lstm_model():
    fd = U.init_financedata()
    fd.get_fix_yahoo_data()
    df = pp.pre_process_data(fd.df)

    # for col in df.columns:
    #     print(col)
    # print(df.head())

    LstmModel(df=df, ticker='AAPL')