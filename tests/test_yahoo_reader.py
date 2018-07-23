import pytest
import numpy as np

from pynt.yahoo_reader import FinanceData


def test_params():
    fd = FinanceData(start_date='2001-01-01', end_date='2001-02-01',
                     tickers=['AAPL'], data_dir='./data/')

    assert type(fd.tickers) == list


def test_data_download():
    fd = FinanceData(start_date='2001-01-01', end_date='2001-02-01',
                     tickers=['AAPL'], data_dir='./data/')

    assert len(fd.df) > 0

def test_multiple_tickers():
    fd = FinanceData(start_date='2001-01-01', end_date='2001-02-01',
                     tickers=['AAPL','AMAG'], data_dir='./data/')

    print(fd.df.ticker.unique())


def test_non_existing_ticker():
    with pytest.raises(ValueError, match=r'.*no data downloaded.*'):
        FinanceData(start_date='2001-01-01', end_date='2001-02-01',
                    tickers=['dickhead'], data_dir='./data/')


def test_columns():
    fd = FinanceData(start_date='2001-01-01', end_date='2001-02-01',
                     tickers=['AAPL'], data_dir='./data/')

    cols = {
        'date': np.datetime64,
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'adjclose': np.float64,
        'volume': np.int64,
        'ticker': object,
        'day_number': np.int64,
        'window': np.int64,
        'normaliser': np.float64,
        'normaliserv': np.int64,
        'close_nmd': np.float64,
        'volume_nmd': np.float64,
        'normaliserh': np.float64,
        'normaliserl': np.float64,
        'high_nmd': np.float64,
        'low_nmd': np.float64,
        'high_nmd_close': np.float64,
        'low_nmd_close': np.float64,
        'open_nmd_close': np.float64
        }

    for col in cols.keys():
        assert col in fd.df_processed.columns
        assert cols[col] == fd.df_processed[col].dtype
