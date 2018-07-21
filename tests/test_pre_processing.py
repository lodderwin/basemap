import pytest
import numpy as np

from pynt.preprocessing import PreProcessing as pp
from tests.test_utils import Utils as U


def test_columns():
    fd = U.init_financedata()
    fd.get_fix_yahoo_data(store=False)
    fd.df = pp.pre_process_data(fd.df)

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
        assert col in fd.df.columns
