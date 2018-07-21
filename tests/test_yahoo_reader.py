import pytest

from tests.test_utils import Utils as U


def test_params():
    fd = U.init_financedata()

    assert type(fd.tickers) == list


# def test_data_download():
#     fd = U.init_financedata()

    # assert len(fd.df) > 0


# def test_none_params():
#     with pytest.raises(ValueError, match=r'.*keyword param has not been specified.*'):
#         fd = U.init_financedata(start_date=None)
#
#
# def test_non_existing_ticker():
#    with pytest.raises(ValueError, match=r'.*no data downloaded.*'):
#        fd = U.init_financedata(tickers=['dickhead'])