import pytest
import numpy as np

from pynt.lstm_utils import LstmUtils
from tests.test_utils import Utils as U


def test_load_user():
    assert type(LstmUtils.load_user_from_yml(yml_file='./data/configs/user_settings.yml')) == str
