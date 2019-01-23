# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
import pytest

from gpflow.base import Parameter

rng = np.random.RandomState(42)

correct_shape = (10, 5)
incorrect_shape = (20, 4)

@pytest.fixture
def parameter_value():
    return rng.randn(*correct_shape)

def test_initialization(parameter_value):
    control_variate = rng.randn(*incorrect_shape)
    with pytest.raises(AssertionError):
        Parameter(parameter_value, control_variate=control_variate)
    with pytest.raises(AssertionError):
        param = Parameter(parameter_value)
        param.control_variate = control_variate

def test_setter(parameter_value):
    control_variate = rng.randn(*correct_shape)
    param = Parameter(parameter_value)
    param.control_variate = control_variate
    assert param.control_variate.shape == correct_shape
