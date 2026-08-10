import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clumsygrad.random import rand, randn
from src.clumsygrad.tensor import TensorType


class TestRand:
    def test_shape_and_dtype(self):
        t = rand((2, 3))
        assert t.shape == (2, 3)
        assert t.data.dtype == np.float32

    def test_default_tensor_type_is_input(self):
        t = rand((3,))
        assert t._tensor_type == TensorType.INPUT
        assert t.requires_grad is False

    def test_respects_tensor_type(self):
        t = rand((3,), tensor_type=TensorType.PARAMETER)
        assert t._tensor_type == TensorType.PARAMETER
        assert t.requires_grad is True

    def test_values_in_unit_range(self):
        np.random.seed(0)
        t = rand((100,))
        assert np.all(t.data >= 0.0)
        assert np.all(t.data < 1.0)

    def test_deterministic_with_seed(self):
        np.random.seed(42)
        t1 = rand((5,))
        np.random.seed(42)
        t2 = rand((5,))
        np.testing.assert_array_equal(t1.data, t2.data)


class TestRandn:
    def test_shape_and_dtype(self):
        t = randn((4, 2))
        assert t.shape == (4, 2)
        assert t.data.dtype == np.float32

    def test_default_tensor_type_is_input(self):
        t = randn((3,))
        assert t._tensor_type == TensorType.INPUT
        assert t.requires_grad is False

    def test_respects_tensor_type(self):
        t = randn((3,), tensor_type=TensorType.PARAMETER)
        assert t._tensor_type == TensorType.PARAMETER
        assert t.requires_grad is True

    def test_deterministic_with_seed(self):
        np.random.seed(7)
        t1 = randn((5,))
        np.random.seed(7)
        t2 = randn((5,))
        np.testing.assert_array_equal(t1.data, t2.data)
