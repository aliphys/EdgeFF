#!/usr/bin/env python3
"""Regression test for basicRun evaluation light validation."""

import unittest
import torch
import numpy as np
from basicRun.evaluation import eval_val_set_light


class DummyLightModel:
    def light_predict_one_sample(self, x, confidence_mean_vec, confidence_std_vec):
        # Return a tensor prediction and an integer layer count.
        return torch.tensor([1], dtype=torch.int64), 2


class TestEvalValSetLight(unittest.TestCase):
    def test_eval_val_set_light_handles_int_layers_used(self):
        model = DummyLightModel()
        inputs = torch.randn(1, 784)
        targets = torch.tensor([1], dtype=torch.int64)
        confidence_mean_vec = np.zeros(1)
        confidence_std_vec = np.zeros(1)

        # The function should complete without raising an AttributeError.
        eval_val_set_light(model, inputs, targets, confidence_mean_vec, confidence_std_vec)


if __name__ == '__main__':
    unittest.main()
