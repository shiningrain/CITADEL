from unittest import TestCase

from tensorflow import constant, float32, reduce_mean
from tensorflow import ragged
from tensorflow.python.keras.metrics import MeanAbsoluteError


class TestEmptyTensorMetrics(TestCase):
    def test_no_data(self):
        metric = MeanAbsoluteError()

        result = metric.result()

        expected = constant(0.0)
        self.assertEqual(expected, result)

    def test_empty_array(self):
        metric = MeanAbsoluteError()
        y_true = constant([])
        y_pred = constant([])

        metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = constant(0.0)
        self.assertEqual(expected, result)

    def test_multiple_batches(self):
        metric = MeanAbsoluteError()
        y_true_batches = constant([
            [39, 22, 73],
            [22, 50, 23]
        ], dtype=float32)
        y_pred_batches = constant([
            [80, 59, 52],
            [87, 8, 38],
        ], dtype=float32)

        for y_true, y_pred in zip(y_true_batches, y_pred_batches):
            metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = reduce_mean(abs(y_true_batches - y_pred_batches))
        self.assertAlmostEqual(expected.numpy(), result.numpy(), 5)

    def test_multiple_batches_with_empty_array(self):
        metric = MeanAbsoluteError()
        y_true_batches = ragged.constant([
            [39, 22, 73],
            [],
            [22, 50, 23]
        ], dtype=float32)
        y_pred_batches = ragged.constant([
            [80, 59, 52],
            [],
            [87, 8, 38],
        ], dtype=float32)

        for y_true, y_pred in zip(y_true_batches, y_pred_batches):
            metric.update_state(y_true, y_pred)
        result = metric.result()

        expected = reduce_mean(abs(y_true_batches - y_pred_batches).flat_values)
        self.assertAlmostEqual(expected.numpy(), result.numpy(), 5)

