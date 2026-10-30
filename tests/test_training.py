import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import api
from model import RedNeuronal


class TrainingTests(unittest.TestCase):
    def test_batch_gradient_is_invariant_to_duplicate_samples(self):
        a, b = RedNeuronal(2, 3, 1), RedNeuronal(2, 3, 1)
        X, y = [[0, 1], [1, 0]], [[1], [0]]
        a.train(X, y, 5, .1)
        b.train(X * 2, y * 2, 5, .1)
        for name in ('W1', 'b1', 'W2', 'b2'):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name))

    def test_early_stopping_restores_best_validation_weights(self):
        model = RedNeuronal(2, 3, 1)
        X, y = [[0, 0], [1, 1]], [[0], [0]]
        val = ([[0, 0], [1, 1]], [[1], [1]])
        metrics = model.train(X, y, 100, .5, validation_data=val, patience=3)
        self.assertLess(metrics['epochs_completed'], 100)
        self.assertEqual(metrics['best_epoch'], 1)
        self.assertEqual(metrics['validation'], model.evaluate(*val))

    def test_xor_training_still_converges(self):
        model = RedNeuronal(2, 3, 1)
        X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]
        metrics = model.train(X, y, 10000, .4)
        self.assertEqual(metrics['train']['accuracy'], 1.0)

    def test_batch_predict_and_training_validation(self):
        client = api.app.test_client()
        resp = client.post('/predecir/lote', json={'X': [[0, 0], [1, 0]]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json['predicciones']), 2)
        self.assertEqual(client.post('/predecir/lote', json={'X': [[1]]}).status_code, 400)
        original = api.model
        try:
            api.model = RedNeuronal(2, 3, 1)
            with tempfile.TemporaryDirectory() as temp, patch.object(api, 'MODEL_PATH', Path(temp) / 'model.npz'), patch.dict(os.environ, {'API_TRAIN_TOKEN': 'secret'}):
                response = client.post('/entrenar', headers={'X-API-Train-Token': 'secret'}, json={
                    'X': [[0, 0], [1, 0]], 'y': [0, 1], 'epochs': 10,
                    'validation': {'X': [[0, 1]], 'y': [1]}, 'patience': 2})
                self.assertEqual(response.status_code, 200)
                self.assertIn('validation', response.json['metricas'])
        finally:
            api.model = original


if __name__ == '__main__':
    unittest.main()
