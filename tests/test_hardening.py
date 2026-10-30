import http.client
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nexus' / 'src'))
import api
import server
from model import RedNeuronal


class HardeningTests(unittest.TestCase):
    def test_prediction_requires_both_inputs(self):
        response = api.app.test_client().post('/predecir', json={'x1': 1})
        self.assertEqual(response.status_code, 400)

    def test_checkpoint_rejects_nonfinite_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.npz'
            np.savez(path, W1=np.array([[np.nan]]), b1=np.zeros((1, 1)),
                     W2=np.ones((1, 1)), b2=np.zeros((1, 1)))
            with self.assertRaises(ValueError):
                RedNeuronal.cargar(path)

    def test_rate_limiter_discards_expired_addresses(self):
        with server._RATE_LOCK:
            server._RATE.clear()
            server._RATE['old'] = [0.0]
        with patch.object(server.time, 'monotonic', return_value=100):
            self.assertTrue(server._rate_allowed('new', 1))
            self.assertFalse(server._rate_allowed('new', 1))
        self.assertNotIn('old', server._RATE)

    def test_rate_limiter_rejects_new_addresses_at_capacity(self):
        with server._RATE_LOCK:
            server._RATE.clear()
            server._RATE['existing'] = [100.0]
        with patch.object(server, '_RATE_MAX_ADDRESSES', 1), patch.object(server.time, 'monotonic', return_value=100):
            self.assertFalse(server._rate_allowed('new', 1))
            self.assertTrue(server._rate_allowed('existing', 2))

    def test_busy_chat_returns_503_and_releases_capacity(self):
        class FakeHandler:
            headers = {}
            client_address = ('test', 123)
            def _send(self, code, body, **kwargs):
                self.status = code
            def _stream_chat(self, *args):
                raise AssertionError('no debe iniciar')

        class Engine:
            kind = 'local'

        handler = FakeHandler()
        semaphore = threading.BoundedSemaphore(1)
        semaphore.acquire()
        old_engine = server._STATE['engine']
        try:
            server._STATE['engine'] = Engine()
            with patch.object(server, 'boot', return_value=object()), patch.object(server, '_CHAT_SLOTS', semaphore):
                server.Handler.chat(handler, {'message': 'hola'})
                self.assertEqual(handler.status, 503)
        finally:
            server._STATE['engine'] = old_engine
            semaphore.release()

    def test_chat_memory_is_isolated_between_requests(self):
        class FakeHandler:
            headers = {}
            client_address = ('test', 123)
            seen = []
            def _stream_chat(self, agent, *args):
                self.seen.append(agent.memory.recall('color'))
                agent.memory.remember('color', 'privado')
            def _send(self, code, body, **kwargs):
                self.fail = code

        class Engine:
            kind = 'local'

        class SharedAgent:
            cfg = {'agent': {'memory_db': 'models/memory.sqlite', 'trajectory_exemplars': 2}}
            engine = Engine()
            index = None

        handler = FakeHandler()
        original = server._STATE['engine']
        try:
            server._STATE['engine'] = SharedAgent.engine
            with patch.object(server, 'boot', return_value=SharedAgent()):
                server.Handler.chat(handler, {'message': 'hola'})
                server.Handler.chat(handler, {'message': 'hola'})
            self.assertEqual(handler.seen, ['(sin hechos coincidentes)', '(sin hechos coincidentes)'])
        finally:
            server._STATE['engine'] = original

    def test_failed_save_does_not_change_live_model(self):
        client = api.app.test_client()
        original = api.model
        try:
            api.model = RedNeuronal(2, 3, 1)
            active = api.model
            before = active.W1.copy()
            with patch.dict(os.environ, {'API_TRAIN_TOKEN': 'secret'}), \
                 patch.object(RedNeuronal, 'guardar', side_effect=OSError('private path')):
                response = client.post('/entrenar', headers={'X-API-Train-Token': 'secret'},
                                       json={'X': [[0, 1]], 'y': [1], 'epochs': 2})
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('private path', response.get_data(as_text=True))
            self.assertIs(api.model, active)
            np.testing.assert_array_equal(active.W1, before)
        finally:
            api.model = original

    def test_download_requires_admin_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'ROOT', Path(tmp)):
            runs = Path(tmp) / 'runs'
            runs.mkdir()
            (runs / 'test.log').write_text('private')
            httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.dict(os.environ, {'NEXUS_ADMIN_TOKEN': 'secret'}):
                    for token, expected in [('', 403), ('wrong', 403), ('secret', 200)]:
                        conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                        conn.request('GET', '/api/download/test.log', headers={'X-Nexus-Admin-Token': token})
                        result = conn.getresponse()
                        self.assertEqual(result.status, expected)
                        result.read()
                        conn.close()
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
