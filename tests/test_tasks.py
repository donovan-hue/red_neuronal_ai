import http.client
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nexus' / 'src'))
import server


class TaskTests(unittest.TestCase):
    def setUp(self):
        with server._TASK_LOCK:
            for name in server._TASKS:
                server._TASKS[name] = False
                server._TASK_RESULTS[name] = {'state': 'idle'}

    def test_records_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'ROOT', Path(tmp)), \
             patch.object(server.subprocess, 'run', return_value=SimpleNamespace(returncode=2)):
            server._run_task('train', 1)
        self.assertEqual(server._TASK_RESULTS['train']['state'], 'failed')
        self.assertEqual(server._TASK_RESULTS['train']['returncode'], 2)
        self.assertFalse(server._TASKS['train'])

    def test_timeout_and_success(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'ROOT', Path(tmp)):
            with patch.object(server.subprocess, 'run', side_effect=server.subprocess.TimeoutExpired('cmd', 1)):
                server._run_task('eval', None)
            self.assertEqual(server._TASK_RESULTS['eval']['state'], 'timed_out')
            with patch.object(server.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
                server._run_task('eval', None)
            self.assertEqual(server._TASK_RESULTS['eval']['state'], 'succeeded')

    def test_task_endpoint_requires_authentication(self):
        with server._RATE_LOCK:
            server._RATE.clear()
        httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {'NEXUS_ADMIN_TOKEN': 'secret'}):
                for token, expected in [('', 403), ('secret', 200)]:
                    conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                    conn.request('GET', '/api/tasks', headers={'X-Nexus-Admin-Token': token})
                    response = conn.getresponse()
                    self.assertEqual(response.status, expected)
                    body = response.read()
                    if expected == 200:
                        self.assertIn(b'"state": "idle"', body)
                    conn.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()
