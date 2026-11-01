import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nexus' / 'src'))
import auth
import server


class UserAuthTests(unittest.TestCase):
    def test_creation_revocation_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'users.json'
            token = auth.add_user(path, 'alice', 'admin')
            self.assertNotIn(token, path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(auth.authenticate(path, 'Bearer ' + token)['role'], 'admin')
            self.assertIsNone(auth.authenticate(path, 'Bearer ' + 'a' * 64))
            with self.assertRaises(ValueError):
                auth.add_user(path, 'alice', 'user')
            path.write_text('{')
            with self.assertRaises(auth.AuthConfigError):
                auth.authenticate(path, 'Bearer ' + token)

    def test_expiry_rotation_and_last_user_revocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'users.json'
            with patch.object(auth.time, 'time', return_value=1_000_000):
                token = auth.add_user(path, 'alice', 'admin', ttl_days=1)
            with patch.object(auth.time, 'time', return_value=1_086_399):
                self.assertIsNotNone(auth.authenticate(path, 'Bearer ' + token))
            with patch.object(auth.time, 'time', return_value=1_086_400):
                self.assertIsNone(auth.authenticate(path, 'Bearer ' + token))
                new_token = auth.rotate_user(path, 'alice', ttl_days=2)
                self.assertIsNone(auth.authenticate(path, 'Bearer ' + token))
                self.assertEqual(auth.authenticate(path, 'Bearer ' + new_token)['role'], 'admin')
            auth.revoke_user(path, 'alice')
            self.assertIsNone(auth.authenticate(path, 'Bearer ' + new_token))
            self.assertEqual(auth.load_users(path), [])

    def test_bad_expiry_rejected_and_legacy_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'users.json'
            with self.assertRaises(ValueError):
                auth.add_user(path, 'alice', 'user', ttl_days=0)
            token = auth.add_user(path, 'alice', 'user')
            data = json.loads(path.read_text())
            data['users'][0].pop('expires_at')
            path.write_text(json.dumps(data))
            self.assertIsNotNone(auth.authenticate(path, 'Bearer ' + token))
            data['users'][0]['expires_at'] = 'tomorrow'
            path.write_text(json.dumps(data))
            with self.assertRaises(auth.AuthConfigError):
                auth.authenticate(path, 'Bearer ' + token)

    def test_http_roles_and_revocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'users.json'
            user = auth.add_user(path, 'alice', 'user')
            admin = auth.add_user(path, 'bob', 'admin')
            with server._RATE_LOCK:
                server._RATE.clear()
            httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            def get(endpoint, token=''):
                conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port)
                conn.request('GET', endpoint, headers={'Authorization': 'Bearer ' + token})
                response = conn.getresponse()
                status = response.status
                body = json.loads(response.read())
                conn.close()
                return status, body
            try:
                with patch.dict(os.environ, {'NEXUS_AUTH_FILE': str(path)}):
                    self.assertEqual(get('/api/config')[0], 401)
                    self.assertEqual(get('/api/config', user)[0], 200)
                    self.assertEqual(get('/api/tasks', user)[0], 403)
                    self.assertEqual(get('/api/tasks', admin)[0], 200)
                    self.assertEqual(get('/api/status', user)[1]['auth_role'], 'user')
                    content = json.loads(path.read_text())
                    content['users'] = [item for item in content['users'] if item['name'] != 'alice']
                    path.write_text(json.dumps(content))
                    self.assertEqual(get('/api/config', user)[0], 401)
                    path.unlink()
                    self.assertEqual(get('/api/config', admin)[0], 503)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()
