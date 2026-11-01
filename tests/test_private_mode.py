import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nexus' / 'src'))
import auth
import engine
import server


class PrivateModeTests(unittest.TestCase):
    def test_forces_local_and_prevents_provider_calls(self):
        cfg = {'provider': 'openai', 'providers': {'openai': {'kind': 'openai',
               'base_url': 'https://example.invalid', 'key_env': 'TEST_KEY', 'model': 'test'}}}
        with patch.dict(os.environ, {'NEXUS_PRIVATE_MODE': '1', 'TEST_KEY': 'test'}):
            with patch.object(engine, 'LocalEngine', return_value='local') as local:
                self.assertEqual(engine.make_engine(cfg, force='openai'), 'local')
                local.assert_called_once()
            api = engine.APIEngine(cfg, 'openai')
            with patch.object(api.httpx, 'post') as post:
                with self.assertRaises(RuntimeError):
                    api.complete('private')
                post.assert_not_called()

    def test_server_requires_auth_and_binds_loopback(self):
        cfg = {'server': {'host': '0.0.0.0', 'port': 8080}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'users.json'
            with patch.dict(os.environ, {'NEXUS_PRIVATE_MODE': '1', 'NEXUS_AUTH_FILE': str(path)}, clear=False):
                os.environ.pop('PORT', None)
                with self.assertRaises(auth.AuthConfigError):
                    server.server_address(cfg)
                auth.add_user(path, 'owner', 'admin')
                self.assertEqual(server.server_address(cfg), ('127.0.0.1', 8080))
                auth.revoke_user(path, 'owner')
                with self.assertRaises(RuntimeError):
                    server.server_address(cfg)
