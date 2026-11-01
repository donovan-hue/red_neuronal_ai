import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nexus' / 'src'))
import limits
import server


class SharedLimitTests(unittest.TestCase):
    def test_window_expiration_and_separate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'limits.sqlite')
            self.assertTrue(limits.allow_request(db, ('a', 'chat'), 2, now=100))
            self.assertTrue(limits.allow_request(db, ('a', 'chat'), 2, now=101))
            self.assertFalse(limits.allow_request(db, ('a', 'chat'), 2, now=102))
            self.assertTrue(limits.allow_request(db, ('a', 'admin'), 2, now=102))
            self.assertTrue(limits.allow_request(db, ('a', 'chat'), 2, now=160))

    def test_independent_processes_share_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'limits.sqlite')
            script = ('import sys; sys.path.insert(0, sys.argv[1]); import limits; '
                      'print(int(limits.allow_request(sys.argv[2], ("client", "chat"), 1, now=100)))')
            args = [sys.executable, '-c', script, str(ROOT / 'nexus' / 'src'), db]
            self.assertEqual(subprocess.check_output(args, text=True).strip(), '1')
            self.assertEqual(subprocess.check_output(args, text=True).strip(), '0')

    def test_daily_quota_is_shared_and_resets_by_utc_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'quota.sqlite')
            self.assertTrue(limits.reserve_daily_call(db, 1, now=100))
            self.assertFalse(limits.reserve_daily_call(db, 1, now=101))
            self.assertTrue(limits.reserve_daily_call(db, 1, now=86400))
            self.assertFalse(limits.reserve_daily_call(db, 1, now=100))

    def test_provider_does_not_call_network_when_quota_exhausted(self):
        from engine import APIEngine
        cfg = {'providers': {'openai': {'kind': 'openai', 'model': 'test',
              'base_url': 'https://example.invalid', 'key_env': 'TEST_PROVIDER_KEY'}}}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'TEST_PROVIDER_KEY': 'test', 'NEXUS_RATE_DB': str(Path(tmp) / 'quota.sqlite'),
            'NEXUS_API_DAILY_CALL_LIMIT': '1'}):
            engine = APIEngine(cfg, 'openai')
            fake = type('Response', (), {'raise_for_status': lambda self: None,
                        'json': lambda self: {'choices': [{'message': {'content': 'ok'}}]}})()
            with patch.object(engine.httpx, 'post', return_value=fake) as post:
                self.assertEqual(engine.complete('hola'), 'ok')
                with self.assertRaises(RuntimeError):
                    engine.complete('otra')
                self.assertEqual(post.call_count, 1)

    def test_unavailable_database_fails_closed(self):
        with patch.dict(os.environ, {'NEXUS_RATE_DB': '/not/a/real/directory/limits.sqlite'}):
            self.assertFalse(server._rate_allowed(('client', 'chat'), 10))
