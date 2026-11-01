import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nexus' / 'src'))
import rag


class RagUpdateTests(unittest.TestCase):
    def test_content_change_invalidates_even_with_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(rag, 'ROOT', tmp):
            data = Path(tmp) / 'data'
            data.mkdir()
            doc = data / 'book.txt'
            doc.write_text('a' * 250, encoding='utf-8')
            original = doc.stat()
            cfg = {'data': {'dir': 'data'}, 'rag': {'index': 'index/data.gz',
                   'chunk_chars': 900, 'overlap': 120, 'top_k': 4}}
            rag._CACHE.clear()
            first = rag.get_index(cfg)
            doc.write_text('b' * 250, encoding='utf-8')
            os.utime(doc, ns=(original.st_atime_ns, original.st_mtime_ns))
            second = rag.get_index(cfg)
            self.assertIsNot(first, second)
            self.assertEqual(len(rag._CACHE), 1)
            self.assertEqual(second.docs[0]['text'], 'b' * 250)
            rag._CACHE.clear()
