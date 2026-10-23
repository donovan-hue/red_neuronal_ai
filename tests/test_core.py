import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import api
from model import RedNeuronal

sys.path.insert(0, str(ROOT / "nexus" / "src"))
import nexus_single
import tools
from engine import APIEngine
from agent import Agent
from rag import BM25
from server import Handler


class CalculatorTests(unittest.TestCase):
    def test_valid_arithmetic_including_zero_operands(self):
        self.assertEqual(tools.calc("3+0"), "3")
        self.assertEqual(tools.calc("3*0"), "0")
        self.assertEqual(tools.calc("(12+7)*3"), "57")
        self.assertEqual(nexus_single.calc("3+0"), "3")
        self.assertEqual(nexus_single.calc("3*0"), "0")

    def test_calculator_rejects_silently_mutated_expressions(self):
        for expression in ("2foo+3", "__import__('os')", "", "1" * 300):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    tools.calc(expression)
                with self.assertRaises(ValueError):
                    nexus_single.calc(expression)

    def test_python_tool_is_disabled(self):
        self.assertIn("deshabilitada", tools.python_exec("print(1)"))
        self.assertIn("deshabilitada", nexus_single.python_exec("print(1)"))


class MemoryTests(unittest.TestCase):
    def test_recall_filters_by_query(self):
        memory = tools.Memory(":memory:")
        try:
            memory.remember("color", "azul")
            memory.remember("ciudad", "Tlaquepaque")
            self.assertIn("azul", memory.recall("color"))
            self.assertNotIn("Tlaquepaque", memory.recall("color"))
        finally:
            memory.c.close()


class ModelTests(unittest.TestCase):
    def test_npz_round_trip_without_pickle(self):
        model = RedNeuronal(2, 3, 1)
        sample = [[0, 0], [0, 1], [1, 0], [1, 1]]
        before = model.forward(sample)
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "weights.npz"
            model.guardar(str(filename))
            loaded = RedNeuronal.cargar(str(filename))
            self.assertTrue((before == loaded.forward(sample)).all())

    def test_committed_checkpoint_is_loadable(self):
        model = RedNeuronal.cargar(str(ROOT / "modelo.npz"))
        values = model.forward([[0, 0], [0, 1], [1, 0], [1, 1]]).round().ravel().tolist()
        self.assertEqual(values, [0.0, 1.0, 1.0, 0.0])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()

    def test_health_and_prediction(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.post("/predecir", json={"x1": 0, "x2": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["entrada"], [0.0, 1.0])

    def test_invalid_training_requests_are_rejected(self):
        missing = self.client.post("/entrenar", json={"X": [[0, 0]], "y": [0], "epochs": 0})
        malformed = self.client.post("/entrenar", json={"X": [[0]], "y": [0]})
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(malformed.status_code, 400)

    def test_non_json_is_rejected(self):
        response = self.client.post("/predecir", data="[]", content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_training_requires_token_and_persists_safe_checkpoint(self):
        with patch.dict(os.environ, {"API_TRAIN_TOKEN": ""}):
            denied = self.client.post("/entrenar", json={"X": [[0, 0]], "y": [0]})
            self.assertEqual(denied.status_code, 503)
        original_model = api.model
        try:
            api.model = RedNeuronal(2, 3, 1)
            with tempfile.TemporaryDirectory() as temp:
                with patch.object(api, "MODEL_PATH", Path(temp) / "model.npz"), patch.dict(os.environ, {"API_TRAIN_TOKEN": "test-secret"}):
                    response = self.client.post("/entrenar", headers={"X-API-Train-Token": "test-secret"},
                                                json={"X": [[0, 0], [0, 1]], "y": [0, 1], "epochs": 2})
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue((Path(temp) / "model.npz").is_file())
        finally:
            api.model = original_model


class FakeEngine:
    kind = "api"

    def choice(self, *_args):
        return []

    def complete(self, *_args, **_kwargs):
        return "Respuesta final: 999"

    def info(self):
        return {"kind": "fake"}


class AgentTests(unittest.TestCase):
    def test_guard_corrects_and_verifies_model_arithmetic(self):
        cfg = {"agent": {"memory_db": "", "max_steps": 2, "temperature": 0.2,
                         "top_p": 0.9, "max_tokens": 80, "compact_prompt": True,
                         "trajectory_exemplars": 0, "verify": True, "reflect": False,
                         "guard": True, "self_consistency": 1},
               "rag": {"top_k": 4}}
        memory = tools.Memory(":memory:")
        agent = Agent(cfg, FakeEngine(), memory=memory)
        result = agent.solve("Calcula 3*4", k=1)
        self.assertEqual(result["answer"], "12")
        self.assertTrue(result["verify_ok"])
        self.assertTrue(result["guarded"])
        agent.engine.complete = lambda *_args, **_kwargs: "No estoy seguro"
        self.assertFalse(agent.verify("pregunta", "razonamiento", "respuesta")[0])
        agent.engine.complete = lambda *_args, **_kwargs: "PASS"
        self.assertTrue(agent.verify("pregunta", "razonamiento", "respuesta")[0])
        agent.engine.complete = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError())
        self.assertFalse(agent.verify("pregunta", "razonamiento", "respuesta")[0])
        memory.c.close()


class EngineTests(unittest.TestCase):
    def test_rendered_prompt_preserves_structured_roles(self):
        prompt = "<|system|>\npolicy<|end|>\n<|user|>\nquestion<|end|>\n<|assistant|>\n"
        self.assertEqual(APIEngine._messages_from_prompt(prompt),
                         [{"role": "system", "content": "policy"},
                          {"role": "user", "content": "question"}])

    def test_cached_bm25_round_trip(self):
        index = BM25([{"src": "test", "text": "gato azul gato"},
                      {"src": "test", "text": "perro rojo"}])
        restored = BM25.from_payload(index.to_payload())
        self.assertEqual(restored.search("gato", 1)[0]["text"], "gato azul gato")


class ServerPathTests(unittest.TestCase):
    def test_static_path_cannot_escape_web_root(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "web"
            base.mkdir()
            target = Path(temp) / "secret"
            target.write_text("secret")
            self.assertIsNone(Handler._safe_file(None, base, "../secret"))
            self.assertIsNotNone(Handler._safe_file(None, base, "index.html"))


class SingleFileTests(unittest.TestCase):
    def test_natural_language_numeric_answer(self):
        self.assertEqual(nexus_single.calc_nl("Halla el mcm de 84 y 126"), "252")

    def test_generated_html_uses_text_content_not_untrusted_inner_html(self):
        self.assertIn("buffer.indexOf(String.fromCharCode(10,10))", nexus_single.HTML)
        self.assertIn("out.textContent", nexus_single.HTML)
        self.assertNotIn("out.innerHTML", nexus_single.HTML)


if __name__ == "__main__":
    unittest.main()
