import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from turing_bot.neural import BOS, EOS, SPECIAL, Generator, context_features


class NeuralTests(unittest.TestCase):
    def test_feature_map_is_deterministic_and_recency_sensitive(self):
        a, aw = context_features("Что ты читаешь?\nЛюблю детективы.", 4096, 224)
        b, bw = context_features("Что ты читаешь?\nЛюблю детективы.", 4096, 224)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(aw, bw)
        self.assertAlmostEqual(float(np.linalg.norm(aw)), 1.0, places=5)
        reverse, _ = context_features("Люблю детективы.\nЧто ты читаешь?", 4096, 224)
        self.assertFalse(np.array_equal(a, reverse))

    def test_packaged_model_ranks_without_changing_outputs(self):
        gen = Generator.load()
        ctx = "Что ты любишь читать?"
        answer = gen.generate(ctx)
        self.assertTrue(answer)
        scores = gen.score(ctx, [answer, "Совсем другая фраза."])
        self.assertTrue(all(math.isfinite(x) and x <= 0 for x in scores))
        self.assertAlmostEqual(scores[0], gen.score(ctx, [answer])[0], places=4)
        self.assertEqual(answer, gen.generate(ctx))
        with self.assertRaises(ValueError):
            Generator(gen.config, {**gen.weights, "out.bias": np.array([np.nan])})

    def test_training_and_numpy_step_have_the_same_logits(self):
        try:
            import torch
            from training.train import Network, snapshot
        except ImportError:
            self.skipTest("Проверка двух реализаций требует PyTorch из среды обучения.")
        cfg = {"format": "arena-context-gru-v1", "tokens": SPECIAL + ["да", "нет", "как", "."],
               "width": 32, "buckets": 128, "embedding": 12, "hidden": 16,
               "dropout": 0.0, "beam": 3, "response_limit": 12}
        torch.manual_seed(42)
        net = Network(cfg).eval()
        q = "Как ты думаешь?"
        ids, values = context_features(q, cfg["buckets"], cfg["width"])
        previous = [BOS, 4, 7]
        with tempfile.TemporaryDirectory() as root:
            snapshot(net, cfg, Path(root))
            gen = Generator.load(root)
            ctx = gen.encode(q)
            h = ctx.copy()
            actual = []
            for token in previous:
                h, logits = gen.step([token], h, ctx)
                actual.append(logits[0])
            with torch.no_grad():
                expected = net(torch.tensor(ids[None]), torch.tensor(values[None]), torch.tensor([previous]))[0]
                logp = expected.log_softmax(-1)[range(3), [4, 7, EOS]]
            np.testing.assert_allclose(np.stack(actual), expected.numpy(), rtol=2e-5, atol=1e-6)
            self.assertAlmostEqual(gen.score(q, ["Да."])[0], float(logp.mean()), places=5)
