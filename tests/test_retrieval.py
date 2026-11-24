"""Reply retrieval: persona filters, NumPy encoder and the ranking contract."""
from pathlib import Path
import unittest

import numpy as np

from turing_bot.language import Pieces, RESERVED
from turing_bot.retrieval import Encoder, Retrieval, context_ids, persona_safe

ASSETS = Path(__file__).resolve().parents[1] / "turing_bot" / "assets" / "retrieval"


def tiny_encoder(dim=8, layers=1, ff=16):
    rng = np.random.default_rng(0)
    vocab = list(RESERVED) + list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") + ["▁привет", "▁как", "▁дела"]
    w = {"embed.weight": rng.normal(size=(len(vocab), dim)), "pos.weight": rng.normal(size=(40, dim)),
         "norm.weight": np.ones(dim), "norm.bias": np.zeros(dim)}
    for i in range(layers):
        p = f"blocks.{i}."
        w.update({p + "ln1.weight": np.ones(dim), p + "ln1.bias": np.zeros(dim), p + "ln2.weight": np.ones(dim),
                  p + "ln2.bias": np.zeros(dim), p + "qkv.weight": rng.normal(size=(3 * dim, dim)) * .1,
                  p + "proj.weight": rng.normal(size=(dim, dim)) * .1, p + "up.weight": rng.normal(size=(ff, dim)) * .1,
                  p + "up.bias": np.zeros(ff), p + "down.weight": rng.normal(size=(dim, ff)) * .1, p + "down.bias": np.zeros(dim)})
    for head in ("ctx", "rsp"):
        w.update({head + ".0.weight": rng.normal(size=(dim, dim)), head + ".0.bias": np.zeros(dim),
                  head + ".2.weight": rng.normal(size=(dim, dim)), head + ".2.bias": np.zeros(dim)})
    w = {k: np.asarray(v, dtype=np.float32) for k, v in w.items()}
    return Encoder(dict(vocab=vocab, layers=layers, heads=2, dim=dim), w)


class PersonaFilterTests(unittest.TestCase):
    def test_feminine_and_foreign_biography_are_rejected(self):
        for reply in ("Я вчера так устала, просто ужас.", "Я не знала, что так бывает.", "Мне 35, я уже на пенсии.",
                      "Моя жена против.", "Я бот, конечно.", "А ты как думаешь?", "Пиши на сайте, там помогут."):
            self.assertFalse(persona_safe(reply), reply)

    def test_ordinary_male_replies_pass(self):
        for reply in ("Я вчера так устал, просто ужас.", "Не знаю, мне больше нравится зима.", "Да нормально всё, спасибо."):
            self.assertTrue(persona_safe(reply), reply)


class EncoderTests(unittest.TestCase):
    def test_vectors_are_unit_length_and_deterministic(self):
        enc = tiny_encoder()
        a, b = enc.context("Привет, как дела"), enc.context("Привет, как дела")
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1., places=5)
        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(enc.context("Привет"), enc.reply("Привет")))

    def test_previous_turn_is_truncated_to_the_length_limit(self):
        pieces = Pieces(tiny_encoder().config["vocab"])
        self.assertLessEqual(len(context_ids(pieces, "как дела " * 30, "привет " * 50)), 40)


class RankingTests(unittest.TestCase):
    def test_filters_and_repetition(self):
        enc = tiny_encoder()
        replies = ["я сама не знаю", "нормально всё, спасибо", "нормально всё, спасибо"]
        contexts = ["как дела", "как дела", "как дела"]
        vectors = enc.contexts(contexts)
        scales = np.abs(vectors).max(axis=1) / 127.
        quant = np.round(vectors / scales[:, None]).astype(np.int8)
        dim = vectors.shape[1]
        bot = Retrieval(enc, replies, contexts, quant, scales.astype(np.float32),
                        np.zeros(dim, np.float32), np.eye(dim, dtype=np.float32))
        reply, score, details = bot.propose("как дела")
        self.assertEqual(reply, "Нормально всё, спасибо")
        self.assertIsNone(bot.propose("как дела", seen={"нормально всё, спасибо"}))

    def test_batched_encoding_matches_single(self):
        enc = tiny_encoder()
        texts = ["привет", "как дела у тебя сегодня"]
        batch = enc.contexts(texts)
        for i, t in enumerate(texts):
            np.testing.assert_allclose(batch[i], enc.context(t), atol=1e-5)


@unittest.skipUnless((ASSETS / "encoder.json").exists(), "индекс ответов не собран")
class ShippedIndexTests(unittest.TestCase):
    def test_index_loads_and_answers_small_talk(self):
        bot = Retrieval.load()
        found = bot.propose("Как дела?")
        self.assertIsNotNone(found)
        self.assertTrue(persona_safe(found[0]))
        self.assertLessEqual(len(found[0]), 300)


if __name__ == "__main__":
    unittest.main()
