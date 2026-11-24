import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from turing_bot.session import Session, display_text
from turing_bot.word_model import WordModel, alignment, fit, realize


class SessionTests(unittest.TestCase):
    def test_unicode_output_limit(self):
        long = "Очень длинное продолжение " * 40
        self.assertEqual(display_text("Первая фраза. " + long), "Первая фраза.")
        self.assertLessEqual(len(display_text(long)), 300)
        self.assertEqual(display_text("е\u0308лка"), "ёлка")

    def test_copy_is_an_operation_on_new_words(self):
        symbolic = alignment("кометой", ["комета"])
        self.assertEqual(realize(symbolic, ["газета"]), "газетой")
        self.assertIsNone(realize(symbolic, []))

    def test_session_supplies_history_and_clears_it(self):
        bot = Session.__new__(Session)
        from collections import deque
        bot.turns = deque(maxlen=1)
        bot.reader = Mock()
        bot.reader.propose.return_value = None
        bot.knowledge = Mock()
        bot.knowledge.propose.return_value = None
        bot.articles = Mock()
        bot.articles.propose.return_value = None
        bot.articles.support.return_value = None
        bot.advice = Mock()
        bot.advice.propose.return_value = None
        bot.social = Mock()
        bot.social.propose.return_value = None
        from turing_bot.rounds import Abstentions
        bot.abstentions = Abstentions()
        bot.last_route = None
        bot.last_retrieval = None
        bot.first_question = None
        bot.first_answer = None
        for component in ("persona", "situations", "imagination", "writing", "reasoning", "practical", "profile"):
            handler = Mock()
            handler.answer.return_value = None
            setattr(bot, component, handler)
        bot.confidence = Mock()
        bot.confidence.needs_evidence.return_value = False
        bot.retrieval = Mock()
        bot.retrieval.propose.return_value = ("Давай сначала посмотрим.", .9, dict(dense=.7))
        bot.answer("С чего начать?")
        self.assertIsNone(bot.retrieval.propose.call_args.args[1])
        bot.answer("А дальше?")
        # A short follow-up is read together with the previous reply.
        self.assertEqual(bot.retrieval.propose.call_args.args[1], "Давай сначала посмотрим.")
        bot.reset()
        bot.answer("А дальше?")
        self.assertIsNone(bot.retrieval.propose.call_args.args[1])

    def test_word_statistics_are_repeatable_and_do_not_store_dialogues(self):
        rows = [{"context": "Куда идёт комета?", "response": "Комета летит очень далеко."},
                {"context": "Летит ли ракета?", "response": "Ракета летит далеко в космос."}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.json.gz"
            fit(rows, path)
            first = path.read_bytes()
            fit(rows, path)
            self.assertEqual(first, path.read_bytes())
            raw = gzip.decompress(first).decode()
            self.assertNotIn(rows[0]["response"], raw)
            model = WordModel(json.loads(raw))
            self.assertTrue(model.candidates("Куда летит ракета?"))
