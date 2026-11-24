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
        bot.generator = Mock()
        bot.generator.generate.return_value = "Давай сначала посмотрим."
        bot.generator.score.return_value = [-1.]
        bot.word_model = Mock()
        bot.word_model.candidates.return_value = []
        bot.word_model.cues.return_value = {}
        bot.word_model.fluency.return_value = -2.
        bot.cloze = Mock()
        bot.cloze.complete.return_value = []
        bot.reader = Mock()
        bot.reader.propose.return_value = None
        bot.stance = Mock()
        bot.stance.propose.return_value = None
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
        for component in ("persona", "situations", "imagination", "writing", "reasoning", "practical", "associations", "profile"):
            handler = Mock()
            handler.answer.return_value = None
            setattr(bot, component, handler)
        bot.planner = Mock()
        bot.planner.adjective.return_value = None
        bot.planner.hints.return_value = []
        bot.language = Mock()
        bot.language.generate.return_value = "Давай сначала посмотрим."
        bot.confidence = Mock()
        bot.confidence.needs_evidence.return_value = False
        bot.confidence.assess.return_value = .9
        bot.confidence.threshold = .77
        bot.answer("С чего начать?")
        bot.answer("А дальше?")
        self.assertEqual(bot.generator.generate.call_args.args[0], "С чего начать?\nДавай сначала посмотрим.\nА дальше?")
        bot.reset()
        bot.answer("Привет.")
        self.assertEqual(bot.generator.generate.call_args.args[0], "Привет.")

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
