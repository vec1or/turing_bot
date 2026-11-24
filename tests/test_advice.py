import unittest

from turing_bot.advice import Advice


class AdviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Advice.load()

    def test_positive_and_contested_advice(self):
        self.assertIn("налево", self.model.propose("Что делать перед тем, как перейти дорогу?")[0])
        for q in ("Что делать, если промокли ботинки?", "Что посоветуешь перед выходом из дома?",
                  "Подскажи, где поискать потерянные звёзды в кармане?", "Почему над городом дождь?"):
            self.assertIsNone(self.model.propose(q))

    def test_evidence_scopes_compared_conditions(self):
        evidence="Местность с засушливым летом и холодной, снежной зимой."
        self.assertIn("потеплее",self.model.propose("Что посоветуешь туристу зимой?",evidence)[0])
        self.assertIsNone(self.model.propose("Что посоветуешь туристу летом?",evidence))
        self.assertIsNone(self.model.propose("Что посоветуешь в мае?","Во дворе стоят три стула."))
