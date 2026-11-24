import unittest

from turing_bot.grounding import Confidence


# Open probes authored for this component before the final independent rerun.
# Related alternatives are comparative examples, not candidate texts at runtime.
PROBES = [
    ("Как ухаживать за рассадой?", "Поливай рассаду, когда земля сверху высохнет.", "Моя машина сегодня в гараже."),
    ("Куда пойдут деньги за билеты?", "Деньги от продажи билетов останутся в кассе.", "Ему очень нравятся яблоки."),
    ("Когда начнётся концерт?", "Концерт начнётся в семь.", "У неё большое окно."),
    ("Чем отмыть чернила с рук?", "Можно попробовать мыло для рук.", "Он учится по средам."),
    ("Где лежит моя закладка?", None, "Ты положил её за диван."),
    ("Кто ждёт меня у калитки?", None, "Конечно, твоя сестра."),
    ("Какая очередь сегодня у канатной дороги?", None, "Сейчас там семь человек."),
    ("Сколько сейчас стоит редкий кадастрон?", None, "Около шестидесяти рублей."),
    ("Почему подзорные трубы вчера начали варить варенье?", None, "Потому что им очень жарко."),
    ("Как лучше упаковать хрупкий компас?", None, "Мы женаты уже много лет."),
    ("Что значит «жерепан»?", None, "Это маленькая рыба."),
    ("Какая из них?", None, "Вторая с конца."),
]


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Confidence.load()

    def test_assessment_is_bounded_and_shared_weights_match(self):
        second = Confidence.load(generator=self.model.semantic.gen)
        for q, _, a in PROBES:
            v = self.model.assess(q, a)
            self.assertGreaterEqual(v, 0.)
            self.assertLessEqual(v, 1.)
            self.assertEqual(second.assess(q, a), v)
        self.assertEqual(self.model.assess("Где?", ""), 0.)

    def test_unknown_observations_cannot_become_assertions(self):
        for q, _, _ in PROBES[4:9]:
            with self.subTest(q=q):
                self.assertTrue(self.model.needs_evidence(q))
        for q, _, _ in PROBES[4:]:
            with self.subTest(q=q):
                clarify = self.model.clarify(q)
                self.assertTrue(clarify.endswith("?"))
                self.assertLessEqual(len(clarify), 300)

    def test_composes_with_unseen_reference_and_viewer_perspective(self):
        self.assertIn("жерепан", self.model.clarify(PROBES[10][0]))
        self.assertIn("твоя закладка", self.model.clarify(PROBES[4][0]))
        self.assertIn("компас", self.model.clarify(PROBES[9][0]))
        self.assertIn("варить варенье", self.model.clarify(PROBES[8][0]))


if __name__ == "__main__":
    unittest.main()
