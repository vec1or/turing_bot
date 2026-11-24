import unittest

from turing_bot.reader import Reader


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = Reader.load()

    def assert_span(self, question, history, expected):
        result = self.reader.propose(question, history)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], expected)
        self.assertGreaterEqual(result[1], 0.)
        self.assertLessEqual(result[1], 1.)

    def test_novel_objects_and_surface_forms(self):
        cases = [
            ("Мне подарили фиолетовый зонт.", "Какого цвета был зонт?", "В сообщении — «фиолетовый зонт»."),
            ("Мы любовались фиолетовыми лурпами.", "Какого цвета были лурпы?", "В сообщении — «фиолетовыми лурпами»."),
            ("В сундуке лежали семь бирюзовых норпов и три жетона.", "Сколько норпов лежало в сундуке?", "Семь."),
            ("В вагоне уцелел рифлёный шипрак.", "Каким был шипрак?", "В сообщении — «рифлёный шипрак»."),
            ("У меня синий шар и красная книга.", "Какого цвета книга?", "В сообщении — «красная книга»."),
            ("Мне отдали камень цвета спелой вишни.", "Какого цвета камень?", "Цвета спелой вишни."),
        ]
        for statement, question, expected in cases:
            with self.subTest(question=question):
                self.assert_span(question, [statement], expected)

    def test_places_and_people(self):
        cases = [
            ("Я спрятала медный флиндар в полосатую шкатулку.", "Куда я спрятала флиндар?", "В полосатую шкатулку."),
            ("Куб на мосту, а шар в синей лодке.", "Где куб?", "На мосту."),
            ("Кристалл лежал под мостом вчера.", "Где кристалл?", "Под мостом."),
            ("Моего робота зовут Ягран.", "Как зовут робота?", "Ягран."),
            ("Я назвала грифона Чекмар.", "Как я назвала грифона?", "Чекмар."),
            ("Лира подарила мне хрупкий маяк.", "Кто подарил маяк?", "Лира."),
        ]
        for statement, question, expected in cases:
            with self.subTest(question=question):
                self.assert_span(question, [statement], expected)

    def test_read_inline_and_across_turns(self):
        q = "В атласе был лиловый грот. Какого цвета был грот?"
        self.assert_span(q, [], "В сообщении — «лиловый грот».")
        self.assert_span("Сколько шпринов?", ["Мы увидели 17 шпринов.", "Ты меня слышишь?"],
                         "17.")
        self.assertIsNone(self.reader.propose("Сколько шпринов?", []))

    def test_missing_unasserted_or_conflicting_evidence(self):
        cases = [
            (["Мне подарили огромный зонт."], "Какого цвета был зонт?"),
            (["Я купила не фиолетовый зонт."], "Какого цвета зонт?"),
            (["Я хотела купить изумрудный зонт."], "Какого цвета был зонт?"),
            (["Какого цвета фиолетовый зонт"], "Какого цвета зонт?"),
            (["Если закончится ливень, я куплю зелёный зонт."], "Какого цвета зонт?"),
            (["У меня жёлтый зонт.", "У меня красный зонт."], "Какого цвета зонт?"),
            (["У меня жёлтый зонт.", "Потом я его перекрасила."], "Какого цвета зонт?"),
            (["Синий куб стоит под стулом и перед дверью."], "Где куб?"),
            (["У меня куб и Лира положила шар в коробку."], "Где куб?"),
            (["Куб стоит на полке."], "Куда я спрятал куб?"),
            (["Лира рассказывала о кубе на вокзале."], "Где куб?"),
            (["Сейчас на столе лежит синий блок."], "Какого цвета тяжёлый блок?"),
            (["Катя купила свисток."], "Кто продал свисток?"),
            (["На столе три ленты."], "Сколько у меня знакомых?"),
            (["Зеркало было белым."], "Каким будет лето?"),
        ]
        for history, question in cases:
            with self.subTest(history=history, question=question):
                self.assertIsNone(self.reader.propose(question, history))

    def test_under_specified_and_unrelated_requests(self):
        for question in ("Где?", "А что дальше?", "Как работает трамвай?", "Какого цвета?", ""):
            with self.subTest(question=question):
                self.assertIsNone(self.reader.propose(question, ["Розовый холм стоял за мостом."]))


if __name__ == "__main__":
    unittest.main()
