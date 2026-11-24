import unittest

from turing_bot.explanations import explain
from turing_bot.imagination import Imagination
from turing_bot.rounds import Abstentions, clean, standalone, short_operation, requested_format
from turing_bot.session import meaningful_generation
from turing_bot.evidence import limitation
from turing_bot.writing import Writing


class RoundTests(unittest.TestCase):
    def test_wrapped_and_internal_quotes(self):
        self.assertEqual(clean(' «  Чай   или кофе? » '), "Чай или кофе?")
        self.assertEqual(clean('"Выбираешь чай"?'), "Выбираешь чай?")
        self.assertEqual(clean('«да» или «нет»'), '«да» или «нет»')

    def test_one_shot_and_variety(self):
        pool = Abstentions()
        answers = [pool.answer("Неизвестный термин") for _ in range(75)]
        self.assertEqual(len(answers),len(set(answers)))
        self.assertTrue(all(standalone(a) and len(a)<=300 for a in answers))
        self.assertFalse(standalone("Потому что потому что луна."))
        self.assertFalse(standalone("А бы было бы?"))
        self.assertFalse(meaningful_generation("Я бы сделал.", "Что бы ты сделал?"))

    def test_generative_slots(self):
        component = Imagination()
        phrase = component.answer("Придумай одно предложение со словом «шмякушка».")
        self.assertIn("шмякушка", phrase)
        self.assertTrue(standalone(phrase))
        self.assertNotEqual(component.answer("Чем пахнет осень?"),component.answer("Чем пахнет суббота?"))
        self.assertIsNone(component.answer("Какого цвета мой новый рюкзак?"))
        self.assertIsNone(explain("Где я оставил термос?"))
        self.assertIn("холодной",explain("Отчего зеркало запотевает?"))

    def test_operations(self):
        self.assertEqual(short_operation("Посчитай 2,5 × 6"),"Получается 15.")
        self.assertEqual(short_operation("7 / 0"),"На ноль делить нельзя.")
        self.assertIsNone(short_operation("Выбери 2 + 3 или 4?"))
        self.assertEqual(short_operation('Напиши слово «трёх» наоборот.'),"хёрт.")
        self.assertEqual(requested_format("Сначала проверю часы. Другу напишу позже.","Одной фразой."),"Сначала проверю часы.")
        self.assertIsNone(requested_format("Беру чай.","Ответь без слова «чай»."))

    def test_text_requests_and_missing_data(self):
        writing = Writing()
        scene = Imagination()
        self.assertNotEqual(writing.answer("Какой запах у тебя ассоциируется с осенью?",scene),
                            writing.answer("Какой запах у тебя ассоциируется с кухней?",scene))
        self.assertRegex(writing.answer("Закончи фразу: «На ветке сидела …»",scene),r"^[А-ЯЁ][а-яё]+\.$")
        self.assertIsNotNone(limitation("Какая влажность сейчас в моей коробке?"))
        self.assertIsNone(limitation("Почему на улице сыро?"))
