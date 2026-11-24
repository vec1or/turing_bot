import re
import unittest

from turing_bot.situations import Situations


class SituationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = Situations()

    def check(self, question, *fragments):
        answer = self.m.answer(question)
        self.assertIsInstance(answer,str,question)
        self.assertLessEqual(len(answer),300)
        for fragment in fragments:
            self.assertIn(fragment,answer.casefold(),(question,answer))
        return answer

    def test_practical_domains(self):
        for q,expected in [
            ("Как уложить стеклянные кружки при переезде?", "отдельно"),
            ("Как устроить уборку, если весь рабочий стол в бумагах?", "стола"),
            ("Как подготовиться к зачёту, когда до него один вечер?", "памяти"),
            ("Что делать с приложением, которое вылетает при запуске?", "ожидал"),
            ("Что делать, если файл с курсовой исчез?", "историю версий"),
            ("Чем помочь команде, если мнения разделились?", "критерии"),
            ("Как организовать студенческий проект?", "результат"),
            ("Как отказаться, когда сосед просит списать контрольную?", "списывания"),
        ]:
            with self.subTest(q=q):
                self.check(q,expected)

    def test_speaker_and_ethics(self):
        self.check("Что будешь делать, если друг обманул тебя?", "объяснил, как поступок")
        self.check("Станешь врать, если друг просит придумать оправдание? Как поступишь?", "за друга")
        self.check("Что бы ты сделал с найденным кошельком?", "владельцу")
        self.check("Что сделаешь при незаслуженном обвинении?", "свою версию")
        self.check("Что бы ты сделал, если чужая тайна касается безопасности человека?", "предупредил")

    def test_hypothetical_choices_have_an_actual_answer(self):
        self.check("Представь, что можно побывать в будущем. Куда направишься?", "университет")
        self.check("Если бы ты мог путешествовать во времени, с какого места начал бы?", "типографию")
        self.check("Если бы ты мог читать мысли, то как распорядился бы способностью?", "просьбе")
        self.check("На что потратишь дополнительный час?", "откладывал")

    def test_count_preserves_intent(self):
        for q,expected in [
            ("Назови три причины медленной работы программы.", "данных"),
            ("Дай три совета перед защитой проекта.", "мысль"),
            ("Назови три причины, почему люди откладывают дела.", "первый шаг"),
            ("Представь необитаемый остров. Какие три вещи возьмёшь?", "воды"),
        ]:
            with self.subTest(q=q):
                out=self.check(q,expected)
                self.assertEqual(re.findall(r"\b\d+\)",out),["1)","2)","3)"])
        out=self.check("Назови две возможные причины, почему ноутбук тормозит.", "данных")
        self.assertEqual(len(re.findall(r"\b\d+\)",out)),2)

    def test_abstains_without_the_needed_relation(self):
        for q in ("Что делает почтальон на Марсе?", "На какой странице моей тетради конспект?",
                  "Сколько стоит новый ноутбук?", "Зачем зебре полоски?", "Какой это телефон?",
                  "Как узнать столицу страны?", "Где купить картину со студентом?", "Что такое проект?",
                  "Какой цвет у моего компьютера?", "Как найти уязвимость в чужом сайте?"):
            with self.subTest(q=q):
                self.assertIsNone(self.m.answer(q))

    def test_action_questions_and_observations_ask_different_things(self):
        self.check("Как понять, что другу нужна поддержка?", "отказываться от привычных встреч")
        self.check("Если бы сеть исчезла на выходные, чем бы ты занялся?", "гулять")
        out=self.check("Если бы удалось остановить время на десять минут, что бы ты сделал?", "обдумал")
        self.assertNotIn("освободившийся час",out)
        self.check("Назови три плюса учёбы в команде.", "другому")
        self.check("Назови три причины срыва сроков у команды.", "задержке")
        self.check("Куда бы ты перенёсся на машине времени и почему?", "типографию", "букв")


if __name__=="__main__":
    unittest.main()
