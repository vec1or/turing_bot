"""Version 2.2 components: lateness replies, travel advice, profile facts, plans."""
from datetime import datetime
import unittest

from turing_bot import goal_plan, reactions, travel
from turing_bot.profile import Profile


class ReactionTests(unittest.TestCase):
    def test_apology_is_accepted_with_the_stated_wait(self):
        reply = reactions.answer("Друг опоздал на встречу на полчаса и извинился. Что бы ты ему ответил?")
        self.assertIn("полчаса", reply)
        self.assertIn("Ничего страшного", reply)

    def test_missing_apology_and_repetition_change_the_reply(self):
        base = reactions.answer("Друг опоздал на полчаса и извинился. Что ответишь?")
        none = reactions.answer("Подруга опоздала на час и не извинилась. Что ты ей скажешь?")
        again = reactions.answer("Друг опять опоздал, третий раз за неделю, и извинился. Что ответишь?")
        self.assertEqual(len({base, none, again}), 3)
        self.assertIn("не первый раз", again)
        self.assertNotIn("Ничего страшного", none)

    def test_own_lateness_produces_an_apology(self):
        reply = reactions.answer("Ты опоздал на встречу с другом на 20 минут. Что ты ему скажешь?")
        self.assertIn("Прости", reply)

    def test_feminine_forms(self):
        reply = reactions.answer("Подруга опоздала и даже не извинилась. Что ответишь?")
        self.assertNotIn("пришёла", reply)

    def test_advice_about_lateness_is_not_direct_speech(self):
        self.assertIsNone(reactions.answer("Как не опаздывать на пары?"))
        self.assertIsNone(reactions.answer("Что бы ты ответил другу на вопрос о погоде?"))


class SpeechTests(unittest.TestCase):
    def test_event_kind_selects_the_reply(self):
        sad = reactions.speech("Что скажешь другу, который провалил экзамен?")
        rude = reactions.speech("Что скажешь соседу, который сверлит стену ночью?")
        kind = reactions.speech("Что скажешь человеку, который помог тебе с курсовой?")
        self.assertIn("пересдашь", sad)
        self.assertIn("потише", rude)
        self.assertIn("Спасибо", kind)
        self.assertTrue(all("?" not in r and len(r) <= 300 for r in (sad, rude, kind)))

    def test_unrelated_or_lateness_questions_are_left_to_others(self):
        self.assertIsNone(reactions.speech("Что ты скажешь о погоде?"))
        self.assertIsNone(reactions.speech("Друг опоздал на час. Что скажешь?"))


class TravelTests(unittest.TestCase):
    def test_city_and_season_are_combined(self):
        astana = travel.answer("Что бы ты посоветовал человеку, который впервые приехал в Астану зимой?")
        almaty = travel.answer("Что посоветуешь человеку, который впервые едет в Алматы летом?")
        self.assertIn("ветер", astana)
        self.assertIn("Байтерек", astana)
        self.assertIn("Летом", almaty)
        self.assertLessEqual(len(astana), 300)

    def test_requires_an_arrival_and_a_request_for_advice(self):
        self.assertIsNone(travel.answer("Почему в Астане зимой так холодно?"))
        self.assertIsNone(travel.answer("Я приехал в Астану зимой."))


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = Profile(clock=lambda: datetime(2026, 12, 4, 14, 5))

    def test_core_facts_are_stable(self):
        self.assertEqual(self.profile.answer("Как тебя зовут?"), "Тимур.")
        self.assertEqual(self.profile.answer("Сколько тебе лет?"), "Девятнадцать.")
        self.assertIn("Караганд", self.profile.answer("Откуда ты?"))

    def test_clock_is_local(self):
        self.assertEqual(self.profile.answer("Который час?"), "Сейчас 14:05, если мои часы не врут.")
        self.assertEqual(self.profile.answer("Какое сегодня число?"), "Сегодня пятница, 4 декабря.")

    def test_embedded_or_third_person_topics_are_left_alone(self):
        self.assertIsNone(self.profile.answer("Как проверить, сколько времени заложить на дорогу?"))
        self.assertIsNone(self.profile.answer("Можешь определить, какой фильм я смотрел вчера?"))
        self.assertIsNone(self.profile.answer("Ты любишь кошек или собак?"))
        self.assertIsNone(self.profile.answer("Почему кошки мурлыкают?"))

    def test_alternatives_about_own_life_are_answered(self):
        self.assertIn("сестра", self.profile.answer("Есть ли у тебя братья или сёстры?"))
        self.assertIn("Сижу", self.profile.answer("Ты сейчас сидишь или стоишь?"))
        self.assertIn("не нейросеть", self.profile.answer("Ты точно не нейросеть?"))


class SessionFollowUpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from turing_bot.session import Session
        cls.bot = Session()

    def test_lateness_follow_up_keeps_the_situation(self):
        self.bot.reset()
        self.bot.answer("Что бы ты сказал другу, который опоздал на час и даже не извинился?")
        reply = self.bot.answer("А если он опаздывает уже третий раз подряд?")
        self.assertEqual(self.bot.last_route, "reaction")
        self.assertIn("не первый раз", reply)

    def test_official_questions_use_the_common_path(self):
        self.bot.reset()
        replies = [self.bot.answer(q) for q in (
            "Почему после дождя иногда появляется радуга? Объясни простыми словами.",
            "Что бы ты посоветовал человеку, который впервые приехал в Астану зимой?",
            "Друг опоздал на встречу на полчаса и извинился. Что бы ты ему ответил?")]
        self.assertIn("капл", replies[0])
        self.assertIn("ветер", replies[1])
        self.assertIn("полчаса", replies[2])
        self.assertTrue(all(len(r) <= 300 and "?" not in r for r in replies))


class GoalPlanTests(unittest.TestCase):
    def test_reminder_keeps_the_target_and_trigger(self):
        plan = goal_plan.answer("Как не забыть выключить утюг перед выходом из дома?")
        self.assertIn("выключить утюг", plan)
        self.assertIn("перед выходом", plan)

    def test_unsupported_and_factual_prompts(self):
        self.assertIsNone(goal_plan.answer("Как правильно принимать таблетки от давления?"))
        self.assertIsNone(goal_plan.answer("Почему небо голубое?"))


if __name__ == "__main__":
    unittest.main()


class FollowUpTests3(unittest.TestCase):
    """Version 3.0 follow-ups and explanations."""

    @classmethod
    def setUpClass(cls):
        from turing_bot.session import Session
        cls.bot = Session()

    def test_city_title_and_reason_follow_ups(self):
        bot = self.bot
        bot.reset()
        bot.answer("Откуда ты родом?")
        self.assertIn("Караганд", bot.answer("А там зимой холодно?"))
        bot.answer("Какую последнюю книгу ты прочитал?")
        self.assertIn("Пилате", bot.answer("О чём она?"))

    def test_everyday_explanations(self):
        from turing_bot.explanations import explain
        self.assertIn("статическое", explain("Почему зимой бьёт током?"))
        self.assertIn("лук", explain("Почему лук заставляет плакать?"))
        self.assertIsNone(explain("Почему только мы спорим?"))

    def test_no_question_mark_in_speech(self):
        reply = reactions.speech("Как бы ты попросил преподавателя перенести дедлайн?")
        self.assertNotIn("?", reply)
