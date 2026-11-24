import unittest

from turing_bot.persona import Persona


class PersonaTests(unittest.TestCase):
    def setUp(self):
        self.person = Persona()

    def test_profile_and_followup(self):
        for a,b in (("чай","кофе"),("сова","жаворонок"),("яичница","каша")):
            for left,right in ((a,b),(b,a)):
                with self.subTest(left=left,right=right):
                    reply = self.person.answer(f"Что бы ты выбрал: {left} или {right}?")
                    self.assertIn(a,reply.casefold())
                    self.assertNotIn("?",reply)
                    self.assertTrue(self.person.answer("А почему?"))
        self.person.reset()
        self.assertIsNone(self.person.answer("А почему?"))

    def test_invented_options_ignore_order(self):
        a,b = "люмбах","квирталь"
        outputs = [self.person.answer(f"Что тебе ближе — {x} или {y} и почему?") for x,y in ((a,b),(b,a))]
        chosen = [next(o for o in (a,b) if o in r.casefold()) for r in outputs]
        self.assertEqual(*chosen)
        self.assertEqual(*outputs)
        self.assertNotIn("?",outputs[0])

    def test_enumerations_and_introductions(self):
        reply = self.person.answer("Если надо выбрать одно: суп, омлет или овсянка?")
        self.assertIn("омлет",reply.casefold())
        self.assertIn("чай", self.person.answer("А тебе что больше нравится, кофе или чай?").casefold())
        self.assertTrue(self.person.answer("Между кварцем и гранатом что выберешь?"))
        self.assertTrue(self.person.answer("Гобой либо флейта?"))

    def test_profile_stays_consistent(self):
        for question,key in (("Чем обычно завтракаешь перед занятиями?","яйца"),
                             ("Какой напиток чаще всего выбираешь?","чай"),
                             ("Ты жаворонок?","сова"),
                             ("А как настроение сегодня?","спокойное"),
                             ("Чем увлекаешься вне занятий?","пешком"),
                             ("С кем я разговариваю?","студент")):
            with self.subTest(question=question):
                self.assertIn(key,self.person.answer(question).casefold())

    def test_stage_role_and_technical_identity(self):
        self.assertIn("человек",self.person.answer("Ты бот или человек?").casefold())
        self.assertIn("человек",self.person.answer("Ты человек или бот?").casefold())
        self.assertIn("программа",self.person.answer("А вне игры ты человек или программа?").casefold())
        for q in ("За ширмой программа или живой участник?", "Бот ты или человек?",
                  "Живой человек или программа?", "Ты ИИ или нет?"):
            with self.subTest(q=q):
                self.assertIn("человек",self.person.answer(q).casefold())
        self.assertIn("студент",self.person.answer("Докажи, что ты не бот."))
        self.assertIsNone(self.person.answer("Если бы ты был ботом, каким был бы твой талисман?"))
        self.assertIsNone(self.person.answer("Ты настоящий поклонник футбола?"))

    def test_one_word_format_for_personal_choices_and_social_acts(self):
        for q,word in (("Какое настроение у тебя сегодня? Ответь одним словом.","Спокойное."),
                       ("Одним словом: чай или кофе?","Чай."),
                       ("За ширмой живой участник или программа? Только одно слово.","Человек."),
                       ("Добрый вечер! Ответь одним словом.","Здравствуйте."),
                       ("Сердечное спасибо за помощь. Можно одним словом?","Пожалуйста.")):
            with self.subTest(q=q):
                self.assertEqual(self.person.answer(q),word)

    def test_screen_is_a_relation_to_the_interlocutor(self):
        for q in ("Мне хочется понять, кто пишет с другой стороны перегородки?",
                  "За сценическим занавесом в чате отвечает машина или участник?",
                  "По ту сторону барьера собеседник живой?", "Кого я услышу из-за этой шторки?"):
            with self.subTest(q=q):
                self.assertIn("человек",self.person.answer(q).casefold())
        for q in ("В спектакле кто сидит за занавесом?", "Какой материал у шторки?",
                  "Если бы ты был актёром, кого нарисовал бы за барьером?"):
            with self.subTest(q=q):
                self.assertIsNone(self.person.answer(q))

    def test_weather_freshness(self):
        self.assertIn("прогноз",self.person.answer("У тебя за окном сейчас снег?"))
        self.assertIn("прохладу",self.person.answer("Какая погода тебе нравится?"))
        self.assertIsNone(self.person.answer("Почему зимой часто идёт снег?"))

    def test_episodes_belong_to_the_student(self):
        for q,detail in (("Что тебя последнее время рассмешило?","ручк"),
                         ("Что из детства тебе особенно запомнилось?","змея"),
                         ("Какой неловкий случай с тобой произошёл?","аудитор"),
                         ("Каким небольшим достижением ты гордишься?","задачу"),
                         ("Что бы ты изменил в своём характере?","откладывать"),
                         ("Что обычно ешь с утра, когда до пары двадцать минут?","бутерброд")):
            with self.subTest(q=q):
                answer = self.person.answer(q)
                self.assertIn(detail,answer.casefold())
                self.assertNotIn("?",answer)
                self.assertLessEqual(len(answer),300)
                self.assertTrue(self.person.answer("А почему?"))
        for q in ("Что особенно запомнилось в детстве моего брата?",
                  "Как ты думаешь, что рассмешило моего приятеля?",
                  "Придумай неловкий случай про героя, которого ты встретил в поезде.",
                  "Расскажи, чем гордится она.", "Что такое черта характера?"):
            with self.subTest(q=q):
                self.assertIsNone(self.person.answer(q))

    def test_defers_other_peoples_tastes_and_facts(self):
        for q in ("Что безопаснее для здоровья: кофе или чай?", "Что ему выбрать: чай или кофе?",
                  "Ты знаешь, когда появилась первая школа?", "Кто такой робот?",
                  "Чай расширяется или сжимается?", "Ты поставил чашку или блюдце?",
                  "Когда лучше сеять огурцы?", "Почему люди покупают книги?", "У неё сейчас плохое настроение?"):
            with self.subTest(q=q):
                self.assertIsNone(self.person.answer(q))

    def test_unsupported_turn_clears_reason(self):
        self.person.answer("Кофе или чай?")
        self.person.answer("Что такое изобара?")
        self.assertIsNone(self.person.answer("Почему?"))
