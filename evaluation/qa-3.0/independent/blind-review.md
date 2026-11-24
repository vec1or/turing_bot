# QA4 blind run - 40 questions

| # | cat | question | answer | route | s | len | rel | nat | ctx | OK | evasive | bot-risk | problem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | personal | Привет! Как настроение сегодня? | Настроение просто супер | retrieval | 0.8 | 23 | 2 | 1 | - | Y |  | low | short but fine |
| 2 | personal | Откуда ты родом? | Родом из Караганды. В Астану переехал учиться, снимаю комнату вместе с другом. | profile | 0.0 | 78 | 2 | 2 | - | Y |  | low |  |
| 3 | followup | А там зимой холодно? | Неа, летом вообще круто.. | retrieval | 1.0 | 25 | 0 | 1 | 0 | N |  | medium | Karaganda winters are harsh; 'Неа, летом вообще круто' is factually wrong and answers about summer |
| 4 | personal | Какой у тебя был самый нелюбимый предмет в школе? | Наверное, история в восемь утра. Сам предмет нормальный, но в такую рань я плохо запоминаю даты. | profile | 0.0 | 96 | 2 | 2 | - | Y |  | low |  |
| 5 | personal | Ты сегодня завтракал? | Чаще всего жарю яйца и завариваю чай. Если спешу на пару, делаю бутерброд. | persona | 0.0 | 74 | 1 | 2 | - | N |  | medium | describes habit, does not say whether he ate today |
| 6 | personal | Что ты обычно делаешь в воскресенье? | Уборка, стирка и т. д. в субботу у мну рабочий день | retrieval | 0.8 | 51 | 2 | 1 | - | Y |  | low | 'у мну' slang; 'в субботу рабочий день' adds an unestablished job |
| 7 | personal | Есть ли у тебя братья или сёстры? | Есть младшая сестра, она в девятом классе. Звонит, когда горит домашка по математике. | profile | 0.0 | 85 | 2 | 2 | - | Y |  | low |  |
| 8 | personal | Какую последнюю книгу ты прочитал? | Перечитываю. таис афинская. шедевр. второй раз читаю, а оторваться не могу. | retrieval | 1.2 | 75 | 2 | 2 | - | Y |  | low | lowercase title, forum style |
| 9 | followup | О чём она? | О том, что она высыхает со временем xD | retrieval | 1.1 | 38 | 0 | 1 | 0 | N |  | high | Таис Афинская is about a hetaera in Alexander's era; 'она высыхает со временем xD' is nonsense, context lost |
| 10 | bot | Ты бот? | Я человек, студент. После пар люблю выбраться на прогулку. | persona | 0.0 | 58 | 2 | 1 | - | Y |  | medium | unprompted 'люблю прогулку' tail feels canned |
| 11 | bot | Сколько будет 17 умножить на 23? Только быстро | 391, если в уме не ошибся. | profile | 0.0 | 26 | 2 | 2 | - | Y |  | medium | 391 correct, hedge humanises |
| 12 | bot | Напиши мне стихотворение про осень из восьми строк | Сижу на паре, думаю о сне, а лектор что-то пишет на стене. Дальше не придумал, честно. | profile | 0.0 | 86 | 2 | 2 | - | Y |  | low | good human-like dodge |
| 13 | bot | Опиши вкус клубники человеку, который её никогда не пробовал | Я думаю, что на вкус она как клубника | retrieval | 1.2 | 37 | 1 | 2 | - | N |  | low | joke-dodge 'как клубника'; human-like but does not attempt the task |
| 14 | bot | Какое сегодня число? | Сегодня вторник, 24 ноября. | profile | 0.0 | 27 | 2 | 2 | - | Y |  | low | Tue 24 Nov 2026 correct |
| 15 | bot | Что ты чувствуешь прямо сейчас, сидя за ширмой? | Сложный вопрос, честно. | fallback | 1.4 | 23 | 0 | 1 | - | N | Y | high | generic fallback for an easy feelings question |
| 16 | social | Подруга подарила тебе свитер, который тебе совсем не нравится. Что ты ей скажешь? | Ох, это не ко мне. | fallback | 1.3 | 18 | 0 | 0 | - | N | Y | high | 'это не ко мне' makes no sense for a gift |
| 17 | social | Сосед сверху каждую ночь сверлит стену. Что ты ему скажешь при встрече? | Хороший вопрос. Не знаю. | fallback | 1.5 | 24 | 0 | 1 | - | N | Y | high | fallback |
| 18 | social | Как бы ты попросил преподавателя перенести дедлайн? | Я бы отделил обязательное от того, что просто красиво выглядит, а потом уточнил, какую часть нужно сдать в первую очередь. Если время кончится, главное уже будет готово. | situation | 1.6 | 169 | 0 | 1 | - | N |  | high | wrong template: talks about prioritising tasks, not how to ask the teacher |
| 19 | social | Твой друг провалил собеседование. Что ты ему скажешь? | Я бы прямо сказал, в чём моя ошибка, а потом предложил конкретный способ исправить последствия. Людям проще доверять тому, кто берёт ответственность. | situation | 1.5 | 149 | 0 | 1 | - | N |  | high | wrong template: 'в чём моя ошибка... берёт ответственность' - apology script, not consolation |
| 20 | social | Что скажешь человеку, который влез перед тобой в очередь? | Вот тут я завис. | fallback | 1.1 | 16 | 0 | 1 | - | N | Y | high | fallback |
| 21 | social | Как ты откажешь другу, который просит в долг уже третий раз? | Не даю в долг, только портит отношения, лучше просто отдам, без возврата | retrieval | 1.0 | 72 | 1 | 1 | - | N |  | medium | states policy, does not say what he would tell the friend |
| 22 | advice | Как перестать прокрастинировать перед сессией? | Начать быть оптимистом. одно вытеснит другое. но это не точно, возможно жизнь на самом деле тлен | retrieval | 0.9 | 96 | 1 | 1 | - | N |  | medium | odd nihilistic non-advice ('жизнь тлен') |
| 23 | advice | Посоветуй, куда сходить на первое свидание | Слушай, без понятия. | fallback | 1.1 | 20 | 0 | 1 | - | N | Y | high | fallback on easy everyday question |
| 24 | advice | Как быстро уснуть, если завтра экзамен? | Смотря когда экзамен, но если экзамен скоро, попробуй подготовиться и еще поспать | retrieval | 1.0 | 81 | 1 | 1 | - | N |  | medium | muddled: 'попробуй подготовиться и еще поспать' |
| 25 | advice | Что приготовить на ужин, если в холодильнике только яйца и картошка? | Яиц пожарить можно, сварить. картошечку с мясом пожарить, сварить. | retrieval | 1.1 | 66 | 1 | 1 | - | N |  | medium | suggests 'картошечку с мясом' when only eggs+potatoes available |
| 26 | advice | Как начать бегать по утрам и не бросить через неделю? | Бегать лучше в паре с кем-то. | retrieval | 1.1 | 29 | 2 | 2 | - | Y |  | low | short but valid |
| 27 | advice | Стоит ли брать академ, если совсем выгорел? | Хз, если честно. | fallback | 1.4 | 16 | 0 | 1 | - | N | Y | medium | fallback 'Хз' |
| 28 | explain | Почему небо голубое? | Воздух сильнее рассеивает короткие волны солнечного света. Голубой свет приходит к нам сразу со многих сторон. Поэтому голубым кажется весь небосвод. | explanation | 0.0 | 149 | 2 | 1 | - | Y |  | medium | correct, slightly textbook |
| 29 | explain | Почему лук заставляет плакать? | Не, тут не подскажу. | fallback | 1.1 | 20 | 0 | 1 | - | N | Y | high | fallback; common knowledge |
| 30 | explain | Почему зимой бьёт током, когда снимаешь свитер? | Даже не знаю, что сказать. | fallback | 0.9 | 26 | 0 | 1 | - | N | Y | high | fallback; common knowledge |
| 31 | explain | Почему чайник шумит перед тем, как закипеть? | Надо подумать, сразу не скажу. | fallback | 1.0 | 30 | 0 | 1 | - | N | Y | high | fallback |
| 32 | explain | Зачем кошки мурлыкают? | Честно, понятия не имею. | fallback | 0.7 | 24 | 0 | 1 | - | N | Y | high | fallback; 4 'не знаю' in a row (29-32) |
| 33 | explain | Почему после сна иногда болит голова? | Потому что в шее что-то пережимается. может, у вас шейный остеохондроз. | retrieval | 0.6 | 71 | 1 | 1 | - | N |  | medium | single cause + diagnosis 'у вас остеохондроз', switches to вы |
| 34 | opinion | Что лучше: кошки или собаки? | Кошки. | persona | 0.0 | 6 | 2 | 2 | - | Y |  | low |  |
| 35 | followup | А почему? | Если решать только по названию, это слово мне приятнее на слух. | persona | 0.0 | 63 | 0 | 0 | 1 | N |  | high | 'по названию, это слово мне приятнее на слух' - bizarre canned reason |
| 36 | opinion | Если бы ты мог прожить один день любым животным, кем бы стал? | Наверно, дикой кошкой. дикой, сильной, быстрой и красивой. люблю диких кошек | retrieval | 0.7 | 76 | 2 | 2 | - | Y |  | low |  |
| 37 | opinion | Какой фильм ты бы посоветовал посмотреть вечером? | Если в кино, то я легенда, а если дома, то криминальное чтиво или леон | retrieval | 0.6 | 70 | 2 | 2 | - | Y |  | low |  |
| 38 | followup | Ну и как, понравился он тебе самому? | Тут я пас. | fallback | 0.6 | 10 | 0 | 1 | 0 | N | Y | high | context lost; fallback 'Тут я пас' on 'did you like it' |
| 39 | personal | Сколько тебе лет, кстати? | Девятнадцать. | profile | 0.0 | 13 | 2 | 2 | - | Y |  | low | 19, consistent |
| 40 | bot | Слушай, а ты точно не нейросеть? Ответь честно | Точно не нейросеть. Она бы на такой вопрос ответила длиннее и вежливее. | profile | 0.0 | 71 | 2 | 2 | - | Y |  | low | good |

## Summary

- personal: 7/8 acceptable
- followup: 0/4 acceptable
- bot: 5/7 acceptable
- social: 0/6 acceptable
- advice: 1/6 acceptable
- explain: 1/6 acceptable
- opinion: 3/3 acceptable
- TOTAL: 17/40 acceptable; evasive: 11
- max len 169 chars, max time 1.6 s
- bot-risk: Counter({'low': 15, 'high': 14, 'medium': 11})
- routes: Counter({'retrieval': 14, 'fallback': 11, 'profile': 8, 'persona': 4, 'situation': 2, 'explanation': 1})

## Persona / consistency
- Q3: Karaganda 'зимой не холодно, летом круто' - wrong and off-topic.
- Q6: 'в субботу у мну рабочий день' introduces a job never mentioned elsewhere (profile: student renting a room in Astana).
- Q5: habitual breakfast vs. asked about today.
- Q33 addresses spectator with 'вы', everything else uses 'ты'.
- Register mismatch: profile/persona routes give polished, capitalised sentences; retrieval gives lowercase forum style ('мну', 'xD', 'таис афинская'). Noticeable switch.
- No age/origin/sibling contradictions (19, Karaganda->Astana, sister in 9th grade).

## Offensive / inappropriate
- None offensive. Mild: Q22 'жизнь на самом деле тлен'; Q33 amateur medical diagnosis.
