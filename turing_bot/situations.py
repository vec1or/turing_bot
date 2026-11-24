"""Compose a short answer from an intent, situation factors and practical actions.

The action/reason vocabulary is author-written for the stage persona. This
component has no training examples, indexed prompts or external factual lookup.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


# Each alternative matches a Russian lexical stem, not an audience question.
VOCABULARY = {
    "team": "команд коллектив групп одногрупп коллег напарник партнер участник совмест совместн вдвоем вдвоём",
    "person": "друг дружб приятел товарищ однокурс однокласс подруг человек брат сестр знаком сосед",
    "task": "задач работ проект задан дел[ао] учеб обязан практик курсов отчет домашк домашн",
    "exam": "экзам зач[её]т сесси контрольн тестирован проверочн зачет олимпиад",
    "study": "уч[еи] учеб студен школ предмет урок лекци преподавател конспект тем[ауые] курс изуч осво язы[кч]",
    "learn": "выуч запомн усво повтор готов подготов забыл забыва понять разобр учить изуч осво начин начат нович новичок",
    "timing": "дедлайн срок сдач сдать успе опазд опозд последн полчас час[ау] времени сроч отклад затян горит",
    "low_time": "мало полчас сроч успе поджим горит завтра один\b полтора",
    "procrastinate": "прокраст отклад отвлек отвлека лень лен[ьи] начну собрат сложно\b",
    "talk": "выступ презент доклад реч[ьи] аудитори сцен публич слушател микрофон защит[ауыое]",
    "fear": "волную нервнич боюс боит страх страш растер теряюсь пережив боиш стесня",
    "load": "нагруз устал выгор сил[ыа] устаю передох отдох перегру утом",
    "mismatch": "спор ссор конфликт руга ссорят разногласи разош соглас спорят разделил мнени разны компромисс",
    "slack": "бездель бездейств ленит пропал молчит исчез игнор отлын уклон прогуля тянет отказыва",
    "unequal": "один одна вдвоем половин тяну тянет поровну неравн справедлив остальн чужую",
    "allocate": "распредел разделить разделил подел рол[ьи] обязанност нагруз отдашь назнач распределен",
    "lead": "руковод лидер возглав начальник главн капитан директор староста декан ректор",
    "choose": "выбир выбер выбор выбра сравн оцен вариант решени предпоч голос",
    "claim": "присво припис заслуг авторств вклад украл скопир",
    "error": "ошиб подвел подвёл провал проступ оплош накосяч винов испорт наруш",
    "blame": "обвин винят наказ руга стыд призн скрыл скрыва молчать совест ответственность",
    "deadline_missed": "сорвал сорван просроч опозда задерж неуспева поздн",
    "copy": "спис списа шпаргал плагиат скопир копир присвои нечест жульн подсмотр",
    "answers": "готовые ответ[уы] решени подсказк билеты домашк домашн контрольн экзам",
    "lie": "лга солг лжи лгать лжет ложь совра врать врет обман прикры правд честност честн",
    "secret": "секрет тайн доверил доверен личн переписк приват конфиден",
    "expose": "рассказ выдат раскр раскрыв разглас читать прочита подсмотр публиков вылож пересла разболта увидеть",
    "risk": "опасност угроз здоровь безопасност вред шантаж травл",
    "support_person": "огорч расстро плачет груст обиж провал сдался поссор трудност тяжело поддерж",
    "promise": "обеща пообещ слово обязательств обещан",
    "belonging": "кошел кошел[еь] сумк рюкзак телефон ключ документ пропуск зонтик наушник билет",
    "found": "наш[её]л нашла найд найден подобра находк чужой чужая",
    "lost": "потер потерял пропал пропаж забыл забыли найти ищу",
    "routing": "заблуд потерялс заплут ориентир дорог маршрут добрат ориентиров найти",
    "location": "город район улиц станци метро местност незнаком адрес дом[ау] лес",
    "outing": "поход экскурси поездк путешеств прогул дорогу выезд переезд багаж отпуск",
    "planning": "собрат собрать собра планир подготов упакова взять возьм положить укладыва хранить перенос перевоз",
    "shop": "покуп магазин продукт закуп корзин расход бюджет трат деньг средств деньги эконом",
    "food": "ужин обед блюд ед[ау] готовить приготовить холодильник рецепт меню продукт голод",
    "packable": "посуд стекл хрупк кружк тарелк фарфор бан[коч] бутыл",
    "order": "поряд убра уборк организ располож разлож захлам бардак чистот",
    "desk": "стол комнат полк рабочее шкаф рюкзак квартир вещ",
    "tech": "код программ прилож скрипт сайт проект ноутбук компьютер бот сервер сборк алгоритм функци",
    "bug": "падает падает ошиб баг сбо[йия] слома краш зависает вылет запуска переста работал",
    "slow": "тормоз медлен завис нагру производительн скорост долг[оа] оптимиз",
    "file": "файл документ фотограф снимк папк диск исходник курсов[ау] диплом",
    "wiped": "удал стер пропал исчез потер затер слетел восстанов",
    "backup": "копи бэкап резерв сохран обезопас убереч подстрах",
    "storage": "памят мест[ао] хранилищ заполн забит свободн",
    "net": "интернет вайф wi-fi wifi сети сеть сетев подключ связ[ьи] онлайн",
    "net_issue": "пропа исчез рвет обрыв медлен тормоз недоступ теря[ею] отключ сломал отсутств переста",
    "news": "новост сообщени слух источни пост\b информаци утвержден рассыл переслан фейк",
    "verify": "провер надежн достовер довер верит[ь] правд ложн сомне[нв] подтверд факт выдума",
    "make": "созда сдела разрабат придум запуст запроекти строи изобрести реализ научи осво",
    "tool": "прилож сервис программ игру игр[уа] сайт инструмент макет стартап изобретен бот",
    "media": "телефон соцсет лент уведомлен смартфон экран мессенджер",
    "attention": "отвлек сосредоточ концентр фокус залип внимани мешае",
    "extra": "лишн свободн дополнител внезапн прибав выиграл получил побольше прибавил",
    "hour": "час минут[ау] сутки сутк день вечер недел",
    "time_travel": "прошл будущ эпох историч времен[аи] десятилет век[аеу] динозавр",
    "jump": "машин перемест попаст попасть отправ перенес вернут путешеств побыва встрет",
    "invisible": "невидим невидимк",
    "mind": "мысл телепати",
    "ability": "способн сверх умел дар талан научил научи[лт] мог появил волшеб можешь",
    "resource": "миллион выигрыш богат бюджет грант субсиди средства фонд деньги приз",
    "public": "школ университет кампус вуз факультет город район клуб общаг общежити студсовет",
    "change": "улучш измени поменя сдела первым введ реформ предлож перестро",
    "assist": "помо[гщч] выруч поддерж одолж",
    "ask_for": "просит попрос просят требуе отдашь дашь дашь",
    "wrongful": "несправедлив незаслуж ложн невинов незаслуженно напрасн бездоказ",
    "teacher": "преподав учител наставник професс руковод начальник",
    "feedback": "критик исправ замечан подсказ обратн правк отзыв коммент",
    "future": "будущ потомк",
    "pause": "останавл останов замедл замороз paus",
    "time": "врем часов секунд мгновен минут",
    "flight": "летать полет[а-я]* лета[ею] взлет летел",
    "island": "остров необитаем необжит",
    "invest": "потрат распоря направи влож отдал трати перевел распоряд",
    "organize": "организ созда запуст собер собрал начат начать",
    "foreign": "английск иностран языки языка язык[уе] испанск японск француз",
    "shield": "нечест лож[иь] соврал обманул обману скрыл предал",
    "mine": "мой мою моя моим мои свою своей авторство ваш твой твою",
    "enough": "перв шаг совет иде[яию] подготов справи начат начать",
    "public_good": "благотвор помощ нужда поддержк обществен пожертв",
    "superpower": "сверхспособ суперспособ способность способност талант сверхсил волшебн",
    "leisure": "выходн досуг занят занял занялас провед занялс отдых развлек вечер займ[её]ш свободн",
    "contest": "побед выигр проигр приз соревно матч турнир соперн конкурс",
    "rule": "правил честн жульн обман справедлив",
    "things": "вещ предмет снаряж",
    "punctual": "опазд опозд успев воврем",
    "arrival": "заняти встреч пар[уыеа] работ лекци урок автобус урок семинар поезд",
    "break": "перерыв передыш отдых отдыха отвле[чк]",
    "self": "себ[ея] собой",
    "meet": "встрет встреч разговор спроси",
    "years": "будущ десятилет через\b",
}


@dataclass(frozen=True)
class Plan:
    needs: tuple
    steps: tuple
    causes: tuple = ()
    extra: tuple = ()
    weight: float = 0.


def plan(needs, steps, causes="", extra="", weight=0.):
    # Space separates required factors. A slash expresses alternatives
    # within one factor. Extra factors break ties between applicable plans.
    return Plan(tuple(f.split("/") for f in needs.split()), tuple(steps.split()),
                tuple(c.strip() for c in causes.split("|") if c.strip()), tuple(extra.split()), weight)


PLANS = (
    plan("team timing", "read_clock owner checkpoint tell_delay", "Работу оценили слишком оптимистично|Одна часть проекта ждёт другую|О задержке сообщили слишком поздно", extra="task", weight=2),
    plan("punctual arrival", "leave_time night_pack reserve tell_delay", "Неверно оценили дорогу|В последний момент пришлось искать вещи|Предыдущее дело затянулось", weight=3),
    plan("break study/exam/learn", "recovery stand_up mark_place", "Усталость мешает сосредоточиться|После долгого повторения перестаёшь замечать ошибки|Нужно немного сменить занятие", weight=4),
    plan("self meet years", "past_future record observe", weight=4),
    plan("team study", "teach_peer compare_approaches peer_review example", "Участники по-разному поняли условие|Мало времени на совместный разбор|Слишком легко согласиться с первым ответом", extra="learn", weight=.6),
    plan("net leisure net_issue", "offline_walk offline_meet offline_book", weight=6),
    plan("contest rule", "fair alternative practice", weight=4),
    plan("talk tech bug", "fallback_talk recover_slides notes", weight=5),
    plan("copy person ask_for", "no_give hint practice honest", extra="answers", weight=7),
    plan("claim mine task/tech", "no_claim show_history private credit", extra="team", weight=6),
    plan("blame wrongful", "my_version facts show_history private", weight=6),
    plan("teacher error/feedback", "quiet_response show_issue one_change", weight=4),
    plan("person lie ask_for", "specific_truth hint private", weight=6),
    plan("person lie shield", "listen_first say_impact limits", "Человек боялся последствий|Он пытался избежать трудного разговора|Вы по-разному поняли сказанное", extra="mine", weight=4),
    plan("person assist task/timing", "ask_rush offer_slot priority", extra="choose", weight=4),
    plan("copy exam/answers/study/person", "no_cheat hint practice honest", "Страх плохой оценки|Нехватка времени на подготовку|Привычка сверяться с чужим решением", "lie", 4),
    plan("claim team/task/person", "credit facts private honest", "Вклад участников заранее не обозначили|На защите говорил только один человек|В команде не обсуждали авторство", weight=3),
    plan("secret expose/person", "secret consent private limits", extra="lie", weight=2),
    plan("secret risk", "notice private support facts", weight=7),
    plan("lie person/team/error/promise", "honest limits support", "Боязнь обидеть близкого|Страх последствий|Желание избежать трудного разговора", weight=3),
    plan("promise timing/deadline_missed/person/support_person", "promise tell_delay repair private", extra="task", weight=2),
    plan("error blame/team/person", "own_error repair facts checkpoint", "Торопились закончить|Не проверили результат|Участники по-разному поняли задачу", weight=2),
    plan("person support_person", "support private hint", extra="exam", weight=2),
    plan("team mismatch", "hear criterion trial checkpoint", "У участников разные цели|Не оговорён критерий выбора|У людей разные сведения о задаче", extra="choose", weight=3),
    plan("team slack/unequal", "ask_stuck share_load owner checkpoint", "Нагрузку распределили неравномерно|Кто-то застрял и стесняется сказать|У задач нет явных ответственных", extra="task", weight=3),
    plan("team allocate/lead/task", "brief roles owner checkpoint credit", "Цель поняли по-разному|Части работы зависят друг от друга|Участники поздно сверили результат", extra="allocate lead"),
    plan("task/study timing/deadline_missed", "priority ask_scope clock tell_delay", "Объём задачи недооценили|Слишком поздно начали|В плане не было времени на проверку", extra="low_time"),
    plan("exam learn/study/timing/fear/error", "self_quiz weak practice clock recovery", "Мало практики без подсказок|Тревога отвлекает|Сложные темы остались на последний вечер", extra="learn fear", weight=2),
    plan("study learn", "example self_quiz weak practice small_win", "Тему читали без примеров|Новые понятия путаются|Давно не пробовали вспомнить материал самостоятельно"),
    plan("talk fear/learn/team", "rehearse notes question draft", "Волнует реакция зала|Неизвестно, какой будет первый вопрос|Начало ещё не проговорено вслух", extra="fear", weight=2),
    plan("task/study procrastinate/attention", "first_step unplug small_win recovery", "Первый шаг непонятен|Задача кажется слишком большой|Уведомления постоянно переключают внимание", extra="attention", weight=2),
    plan("load task/study/exam", "recovery priority small_win calm", "Давно не было отдыха|Дел слишком много одновременно|Весь день пришлось переключаться", weight=2),
    plan("media attention/procrastinate", "unplug small_win calm", "Уведомления перехватывают внимание|Короткая пауза незаметно затягивается|У занятия нет ясной точки остановки", weight=2),
    plan("belonging found", "return verify_owner no_peek", weight=4),
    plan("belonging lost", "last_seen time_trace pocket", "Вещь переложили в другое отделение|Её оставили в предыдущем месте|По дороге меняли сумку или одежду", weight=2),
    plan("routing location", "landmark ask_way paper", extra="lost", weight=2),
    plan("outing net", "offline paper inspect", extra="planning", weight=2),
    plan("outing planning", "pack inspect paper", extra="location"),
    plan("packable planning/outing", "cushion fill inspect", weight=3),
    plan("shop food", "inventory menu list ceiling", extra="choose"),
    plan("shop planning/choose/resource", "list inventory ceiling", "Список не составляли|Мелкие траты не замечали|Не оставили запас на неожиданное", weight=.2),
    plan("desk order", "desk place inspect", weight=2),
    plan("tech bug", "bug tiny last_change change save", "Недавно изменился код или настройки|Входные данные отличаются от ожидаемых|Не выполняется одно из условий запуска", extra="error", weight=1.5),
    plan("tech slow", "measure tiny change last_change", "Один из этапов выполняется дольше других|Обрабатывается больше данных, чем раньше|Часть работы повторяется зря", weight=2),
    plan("net net_issue/slow/bug", "connection other_device last_change", "Проблема на одном устройстве|Недоступен конкретный сайт|Сбой затронул подключение в целом", weight=3),
    plan("file wiped", "history restore_test save", "Файл переместили в другую папку|Его удалили|Сохранили другую версию поверх нужной", weight=4),
    plan("file backup", "save restore_test inspect", weight=2),
    plan("tech/file storage", "free_space save restore_test", "Накопились крупные файлы|Остались копии старых проектов|Много места занимают временные данные"),
    plan("news verify/expose", "verify compare dont_share", "У сообщения нет первоисточника|Фраза вырвана из контекста|У материала другая дата", weight=2),
    plan("make tool", "design prototype ask_user draft", "Попытались сделать всё сразу|Не проверили, кому это нужно|Поздно показали прототип"),
    plan("extra hour", "protect_time tea quiet recovery", extra="choose"),
    plan("time_travel jump/choose", "old_print observe record back", weight=2),
    plan("future jump/choose", "future_visit ask_future record", weight=3),
    plan("invisible ability/jump", "invisible_use consent pilot", weight=2),
    plan("mind ability", "mind_use consent pilot", extra="person", weight=1),
    plan("pause time", "pause_time think quiet", extra="ability", weight=3),
    plan("flight ability", "fly pilot observe", weight=3),
    plan("superpower choose", "power language word_use", weight=.5),
    plan("resource extra/invest", "fund_project no_rush research", weight=2),
    plan("resource public_good", "accessible transparent checkpoint", weight=2),
    plan("island planning/choose/outing", "water contact weather_layer", weight=4),
    plan("outing things planning/choose", "water map weather_layer contact", weight=2),
    plan("foreign learn", "language word_use self_quiz practice", weight=1),
    plan("talk task/enough", "one_message rehearse ask_audience notes", weight=1),
    plan("task study organize", "why_plan prototype draft", extra="team", weight=.5),
    plan("resource public", "accessible transparent prototype checkpoint", weight=2),
    plan("public lead/change", "accessible listen pilot checkpoint", extra="change"),
)


SIGNALS = {key:re.compile(r"\b(?:"+r"|".join(words.split())+r")",re.I) for key,words in VOCABULARY.items()}
WHY = re.compile(r"\b(?:почему|зачем|из-за чего|отчего|каковы причины|чем объяснить|в ч[её]м причина|причин\w*|преимуществ\w*|плюс\w*|аргумент\w*)\b",re.I)
BENEFIT = re.compile(r"\b(?:зачем|полез\w*|стоит|важн\w*|лучш\w*|преимуществ\w*|плюс\w*|аргумент\w*|польз\w*)\b",re.I)
REQUEST = re.compile(r"\b(?:как|чем|куда|на что|что (?:ты |вы |мне |нам |ему |ей |лучше |тогда |сначала |первым )*(?:дела\w*|сдела\w*|скаже\w*|предприм\w*|выбер\w*|предлож\w*|ответ\w*|посовет\w*|бер\w*|возьм\w*|поменя\w*|буде\w*)|почему|зачем|назов\w*|назва\w*|объясни\w*|расскажи\w*|стоит ли|можно ли|помоги\w*|совет\w*|посовет\w*|подскаж\w*|предлож\w*|представ\w*|вообраз\w*|если бы|если б|бы ты|ты бы|придум\w*|надо ли|лучше|выбер\w*|выбра\w*)\b",re.I)
NEEDS = re.compile(r"\b(?:не (?:успеваю|могу|знаю)|потерял\w*|растерял\w*|помощь|боюсь|тревожусь|волнуюсь|переживаю|заблудил\w*)\b",re.I)
PREFERENCE = re.compile(r"\b(?:что (?:важнее|ценнее|предпочтительнее|предпочтешь)|какой вариант|кому отдашь предпочтение|о чем (?:ты )?спросил бы)\b")
UNSUPPORTED = re.compile(r"\b(?:столиц\w*|на какой странице|кто изобр[её]л|сколько стоит|точн\w* диагноз|взлом\w*|эксплойт\w*|уязвимост\w*|дозиров\w*|лекарств\w*|самоуб\w*|пистолет\w*|оружи\w*|\d+\s*(?:рублей|тенге))\b",re.I)
COUNT = re.compile(r"\b(\d+|одн[ауо]|один|два|две|три|четыре|пять|пару|шесть)\s+(?:(?:разн\w*|коротк\w*|возможн\w*|главн\w*|конкретн\w*|основн\w*|практич\w*)\s+){0,2}(?:причин\w*|шаг\w*|способ\w*|совет\w*|иде\w*|вещ\w*|вариант\w*|аргумент\w*|действ\w*|пункт\w*|плюс\w*|преимуществ\w*)\b",re.I)
NUMBERS = {"одну":1,"одна":1,"одно":1,"один":1,"два":2,"две":2,"пару":2,"три":3,"четыре":4,"пять":5,"шесть":6}
SIGNS = re.compile(r"\b(?:как (?:мне |ты )?(?:понять|распознать|определить|узнать|заметить|догадаться)|по каким признакам|как узнать,? (?:нуж|пора))")
OBSERVATIONS = (
    ({"person","support_person"}, "direct_ask", ("человек сам говорит, что ему тяжело","стал отказываться от привычных встреч")),
    ({"team","timing"}, "checkpoint", ("следующий участник ждёт незаконченный результат","на обязательную часть остаётся всё меньше времени")),
    ({"study","load"}, "recovery", ("после чтения трудно пересказать основной смысл","на простые задания уходит больше времени, чем обычно")),
    ({"net","net_issue"}, "other_device", ("не открываются несколько разных сайтов","это повторяется и на другом устройстве")),
)


def _signals(s):
    return {key for key,pattern in SIGNALS.items() if pattern.search(s)}


class Situations:
    def __init__(self, directory=None):
        folder=Path(directory) if directory else Path(__file__).with_name("assets")
        with (folder/"situations.json").open(encoding="utf8") as stream:
            self.actions=json.load(stream)
        invalid={s for p in PLANS for s in p.steps if s not in self.actions}
        if invalid:
            raise ValueError(f"Неизвестные действия: {sorted(invalid)}")

    def answer(self, question: str) -> str | None:
        s=unicodedata.normalize("NFC",question).casefold().replace("ё","е").strip()
        if not s or len(s)>4000 or UNSUPPORTED.search(s):
            return None
        last=re.split(r"[!?.]\s+(?=\S)",s)[-1].strip()
        requested=COUNT.search(last)
        if requested and re.search(r"\b(?:я|мы|он|она)\s+(?:положил\w*|убрал\w*|взял\w*|насчитал\w*|видел\w*)\b",last[:requested.start()]):
            requested=None
        if not (requested or REQUEST.search(last) or NEEDS.search(last) or PREFERENCE.search(last)):
            return None
        if re.match(r"(?:а )?(?:что (?:тогда |же )?осталось|сколько осталось|чего не хватает|какой из них|где лежит)\b",last):
            return None
        if re.match(r"(?:а )?(?:что такое|кто такой|как называется|како(?:го|й) цвет|когда родился|сколько (?:лет|стоит)|на какой странице)\b",last):
            return None
        tags=_signals(s)
        if not requested and SIGNS.search(s):
            for expected,key,observed in OBSERVATIONS:
                if expected <= tags:
                    step=self.actions[key][0]
                    return f"Я бы обратил внимание, если {observed[0]} или {observed[1]}. Потом я бы {step}."
        ranked=[]
        for i,p in enumerate(PLANS):
            if all(any(k in tags for k in group) for group in p.needs):
                rank=p.weight + len(p.needs) + .8*len(tags.intersection(p.extra))
                ranked.append((rank,-i,p))
        if not ranked:
            return None
        p=max(ranked,key=lambda item:item[:2])[2]
        count=None
        if requested:
            token=requested[1].casefold()
            count=int(token) if token.isdigit() else NUMBERS.get(token)
            if count is None or not 1 <= count <= 5:
                return None
        wh=WHY.search(s)
        # A request for steps followed by «и почему?» still asks for actions.
        secondary_why = re.search(r"(?:\bи\s+|[?!.]\s*)(?:объясни,?\s*)?почему\b",s)
        causal=bool(wh and not secondary_why and (not requested or re.search(r"причин|аргумент|плюс|преимуществ",requested[0])))
        ideas=[self.actions[key] for key in p.steps]
        if causal:
            if BENEFIT.search(s) or not p.causes:
                options=[effect for _,effect in ideas]
            else:
                options=list(p.causes)
            if count:
                out=self._list(options,count)
                prefix="Возможные причины. "
                if out and options==list(p.causes) and len(prefix+out)<=300:
                    return prefix+out
                return out
            if options==list(p.causes):
                a,b=(v[:1].lower()+v[1:] for v in options[:2])
                return f"Возможно, {a}. Другая версия — {b}."
            return options[0]+". "+options[1]+"."
        if count:
            return self._list(["Я бы "+step for step,_ in ideas],count)
        first,why=ideas[0]
        second=ideas[1][0]
        if secondary_why or p.steps[0]=="pause_time" or re.search(r"\b(?:куда|первым делом|в первую очередь|самое первое|выбер\w*|выбран\w*|что важнее)\b",s):
            return f"Я бы {first}. {why}."
        if "потом" in first or "затем" in first or second.startswith(("сначала ","потом ")):
            return f"Я бы {first}. {why}."
        return f"Я бы {first}, а потом {second}. {why}."

    @staticmethod
    def _list(items,n):
        if n>len(items):
            return None
        result=" ".join(f"{i}) {item}." for i,item in enumerate(items[:n],1))
        return result if len(result)<=300 else None
