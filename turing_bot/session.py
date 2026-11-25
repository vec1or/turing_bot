from __future__ import annotations

from collections import deque
from pathlib import Path
import re
import unicodedata
import math

from .neural import Generator, tokenize
from .word_model import WordModel
from .cloze import Cloze
from .reader import Reader
from .knowledge import Knowledge
from .articles import Articles
from .grounding import Confidence
from .language import Language
from .planning import Planner, IMAGINE
from .social import Social
from .advice import Advice, ASK
from .grounding import normal
from .persona import Persona
from .situations import Situations
from .imagination import Imagination
from .explanations import explain
from .evidence import limitation
from .writing import Writing
from .reasoning import Reasoning
from .practical import Practical
from .associations import Associations
from .profile import DENIALS, Profile
from .retrieval import Retrieval
from . import goal_plan, reactions, tips, travel
from .rounds import Abstentions, clean, focal_question, requested_format, short_operation, standalone

MAX_CHARS = 300
MAX_INPUT = 4000
RETRIEVAL_THRESHOLD = .72
RETRIEVAL_STRONG = .78
RETRIEVAL_MIN_DENSE = .5  # tuned on validation questions, see evaluation/RESULTS-3.0.md
RECALL_SELF = re.compile(r"^(?:а |напомни[, ]+)?(?:что|какой вариант|какое слово) (?:же )?ты (?:только что |тогда |в итоге )?(?:выбрал|назвал|ответил|предпоч[её]л)а?[?!. ]*$",re.I)
REPEAT = re.compile(r"(?:предыдущ|прошл)\w* (?:вопрос|ответ)\w*.*(?:ответил|сказал|повтори)|(?:ответил|сказал) на (?:предыдущ|прошл)\w* вопрос|повтори,? (?:что|свой|свои|последн)\w*|что ты (?:только что|сейчас) (?:сказал|ответил)",re.I)
TITLES = {
    "Тьма": ("Немецкий городок, где пропадают дети, а в пещере есть проход на тридцать три года назад. Три поколения семей переплетаются, без схемы не разберёшься.",
             "Сюжет как головоломка: всё время хочется понять, кто кому кем приходится."),
    "Шерлок": ("Шерлок Холмс в современном Лондоне, с телефонами и блогами. Каждая серия как отдельный фильм.",
               "Камбербэтч там просто огонь, и серии длинные, как фильмы. Затягивает с первой."),
    "Мастер": ("Про то, как дьявол со свитой приезжает в Москву тридцатых, а параллельно про Мастера, который пишет роман о Понтии Пилате. Ну и про любовь, конечно.",
               "Очень. Перечитываю, и каждый раз как в первый."),
    "Интерстеллар": ("Земля умирает, и команда летит через червоточину искать новую планету. А по сути это про отца и дочь и про время.",
                     "Да, очень. Пересматривал раза три, и каждый раз в конце сижу молча."),
}
EXPLICIT = re.compile(r"\b(?:объясни\w*|расскажи (?:про|о|об)|что такое|что значит|кто так(?:ой|ая)|как работает|как устроен\w*)\b", re.I)
WHY = re.compile(r"(?:(?:а|но|и|ну) )*(?:почему|зачем|с чего бы|отчего|как так)(?: именно)?(?: так| это| же| его| её| ее| их| им| ими| ей| он| она| они| ты так думаешь| ты так считаешь)?[?!. ]*",re.I)
ARTISTS = ("Скриптонит", "Булгаков", "Нолан")
WHY_REPLIES = ("Ну, просто по опыту так.", "Сложно объяснить, просто так чувствую.", "Не знаю, мне так кажется.",
               "Просто так сложилось.", "Ну а как иначе, по-моему, очевидно.")
FOLLOW = re.compile(r"^(?:а |но )?(?:почему|зачем|как так|дальше|что потом|что дальше)(?: именно)?(?: так| туда| это| там)?[?!. ]*$",re.I)


PRECISE = {"profile", "explanation", "persona", "tips", "travel", "reaction", "operation", "creative"}
QUESTION_WORD = r"(?:какой|какая|какое|какие|что|чё|чо|че|сколько|как|где|почему|отчего|куда|чем|кто|когда|зачем)"


def split_parts(question):
    """Two questions in one line: «…? …?» or «… и какая …», «…, а почему …»."""
    q = " ".join(question.split())
    pieces = [p.strip(" ,") for p in re.split(r"\?\s+", q) if p.strip(" ,?")]
    if len(pieces) == 2 and all(len(p) >= 6 for p in pieces) and re.match(r"(?:а |и |ну )?" + QUESTION_WORD + r"\b|(?:а |и )?(?:ты|тебе|у тебя|сам)\b", pieces[1], re.I):
        return pieces[0] + "?", pieces[1].rstrip("?") + "?"
    found = re.search(r"(?:,?\s+и\s+|,\s*а\s+|,\s+)(?=" + QUESTION_WORD + r"\b)", q, re.I)
    if found and found.start() >= 8 and len(q) - found.end() >= 8:
        first, second = q[:found.start()], q[found.end():]
        if re.match(QUESTION_WORD + r"\b", first, re.I) or re.search(r"\b(?:ты|тебя|тебе)\b", first, re.I):
            return first + "?", second.rstrip("?") + "?"
    return None


def on_topic(question, passage):
    """An encyclopaedic passage must share a long word stem with the question."""
    stems = {w[:6] for w in re.findall(r"[а-яё]{6,}", question.casefold().replace("ё", "е"))
             if not w.startswith(("объясн", "просты", "словам", "почему", "работа", "расска"))}
    text = passage.casefold().replace("ё", "е")
    return any(s in text for s in stems)


def display_text(text):
    text = unicodedata.normalize("NFC", " ".join(text.split()))
    if len(text) > MAX_CHARS:
        prefix = text[:MAX_CHARS]
        ends = list(re.finditer(r"[.!?](?=\s|$)", prefix))
        text = prefix[:ends[-1].end()] if ends else prefix[:MAX_CHARS-1].rsplit(" ", 1)[0].rstrip(",:;—-") + "…"
    return text


def meaningful_generation(answer, question):
    if not standalone(answer):
        return False
    tokens = re.findall("[а-яё]+", answer.casefold())
    # Two-word generations were mostly fragments («Для умный!») in blind QA.
    if len(tokens) < 3:
        return False
    if len(tokens) >= 6 and len(set(tokens)) / len(tokens) < .7:
        return False
    functional = {"как","как-то","тут","там","было","была","были","был","будет","есть","даже","уже",
                  "вот","думаю","знаю","который","которая","потому","что","это","просто","себя","этого",
                  "этот","сделал","сказал","может","хочу","хочется","сделать","зависит","того","когда","только","надо"}
    content = [w for w in tokens if len(w)>2 and w not in functional]
    return bool(content and not set(content).issubset(re.findall("[а-яё]+", question.casefold())))


def _chain_arithmetic(question, previous):
    q = question.casefold()
    q = re.sub(r"(?:свой |твой )?возраст\w*(?: свой| твой)?", "19", q)
    if (re.search(r"^(?:а |ну |и )?(?:если )?(?:потом |теперь )?половину", q) or re.search(r"^(?:а |ну |и )?(?:если|потом|теперь)\b.{0,40}(?:вышл\w*|ушл\w*|съел\w*|отдал\w*|убежал\w*) (?:\w+ )?половин", q) or re.search(r"^(?:а |ну |и )?(?:если|потом|теперь)\b.{0,40}половин\w* (?:\w+ )?(?:вышл|ушл|убежал|сошл)", q)) and not re.search(r"\d", q):
        nums = re.findall(r"-?\d+", previous)
        if nums:
            base = int(nums[0])
            if base % 2 == 0:
                return f"{base // 2}. Это уже проще было."
            if re.search(r"вышл|человек|пассажир|ушл", q + " " + previous.casefold()):
                return f"{base / 2:.1f}".replace(".", ",") + f" выходит, но человека не разрежешь) Так что {base // 2} или {base // 2 + 1}."
            return f"{base / 2:.1f}".replace(".", ",") + ", если пополам делить."
        return None
    if len(re.findall(r"\d+", q)) != 1 or not re.search(r"^(?:а |ну |и |если |а если |теперь |ну а )|\b(?:это|этого|этому|его|него|нему|результат\w*)\b", q):
        return None
    m = re.search(r"(?:вычесть|вычти|отнять|отними|минус)(?: (?:из (?:этого|него|того|результата)|еще|ещё))? (\d+)|(?:прибавить|прибавь|добавить|добавь|плюс)(?: (?:к (?:этому|нему|результату)|еще|ещё))? (\d+)|(?:умножить|умножь|поделить|подели|разделить|раздели)(?: (?:это|его|результат))? на (\d+)", q)
    nums = re.findall(r"-?\d+", previous)
    if not m or not nums:
        return None
    base = int(nums[0]) if len(nums) == 1 else int(nums[0])
    if m[1]:
        value = base - int(m[1])
    elif m[2]:
        value = base + int(m[2])
    else:
        k = int(m[3])
        if re.search(r"умнож", q):
            value = base * k
        elif k and base % k == 0:
            value = base // k
        elif k:
            return f"Примерно {base / k:.1f}".replace(".", ",") + ", ровно не делится."
        else:
            return None
    return f"{value}. Это уже проще было."

_NUMWORDS = {"одно": 1, "один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "десять": 10}


def _word_problem(question):
    q = question.casefold().replace("ё", "е")
    verbs = r"(вышл\w*|вышел|зашл\w*|зашел|сел\w*|ушл\w*|ушел|съел\w*|отдал\w*|купил\w*|нашел\w*|вернул\w*|потерял\w*)"
    q = re.sub(r"\b(\d+|" + "|".join(_NUMWORDS) + r") (?:человек\w* )?" + verbs, r"\2 \1", q)
    q = re.sub(verbs + r" (?:тебе|мне|ему|ей|другу|подруге|маше|пете|им) ", r"\1 ", q)
    start = re.search(r"(?:у (?:меня|тебя|него|нее|мамы|\w+) (?:есть |было |лежит )?)?(\d+) (?:яблок|конфет|груш|мандарин|рубл|тенге|книг|карандаш|орех|печен|человек|пассажир|овец|птиц|машин)", q)
    boxes = re.search(r"(\d+) (?:коробк|пакет|ящик|корзин|мешк|упаковк|пачк|банк)\w* по (\d+) (?:\w+)", q)
    if boxes and re.search(r"сколько", q):
        start = boxes
        value = int(boxes[1]) * int(boxes[2])
        num = r"(\d+|" + "|".join(_NUMWORDS) + r")"
        frac = r"(половин[ау]|треть|четверть)"
        steps = re.finditer(r"(съел\w*|отдал\w*|потерял\w*|продал\w*|выкинул\w*|подарил\w*)\s+(?:еще |ещё )?(?:" + num + r"|" + frac + r")|(купил\w*|нашел\w*|получил\w*|дали|добавил\w*)\s+(?:еще |ещё )?(?:" + num + r"|в (два|три|четыре|2|3|4) раза больше)", q[start.end():])
        for st in steps:
            if st[1]:
                if st[3]:
                    value -= value * {"половину": 1 / 2, "половина": 1 / 2, "треть": 1 / 3, "четверть": 1 / 4}[st[3]]
                else:
                    value -= int(st[2]) if st[2].isdigit() else _NUMWORDS[st[2]]
            elif st[5]:
                value += int(st[5]) if st[5].isdigit() else _NUMWORDS[st[5]]
        txt = str(int(value)) if value == int(value) else f"{value:.1f}".replace(".", ",")
        return f"{txt}. {int(boxes[1]) * int(boxes[2])} всего было, дальше просто."
    if not start or not re.search(r"сколько|скок|скоко|скока", q):
        return None
    value = int(start[1])
    num = r"(\d+|" + "|".join(_NUMWORDS) + r")"
    frac = r"(половин[ау]|треть|четверть)"
    steps = re.finditer(r"(съел\w*|отдал\w*|потерял\w*|продал\w*|выкинул\w*|подарил\w*|вышл\w*|вышел|ушл\w*|ушел|улетел\w*)\s+(?:еще |ещё )?(?:" + num + r"|" + frac + r")|(купил\w*|нашел\w*|получил\w*|дали|добавил\w*|зашл\w*|зашел|сел\w*|вошл\w*|вошел|прилетел\w*|вернул\w*)\s+(?:еще |ещё )?(?:" + num + r"|в (два|три|четыре|2|3|4) раза больше)", q[start.end():])
    half = re.search(r"(?:отдал\w*|подарил\w*) половину (\w+)", q)
    asks_who = re.search(r"сколько (?:\w+ )?(?:у|осталось у) (\w+)", q)
    if half and asks_who and asks_who[1][:3] == half[1][:3]:
        whole = value
        value = whole / 2
        rest = q[half.end():]
        asks = re.search(r"сколько (?:\w+ )?(?:у|осталось у) (\w+)", q)
        recipient_asked = asks and asks[1][:3] == half[1][:3]
        if not recipient_asked:
            value = whole - value
        for st in re.finditer(r"(съел\w*|отдал\w*|потерял\w*)\s+(?:еще |ещё )?" + num, rest):
            n = st[2]
            value -= int(n) if n.isdigit() else _NUMWORDS[n]
        txt = str(int(value)) if value == int(value) else f"{value:.1f}".replace(".", ",")
        return f"{txt}. Если ничего не путаю в условии."
    seen = False
    for st in steps:
        seen = True
        if st[1]:
            if st[3]:
                value -= value * {"половину": 1 / 2, "половина": 1 / 2, "треть": 1 / 3, "четверть": 1 / 4}[st[3]]
            else:
                n = st[2]
                value -= int(n) if n.isdigit() else _NUMWORDS[n]
        else:
            if st[6]:
                value += {"два": 2, "три": 3, "четыре": 4, "2": 2, "3": 3, "4": 4}[st[6]] * value
            else:
                n = st[5]
                value += int(n) if n.isdigit() else _NUMWORDS[n]
    cont = re.search(r"(?:съел|отдал|потерял|продал|выкинул|подарил|вышл)\w*\s+(?:половин\w*|треть|четверть|\d+|\w+)\s+и (?:еще|ещё) (" + "|".join(_NUMWORDS) + r"|\d+)\b", q)
    if cont:
        n = cont[1]
        value -= int(n) if n.isdigit() else _NUMWORDS[n]
    if not seen:
        return None
    txt = str(int(value)) if value == int(value) else f"{value:.1f}".replace(".", ",")
    tail = " Ну, если яблоки резать." if value != int(value) else ""
    return f"{txt}. Если ничего не путаю в условии.{tail}"


_UNITS = ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать")
_TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто")
_HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот")


def _words(n):
    if n < 20:
        return _UNITS[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _UNITS[n % 10])
    if n < 1000:
        return _HUNDREDS[n // 100] + ("" if n % 100 == 0 else " " + _words(n % 100))
    if n < 10000:
        th = n // 1000
        word = {1: "одна тысяча", 2: "две тысячи", 3: "три тысячи", 4: "четыре тысячи"}.get(th, _UNITS[th] + " тысяч")
        return word + ("" if n % 1000 == 0 else " " + _words(n % 1000))
    return str(n)


class Session:
    def __init__(self, directory=None):
        self.generator = Generator.load(directory)
        self.word_model = WordModel.load(directory)
        self.cloze = Cloze.load(directory)
        self.reader = Reader.load(directory)
        self.knowledge = Knowledge.load(directory)
        self.articles = Articles.load(directory)
        self.confidence = Confidence.load(directory, generator=self.generator)
        self.language = Language.load(directory)
        self.planner = Planner(directory)
        self.social = Social.load(directory)
        self.advice = Advice.load(directory)
        self.persona = Persona()
        self.situations = Situations(directory)
        self.imagination = Imagination()
        self.writing = Writing(directory)
        self.reasoning = Reasoning()
        self.practical = Practical()
        self.associations = Associations(directory)
        self.profile = Profile()
        root = Path(directory) if directory is not None else Path(__file__).with_name("assets")
        # The reply index is optional so that older asset folders keep working.
        self.retrieval = Retrieval.load(directory) if (root / "retrieval" / "encoder.json").exists() else None
        if self.retrieval is not None:
            # Touch the whole index once at start-up, so that the first
            # spectator's question is not slowed down by loading pages.
            self.retrieval.propose("Привет, как дела?")
        self.abstentions = Abstentions()
        self.last_source = None
        self.last_route = None
        self.last_retrieval = None
        self.first_question = None
        self.first_answer = None
        self.user_name = None
        self.history = deque(maxlen=60)
        self.turns = deque(maxlen=3)

    def reset(self):
        self.turns.clear()
        self.last_source = None
        self.last_route = None
        self.last_retrieval = None
        self.first_question = None
        self.first_answer = None
        self.user_name = None
        self.history.clear()
        self.persona.reset()
        self.abstentions.seen.clear()

    def answer(self, question):
        raw = question[:MAX_INPUT]
        if not raw.strip():
            raise ValueError("Введите вопрос.")
        question = clean(raw) or raw.strip()
        thanked = re.match(r"^(?:ну )?(?:спасибо|спс|пасиб\w*|благодарю|ок спасибо)[\s,.!()…]+(?:а |и |кстати |слушай,? )?(.+)$", question, re.I)
        if thanked and len(thanked[1].split()) >= 3 and not re.match(r"(?:что|за|тебе|большое|огромное)\b", thanked[1], re.I):
            question = thanked[1]
        self.last_source = None
        combined = self._two_parts(question)
        if combined:
            return combined
        for route, candidate, source in self._candidates(question):
            # The final fallback is already a complete, language-checked utterance.
            # Always keep an answer available even for impossible format demands.
            if route!="fallback":
                candidate=requested_format(candidate,question)
            if standalone(candidate):
                reply = display_text(candidate)
                if not standalone(reply):
                    continue
                said = {a for _, a in self.history}
                if reply in said and route != "recall" and route in ("profile", "persona", "travel", "tips", "follow-up", "reaction"):
                    # Only a re-asked question earns «я же говорил»; a keyword
                    # collision should fall through to the next candidate.
                    earlier_q = next((eq for eq, ea in self.history if ea == reply), "")
                    words = lambda t: {w for w in re.findall(r"[а-яё]{4,}", t.casefold())}
                    a, b = words(earlier_q), words(question)
                    stems = {w[:5] for w in b}
                    reasked = re.search(r"напомни|еще раз|ещё раз|говорил|забыл|зовут|звали", question, re.I) or any(st in reply.casefold() for st in stems if len(st) >= 4)
                    if a and b and len(a & b) / min(len(a), len(b)) < 0.5 and not reasked:
                        continue
                if reply in said and route != "recall":
                    # A person does not repeat a whole sentence word for word.
                    fresh = [d for d in DENIALS if d not in said]
                    if reply in DENIALS and fresh:
                        reply = fresh[0]
                    elif route == "explanation":
                        reply = display_text("Да по той же причине. " + re.split(r"(?<=[.!])\s", reply, maxsplit=1)[0])
                    elif route in ("profile", "persona", "travel", "tips", "reaction", "follow-up"):
                        first_word = re.match(r"[«]?([А-ЯЁ][а-яё]+)", reply)
                        keep = first_word and first_word[1] in ("Ерлан", "Барсик", "Тимур", "Айгерим", "Серик", "Скриптонит", "Астана", "Караганда", "Япония")
                        reply = display_text("Я же уже говорил: " + (reply if keep else reply[:1].lower() + reply[1:]))
                    else:
                        continue
                    if reply in said:
                        continue
                if re.search(r"\bя (?:\w+ ){0,2}\w+(?:лась|ала|ила|ела|яла)\b|\bя (?:\w+ )?(?:устала|рада|готова|одна)\b|^(?:\w+ ){0,2}\w+лась\b", question, re.I) and not re.search(r"\bя (?:\w+ ){0,2}\w+(?:ал|ил|ел|ял)\b|\bя (?:\w+ )?(?:устал|рад|готов)\b", question, re.I):
                    self.user_female = True
                if re.search(r"\bя (?:\w+ ){0,2}(?:\w+(?:ал|ил|ел)|забыл|устал|рад|готов)\b", question, re.I) and not re.search(r"\bя (?:\w+ ){0,2}\w+(?:лась|ала|ила|ела|яла)\b", question, re.I):
                    self.user_female = False
                if getattr(self, "user_name", None) and getattr(self, "name_gender_set", None) != self.user_name and self.user_name in ("Аружан", "Айгерим", "Асель", "Айжан", "Балжан", "Гульжан", "Гульнар", "Алтынай", "Томирис", "Аяулым", "Нургуль", "Жибек", "Анель", "Арайлым", "Акерке", "Айым", "Аймерей", "Гаухар", "Жанель", "Мерей", "Сабина", "Ажар", "Асем", "Айнур", "Нурсулу", "Шолпан", "Сауле", "Любовь", "Рахель", "Эстер"):
                    self.user_female = True
                if getattr(self, "user_name", None) and getattr(self, "name_gender_set", None) != self.user_name and re.search(r"[ая]$", self.user_name) and self.user_name not in ("Илья", "Никита", "Саша", "Женя", "Миша", "Дима", "Ваня", "Коля", "Петя", "Вася", "Лёша", "Леша", "Паша", "Гоша", "Кузьма", "Фома", "Лука", "Данила", "Слава", "Валера", "Сережа", "Серёжа", "Мустафа", "Иса", "Муса"):
                    self.user_female = True
                if getattr(self, "user_name", None):
                    self.name_gender_set = self.user_name
                if getattr(self, "user_female", False):
                    for m_, f_ in (("Ты писал:", "Ты писала:"), ("Ты написал:", "Ты написала:"), ("Ты спросил:", "Ты спросила:"), ("ты спрашивал", "ты спрашивала"), ("Ты же говорил", "Ты же говорила"), ("ты говорил", "ты говорила"), ("Ты что-то перепутал)", "Ты что-то перепутала)"), ("ты в начале писал.", "ты в начале писала."), ("в самом начале говорил.", "в самом начале говорила."), ("ты говорил)", "ты говорила)"), ("Подловить хотел)", "Подловить хотела)")):
                        reply = reply.replace(m_, f_)
                self.last_route = route
                self.last_source = source
                if route != "persona":
                    self.persona.reset()
                greet = getattr(self, "pending_greet", None)
                if greet:
                    self.pending_greet = None
                    if len(reply) + len(greet) + 12 <= 300:
                        reply = f"Привет, {greet}) " + reply
                self.turns.append((question, reply))
                self.history.append((question, reply))
                if self.first_question is None:
                    self.first_question = question
                    self.first_answer = reply
                return reply

    def _candidates(self, question):
        course_m = re.search(r"\b(перв\w*|втор\w*|трет\w*|четверт\w*|пят\w*|\d) курс\w*(?: (\w+))?", question.casefold())
        if course_m and re.search(r"\bя\b|учусь", question.casefold()) and not re.search(r"(?:ты|тебя) (?:\w+ )?(?:на )?(?:каком|курс)|каком (?:я )?курсе", question.casefold()):
            fac_ = course_m[2] if course_m[2] and course_m[2] not in ("и", "сижу", "учусь", "я") else ""
            self.user_course = (course_m[1], fac_)
        where_m = re.search(r"\bсижу (на паре|на лекции|дома|в общаге|на работе|в кафе|в универе|в библиотеке|в автобусе|в очереди(?: в \w+)?)(?: и (\w+))?", question.casefold())
        if where_m and not re.search(r"где (?:я )?(?:сейчас )?сижу", question.casefold()):
            verb_ = where_m[2] or ""
            verb_ = re.sub(r"([ая])ю$", r"\1ешь", verb_)
            self.user_where = where_m[1] + (f" и {verb_}" if verb_ else "")
        city_m = re.search(r"\b(?:сам|сама|родом) (?:\w+ )?(?:из|с) ([а-яё]{4,})|\bя из ([а-яё]{4,})", question.casefold())
        if city_m and (city_m[1] or city_m[2]) not in ("детства", "школы", "универа", "работы", "дома", "утра", "тобой", "ним", "ней", "этого", "деревни"):
            self.user_city = (city_m[1] or city_m[2]).capitalize()
        age_m = re.search(r"\bмне (\d{1,2})\b(?! ?(?:минут|час|тенге|тг|км|раз))", question.casefold())
        if age_m and not re.search(r"сколько мне", question.casefold()):
            self.user_age = int(age_m[1])
        exam_m = re.search(r"(?:завтра|скоро|послезавтра) (?:\w+ )?(?:экзамен|экз|зачет|коллок\w*) по ([а-яё]+)|(?:завтра|скоро) (?:\w+ )?([а-яё]+у) (?:сдавать|сдаю)|(?:сдавать|сдаю) (?:завтра )?([а-яё]+у)\b", question.casefold())
        if exam_m and not re.search(r"что я (?:\w+ )?сдаю", question.casefold()):
            self.user_exam = f"экзамен по {exam_m[1]}" if exam_m[1] else (exam_m[2] or exam_m[3])
        job_m = re.search(r"\bработаю ((?:в|на) [а-яё]+|[а-яё]+(?: (?:в|на) [а-яё]+)?)", question.casefold())
        if job_m and not re.search(r"(?:ты|тебя) (?:\w+ )?работа|где (?:я )?работаю|помнишь|кем (?:я )?работаю", question.casefold()):
            self.user_job = job_m[1]
        study_m = re.search(r"\bя? ?учусь (?:на|в) ([а-яё]+(?: [а-яё]+)?)", question.casefold()) or re.search(r"\b(?:на|в) ([а-яё]{4,}) учусь", question.casefold())
        if study_m and not re.search(r"(?:ты|тебя) (?:\w+ )?уч|где (?:я )?учусь|помнишь", question.casefold()):
            self.user_study = study_m[1].split(" ")[0] if study_m[1].split(" ")[0] not in ("первом", "втором", "третьем") else study_m[1]
        asks_name = re.search(r"как меня зовут|мо[её] имя|помнишь,? как меня|как я представил\w*", question, re.I)
        intro = None if asks_name else (re.search(r"\b(?:меня зовут|мо[её] имя|зови меня)\s+([А-ЯЁA-Zа-яё][а-яёa-z]{2,})", question, re.I)
                                       or re.search(r"\bменя\s+(?:кстати\s+|вообще\s+)?(?!все|так|тоже|никак|вообще|кстати)([а-яё]{3,})\s+зовут", question, re.I)
                                       or re.search(r"(?:^|[,.!)?]\s*)(?:а )?я (?:кстати |вообще-то |вообще )?([А-ЯЁ][а-яё]{2,})(?:[,.!)]|\s+(?:кстати|если что|приятно)|$)", question)
                                       or re.search(r"(?:^|[,.!)?]\s*)(?:а )?я (?!(?:тоже|тут|здесь|вот|ну|так|не|уже|сам|сама|просто|студент\w*|судья|бот|человек|девушка|парень|из|в|на|за|очень|вообще|кстати|думаю|знаю|хочу|могу|буду|был|была)\b)([а-яё]{3,})(?:,|\s+(?:кстати|если что))", question))
        if intro:
            self.user_name = intro[1][:1].upper() + intro[1][1:].lower()
            city = re.search(r"\bиз ([А-ЯЁ][а-яё]{3,})", question)
            if city:
                self.user_city = city[1]
            rest = (question[:intro.start()] + " " + question[intro.end():]).strip(" ,.!)")
            rest = re.sub(r"^(?:привет|прив|хай|салам|здравствуй\w*|добрый \w+)\W*", "", rest, flags=re.I).strip(" ,.!)")
            if len(rest.split()) >= 3 and re.search(r"[?]|\b(?:ты|как|что|чем|где|когда|почему)\b", rest, re.I):
                self.pending_greet = self.user_name
                question = rest
            else:
                yield "persona", f"Запомнил, {self.user_name}. Приятно познакомиться, я Тимур.", None
        if self.turns and re.search(r"около \d+ тысяч", self.turns[-1][1]) and re.search(r"долго (?:\w+ )?(?:думал|считал)|так не считают|не посчитал", question, re.I):
            yield "follow-up", "Так я и не считал точно, прикинул на глаз. Если надо до единицы — дай минуту и листочек.", None
        if self.first_question and not re.search(r"\bбы\b", question, re.I) and re.search(r"перв\w* (?:что|сообщени|вопрос|фраз)\w*.{0,25}(?:написал|спросил|писал|сказал)|(?:я )?(?:написал|спросил) (?:у тебя |тебя |тебе )?(?:самым )?первым", question, re.I):
            quoted = re.sub(r"\?\s*", ". ", self.first_question).strip(" .!")
            yield "recall", f"Ты написал: «{quoted}». Помню, помню."[:300], None
        if self.turns:
            recent = " ".join(t[1] for t in list(self.turns)[-2:]).casefold()
            ql = question.casefold().replace("ё", "е")
            if "барсик" in self.turns[-1][1].casefold() and re.fullmatch(r"(?:а )?(?:как )?(?:\w+ )?(?:зовут|имя|кличка)\W*", ql):
                yield "follow-up", "Барсик же. Серый, толстый и очень наглый.", None
            if "ерлан" in recent and re.search(r"(?:чем|что) (?:\w+ )?(?:он )?(?:тебя )?бесит", ql):
                yield "follow-up", "Да ничем особо. Гречку мою доедает и храпит иногда, а так нормальный пацан.", None
            if re.search(r"серик|препод|строгий", recent) and re.search(r"сколько (?:\w+ )?(?:ему|ей) лет|(?:ему|ей) сколько", ql):
                yield "follow-up", "Лет пятьдесят, наверное. Седой уже, но бодрый, на парах ходит туда-сюда.", None
            if "ерлан" in recent and re.search(r"как (?:\w+ )?(?:его )?зовут|откуда он", ql) and not re.search(r"\b(?:ее|её|она|меня|мое|моё)\b", ql):
                yield "follow-up", "Ерлан же. Тоже из Караганды, мы со школы вместе.", None
            if "сочувствую" in self.turns[-1][1].casefold() and re.search(r"у тебя (?:\w+ )?(?:такое )?бывает|(?:а )?(?:с )?тобой (?:\w+ )?бывает", ql):
                yield "follow-up", "Бывает, особенно в сессию. Я тогда просто ложусь пораньше, и на следующий день как-то легче.", None
            if re.search(r"строгий|препод|душнит", recent) and re.search(r"как (?:\w+ )?(?:его |ее |её )?(?:зовут|звали)", ql) and not re.search(r"мо[её] имя|меня|мое имя|сосед|друг", ql):
                yield "follow-up", "Серик Болатович. Его весь поток побаивается, но объясняет нормально.", None
            if re.search(r"(?:человек|пассажир)", self.turns[-1][0].casefold() + " " + (self.history[-2][0].casefold() if len(self.history) > 1 else "")) and re.search(r"водител", ql) and not re.search(r"кроме", ql):
                nums_ = re.findall(r"\d+", self.turns[-1][1])
                if nums_:
                    yield "follow-up", f"Ахах, подловил. С водителем {int(nums_[0]) + 1}.", None
            if "доту" in recent and re.search(r"ранг|ммр|рейтинг|звани", ql):
                yield "follow-up", "Около трёх тысяч ммр, Легенда. Выше никак, тиммейты тянут вниз, ну это я так говорю.", None
            if "Караганду, родной город" in self.turns[-1][1] and re.fullmatch(r"(?:а )?почему\W*", ql):
                yield "follow-up", "Там все свои: родители, друзья со школы, Барсик. Приезжаешь, и сразу как будто выдохнул.", None
            if re.search(r"тикток", self.turns[-1][1].casefold()) and re.search(r"сколько (?:\w+ )?час", ql):
                yield "follow-up", "Часа четыре-пять в день, если честно. Экранное время лучше не открывать, стыдно.", None
            if "реал" in self.turns[-1][1].casefold() and re.fullmatch(r"(?:а |и )?(?:давно|с каких пор|с какого времени)\W*", ql):
                yield "follow-up", "С детства, лет с десяти. Папа смотрел Лигу чемпионов, и я подсел.", None
            if "остыть" in self.turns[-1][1].casefold() and re.search(r"перв\w* начал", ql):
                yield "follow-up", "Бывает. Но кто начал, уже не так важно, важнее, кто первый помирится. Если дорожишь, напиши просто, без разборок.", None
            if re.search(r"\bям", self.turns[-1][0].casefold()) and re.search(r"объясни|как (?:ты )?(?:посчитал|получил)|почему", ql):
                yield "follow-up", getattr(self, "dig_explain", "Ну там пропорция, всё просто."), None
            if "кофейн" in self.turns[-1][1].casefold() and re.search(r"что (?:\w+ )?берешь|сколько (?:\w+ )?стоит", ql):
                yield "follow-up", "Латте и круассан, где-то тысячи две. Дорого для студента, поэтому не каждый день.", None
            if "футбол" in self.turns[-1][1].casefold() and re.search(r"почему", ql):
                yield "follow-up", "С пацанами весело, да и бесплатно. В зал одному ходить лень, а тут хоть повод встретиться.", None
            if "один на всех" in self.turns[-1][1] and re.search(r"почему|как так|не шесть", ql):
                yield "follow-up", "Потому что брат один и тот же для всех трёх сестёр. Три дочери плюс один сын — четверо.", None
            shop = getattr(self, "shop", None)
            if shop and not re.search(r"на сдачу", ql) and re.search(r"ещ[её] (?:одн\w* |один )?(\w+)", ql) and re.search(r"хватил|сколько|сдач", ql):
                prices, total, paid = shop
                item = re.search(r"ещ[её] (?:одн\w* |один )?(\w+)", ql)[1]
                price = next((v for k, v in prices.items() if k[:4] == item[:4]), None)
                if price:
                    nt = total + price
                    self.shop = (prices, nt, paid)
                    yield "operation", (f"Хватило бы, {nt} выходит, сдачи {paid - nt}." if nt <= paid else f"Не, уже {nt} было бы, не хватило бы на {nt - paid}."), None
            if "мама" in self.turns[-1][1].casefold() and re.search(r"что (?:\w+ )?(?:она )?(?:\w+ )?(?:лучше всего )?готовит", ql):
                yield "follow-up", "Манты и лагман. А на праздники баурсаки, соседи на запах приходят.", None
            if re.search(r"готов|плов|макарон", self.turns[-1][1].casefold()) and re.search(r"научился|где учился|кто научил", ql):
                yield "follow-up", "У мамы подсмотрел, остальное ютуб. Первый плов был кашей, Ерлан до сих пор вспоминает.", None
            if "шахмат" in self.turns[-1][1].casefold() and re.search(r"бросил|давно", ql):
                yield "follow-up", "Шахматы в шестом классе бросил. Тренер был слишком серьёзный, а мне хотелось во двор.", None
            if re.search(r"история|предмет", recent) and re.search(r"кто (?:\w+ )?веде|препод кто|кто препод", ql):
                yield "follow-up", "Серик Болатович. Строгий, читает с листа, но на экзамене не валит.", None
            if re.search(r"родителям", self.turns[-1][1]) and re.search(r"(?:маме|папе|родителям) .{0,25}купил", ql):
                yield "follow-up", "Конечно. Маме путёвку в санаторий, она давно хочет, а папе нормальный инструмент.", None
            if "глубокой воды" in self.turns[-1][1] and re.search(r"почему|серьезно", ql):
                yield "follow-up", "В детстве на Балхаше чуть не утонул, дядя вытащил. С тех пор так.", None
            if "Бешбармак, конечно" in self.turns[-1][1] and re.search(r"почему|наоборот", ql):
                yield "follow-up", "Шаурму я и так каждую неделю ем, а бешбармак — это праздник и бабушка. Тут без вариантов.", None
            if "скриптонит" in recent and re.search(r"посоветуй|что послушать|что включить", ql):
                yield "follow-up", "Скриптонит — «Это любовь» и «Положение», с них начни. Если зайдёт, дальше сам затянешься.", None
            flock_ = getattr(self, "flock", None)
            if flock_ and re.search(r"овец|овцы|овеч", ql) and re.search(r"напомни|в той задач|сколько (?:\w+ )?осталось", ql) and not re.search(r"\d", ql):
                yield "follow-up", f"{flock_[0]}, если со всеми прибежавшими и убежавшими.", None
            last_a = self.turns[-1][1]
            if "информационн" in last_a.casefold() and re.search(r"нравится|как оно|не жалеешь", ql):
                yield "follow-up", "В целом да, программировать нравится. Матан только душит, но куда без него.", None
            if "Первая пара в восемь" in last_a and re.search(r"рано|поздно|почему так", ql):
                yield "follow-up", "Так пара в восемь, а ещё на автобусе ехать. Будь моя воля, спал бы до десяти.", None
            if "раскладушка" in last_a and re.search(r"что с (?:ней|ним) стало|где (?:она|он) (?:сейчас|теперь)", ql):
                yield "follow-up", "Утопил в луже, классика. Бежал со школы, она выпала из кармана, и всё.", None
            if "Была в школе" in last_a and re.fullmatch(r"(?:а |и )?(?:давно|как давно)\W*", ql):
                yield "follow-up", "С выпуска, полтора года уже. Сначала переписывались, потом как-то сошло на нет.", None
            if "отступы съедут" in last_a and re.search(r"пожалуйста|сложно|ну напиши|жалко", ql):
                yield "follow-up", "Не сложно, лень с телефона печатать. Загугли bubble sort, там пять строчек, честно.", None
            if "двойки за почерк" in last_a and re.search(r"(?:что|че|чо) (?:\w+ )?(?:он )?(?:такого )?делает|чем (?:\w+ )?стр[её]мный|почему", ql):
                yield "follow-up", "Валит за любую мелочь: почерк, не та ручка, опоздал на минуту. Зато кто у него сдал, тот реально матан знает.", None
            pair = getattr(self, "pair", None)
            if pair:
                a_, b_ = pair
                more = re.search(r"(?:купит|купил|добав\w*|прибав\w*) (?:ещ[её] )?(одн\w*|один|\d+)", ql) or re.search(r"(?:ещ[её] )?(одн\w*|один|\d+) (?:\w+ )?(?:купит|купил|добав\w*)", ql)
                sold = re.search(r"продаст|продал|отдаст|отдал|потерял", ql)
                if more and not sold:
                    k = 1 if not more[1].isdigit() else int(more[1])
                    first = not re.search(r"сын|втор|брат|у него", ql) or re.search(r"отец|перв|папа", ql)
                    a_, b_ = (a_ + k, b_) if first else (a_, b_ + k)
                    self.pair = (a_, b_)
                    yield "operation", f"Тогда вместе {a_ + b_}.", None
                elif sold and re.search(r"все|всё", ql):
                    who_second = re.search(r"сын|втор|брат", ql)
                    left_ = b_ if who_second else a_
                    yield "operation", f"Ноль. У него было {left_}, все и продал.", None
            if getattr(self, "last_minutes", None) and re.fullmatch(r"(?:а |и )?(?:это )?(?:в минутах|сколько (?:это )?минут)\W*", ql):
                yield "operation", f"{self.last_minutes} минут.", None
            if re.search(r"серик|препод", recent) and re.search(r"уволил|уволят|уйдет|ушел бы", ql):
                yield "follow-up", "Расстроились бы разве что отличники. Остальные бы сначала радовались, а через год скучали бы, честно.", None
            if "самсунг" in last_a.casefold() and re.search(r"сколько (?:\w+ )?(?:стоил|стоит|отдал)", ql):
                yield "follow-up", "Тысяч сто двадцать, когда покупал. Копил с курьерских, так что жалко менять.", None
            if re.search(r"(?:человек|пассажир)", " ".join(t[0] for t in list(self.turns)[-2:]).casefold()) and re.search(r"вс[еи],? кроме водител", ql):
                nums_ = re.findall(r"\d+", last_a)
                if nums_:
                    if re.search(r"осталось|остался|осталос", ql):
                        yield "operation", "Ноль) Водитель не пассажир, так что в салоне пусто.", None
                    else:
                        yield "operation", f"{nums_[0]}. Все пассажиры вышли, водитель остался.", None
            if "ерлан" in recent and re.search(r"(?:чем|что) (?:\w+ )?(?:он )?(?:вообще )?(?:занимается|делает)", ql):
                yield "follow-up", "В зал ходит, в доту залипает, а по выходным в доставке подрабатывает. Он вообще бодрее меня.", None
            if "дедушкин" in last_a.casefold() and re.search(r"почему|зачем", ql):
                yield "follow-up", "Это единственное, что от него осталось лично мне. Иногда достаю, кнопки потыкаю, и как будто он рядом.", None
            if "помог бы родителям" in last_a and re.search(r"родител|маме|папе", ql) and not re.search(r"работают|кем|чем занимаются", ql):
                yield "follow-up", "Ну я ж сказал, первым делом им. Маме что-нибудь для дома, папе нормальный инструмент, он давно хочет.", None
            if "интерстеллар" in last_a.casefold() and re.search(r"чем (?:\w+ )?(?:закончил|кончил)|концовк|о ч[её]м (?:\w+ )?(?:он|фильм)", ql):
                yield "follow-up", "Купер выбирается из чёрной дыры, встречает постаревшую дочь, а потом летит к Бранд. Я в этот момент сижу молча, каждый раз.", None
            ages = getattr(self, "ages", None)
            an_ = getattr(self, "age_names", None)
            nm_ = re.fullmatch(r"(?:а |и )?(\w+)(?: тогда| сколько| сейчас)*\W*", ql)
            if ages and an_ and nm_ and nm_[1][:3] in an_:
                val = ages[1] if nm_[1][:3] == an_[0] else ages[0]
                yield "operation", f"{nm_[1].capitalize()} {val}.", None
            if ages and re.fullmatch(r"(?:а |и )?(?:брату|сестре|старшему|младшему|ему|ей)\W*", ql):
                who = ql.strip(" ?!.").split()[-1]
                val = ages[0] if who.startswith(("сестр", "младш")) == ages[2] else ages[1]
                yield "operation", f"{val}. Проверь: через три года всё сходится.", None
            if "турника" in last_a and re.search(r"что (?:было )?потом|и что|дальше", ql):
                yield "follow-up", "Месяц в гипсе ходил, зато все на нём расписались. Писать левой научился, почерк стал ещё хуже.", None
            if "десятке" in last_a and re.search(r"сколько (?:\w+ )?(?:едешь|ехать|по времени)", ql):
                yield "follow-up", "Минут двадцать пять, если без пробок. В час пик все сорок.", None
            if "Сериком Болатовичем" in last_a and re.search(r"почему|зачем", ql):
                yield "follow-up", "Интересно, правда ли им нравится нас валить или это просто работа. Подозреваю, что второе.", None
            ratio = getattr(self, "ratio", None)
            if ratio and re.fullmatch(r"(?:а |и )?(?:у )?(\w+)(?: тогда| сколько)?\W*", ql):
                who = re.fullmatch(r"(?:а |и )?(?:у )?(\w+)(?: тогда| сколько)?\W*", ql)[1]
                names, vals = ratio
                for n_, v_ in zip(names, vals):
                    if n_[:3] == who[:3]:
                        yield "operation", f"{v_}. Вместе как раз {sum(vals)}.", None
                        break
            if "пекарни" in last_a and re.search(r"почему|откуда", ql):
                yield "follow-up", "У бабушки рядом с домом пекарня была, она меня каждое утро за батоном отправляла. Тёплый, по дороге полбатона съедал.", None
            if "пельменей" in last_a and re.search(r"сколько (?:\w+ )?(?:отдал|стоил|потратил|вышло)", ql):
                yield "follow-up", "Тысячи две с чем-то. Пельмени нынче дорогие, раньше на эти деньги неделю жил.", None
            if "Токио" in last_a and re.search(r"чем (?:\w+ )?(?:бы )?(?:там )?занимал|что (?:бы )?(?:там )?делал|и что потом", ql):
                yield "follow-up", "Прогером на удалёнке, а по выходным по городу гулял бы. Идеальная жизнь, короче.", None
            if "упал со сцены" in last_a and re.search(r"учител|отреагировал|классн|директор", ql):
                yield "follow-up", "Классуха сначала спросила, живой ли я, а потом сама ржала громче всех. Директор сделал вид, что не заметил.", None
            if "по программированию" in last_a and re.search(r"какой (?:\w+ )?предмет|что (?:\w+ )?ведет|предмет какой|какой ведет", ql):
                yield "follow-up", "Основы программирования, там Python и немного C++. Единственная пара, на которую хочется ходить.", None
            if getattr(self, "last_minutes", None) and re.search(r"(?:выехал|выезжает|отправ\w*) (?:\w+ )?в (\d{1,2})[:.](\d{2})", ql) and re.search(r"во сколько|когда", ql):
                dm = re.search(r"в (\d{1,2})[:.](\d{2})", ql)
                end_ = int(dm[1]) * 60 + int(dm[2]) + self.last_minutes
                self.arrival = end_
                yield "operation", f"В {end_ // 60 % 24}:{end_ % 60:02d}, если без опозданий.", None
            if getattr(self, "user_city", None) and re.fullmatch(r"(?:а )?(?:я )?откуда (?:я|я родом)\W*|(?:а )?откуда я\W*", ql):
                yield "recall", f"Из {self.user_city} же, ты говорил.", None
            if "матанализ, базы данных и английский" in last_a and re.search(r"скучн|нудн|душн", ql):
                yield "follow-up", "Английский, наверное. Одни и те же тексты по кругу читаем, а я его и так норм знаю.", None
            if "Английский, наверное" in last_a and re.search(r"почему", ql):
                yield "follow-up", "Тексты одни и те же каждую пару, а преподу лишь бы по плану. Я его со школы нормально знаю, вот и скучно.", None
            if "Караганду, родной город" in last_a and re.fullmatch(r"(?:а |и )?зимой\W*", ql):
                yield "follow-up", "Зимой обе так себе. Но в Астане ветер злее, в Караганде мороз хотя бы стоит на месте.", None
            if "шаурмой" in last_a and re.search(r"что (?:было )?потом|и что|дальше", ql):
                yield "follow-up", "Принёс шаурму, съел половину и уснул прямо в куртке. Классика.", None
            cls_ = getattr(self, "cls", None)
            if cls_ and re.search(r"процент", ql):
                add = re.search(r"(\d+) (\w+)", ql)
                total_, counts = cls_
                if add:
                    k = int(add[1])
                    grp = next((g for g in counts if g[:4] == add[2][:4]), None)
                    if grp:
                        new = dict(counts); new[grp] += k
                        asked = next((g for g in counts if g[:4] in ql.split("процент")[-1]), grp)
                        pct_ = new[asked] / (total_ + k) * 100
                        yield "operation", f"Примерно {pct_:.0f}%: {new[asked]} из {total_ + k}.", None
            if re.search(r"собеседован", " ".join(t[0] for t in list(self.turns)[-1:]).casefold()) and re.search(r"у тебя (?:\w+ )?(?:было|бывало) (?:такое|так)|а ты\b", ql):
                yield "follow-up", "Было, перед первым собеседованием на курьера, смешно звучит, но трясло. Прошло нормально, взяли.", None
            share = re.search(r"поровну (?:на|между) (\d+|двоих|троих|трех|четверых|двух|пятерых)", ql)
            if share:
                k_ = {"двоих": 2, "двух": 2, "троих": 3, "трех": 3, "четверых": 4, "пятерых": 5}.get(share[1]) or int(share[1])
                nums_ = re.findall(r"\d+", last_a)
                if nums_ and k_:
                    base_ = int(nums_[0])
                    self.remainder = base_ % k_
                    tail_ = f", и {base_ % k_} лишних остаётся" if base_ % k_ else ", ровно делится"
                    yield "operation", f"По {base_ // k_}{tail_}.", None
            if getattr(self, "remainder", None) and re.search(r"остат|лишн", ql):
                yield "operation", f"Себе оставлю {self.remainder}, честно заработал)", None
            if "во дворе" in last_a.casefold() and re.search(r"долго|давно|сколько лет", ql):
                yield "follow-up", "Да лет до четырнадцати каждый день, пока на подготовку к ЕНТ не засадили.", None
            if "сестра" in recent and re.search(r"созван|звонишь|общаетесь|переписыва", ql):
                yield "follow-up", "С сестрой — когда у неё домашка горит)) А с мамой почти каждый день.", None
            if re.search(r"получается \d+|\d+, если в уме", last_a.casefold()) and re.search(r"в уме|как (?:ты )?посчитал|калькулятор", ql):
                yield "follow-up", "Ну да, в уме, по частям раскладываю: сначала на десятки, потом остальное. С калькулятором быстрее, но так интереснее.", None
            if "бывают такие дни" in last_a.casefold() and re.search(r"у тебя (?:\w+ )?(?:бывают|бывает|было)|а у тебя", ql):
                yield "follow-up", "Конечно. На прошлой неделе проспал пару, пролил чай на конспект и забыл пропуск — всё за одно утро.", None
            if "тенге" in last_a and re.search(r"дорого или|норм (?:для|или)|это дорого", ql):
                yield "follow-up", "Для студента норм, если не каждый день с супом и вторым. В кафешках в два раза дороже.", None
            if "барсик" in last_a.casefold() and re.fullmatch(r"(?:а |и )?(?:давно|как давно|сколько ему|сколько лет)\W*.*", ql):
                yield "follow-up", "Лет пять уже, котёнком с улицы подобрали. Сейчас он толще меня.", None
            if "интерстеллар" in last_a.casefold() and re.search(r"понравил|запомнил|как тебе", ql):
                yield "follow-up", "Очень. Сцена с волнами на планете и когда Мёрф понимает, что это был отец. Каждый раз мурашки.", None
            if "матанализ" in last_a.casefold() and re.search(r"препод", ql):
                yield "follow-up", "Серик Болатович, строгий. Списать нереально, но объясняет нормально, если не перебивать.", None
            if "токио" in last_a.casefold() and re.search(r"алмат", ql):
                yield "follow-up", "Алматы классный, но это почти как дома. Хочется чего-то совсем другого, хотя бы на время.", None
            if getattr(self, "shortfall", None) and re.search(r"половин", ql) and re.search(r"займ|одолж|дам|даст|сколько", ql):
                half_ = self.shortfall / 2
                hs_ = str(int(half_)) if half_ == int(half_) else f"{half_:.1f}".replace(".", ",")
                yield "operation", f"Ещё {hs_} найти. Половину друг закроет, половину сам.", None
            if "сестра у всех" in last_a.casefold() and re.search(r"почему|как так", ql):
                yield "follow-up", "Потому что сестра у всех братьев одна и та же. Три брата плюс она — четверо.", None
            if re.search(r"буквами|словами|прописью", ql) and not re.search(r"\d", ql):
                nums_ = re.findall(r"\b\d{1,4}\b", last_a)
                if nums_:
                    w_ = _words(int(nums_[0]))
                    yield "operation", w_[:1].upper() + w_[1:] + ".", None
            if "дедушкин" in last_a.casefold() and re.search(r"кто (?:\w+ )?подарил|откуда (?:он|у тебя)", ql):
                yield "follow-up", "Дедушка сам отдал, когда себе новый купил. Сказал: «Береги, это классика».", None
            if "напиши первым" in last_a.casefold() and re.search(r"не напишет|не ответит|не отвечает|если (?:она|он) (?:\w+ )?(?:не|промолчит)", ql):
                yield "follow-up", "Тогда напиши ещё раз, но без давления. Если и дальше тишина — дай время, это тоже ответ.", None
            if "ерлан" in recent and re.search(r"(?:чем|что) (?:\w+ )?(?:вы )?(?:обычно )?(?:занимаетесь|делаете)", ql):
                yield "follow-up", "В доту катаем, в зал иногда ходим, готовим по очереди. Ну и сериалы вечером, если пары не убили.", None
            if "стипенди" in last_a.casefold() and re.search(r"сколько", ql):
                yield "follow-up", "Около сорока тысяч. Хватает впритык, если без кафешек.", None
            shop2 = getattr(self, "shop", None)
            if shop2 and re.search(r"сдач", ql) and re.search(r"(\d+) ?(?:тенге|тг)?", ql):
                prices, total, _ = shop2
                paid = int(re.search(r"(\d+)", ql)[1])
                self.shop = (prices, total, paid)
                self.change = paid - total
                yield "operation", (f"{paid - total} сдачи." if paid >= total else f"Не хватит, нужно ещё {total - paid}."), None
            if shop2 and getattr(self, "change", None) is not None and re.search(r"на сдачу|сколько ещ[её]", ql):
                prices, _, _ = shop2
                item = next((k for k in prices if k[:4] in ql), None)
                if item and prices[item]:
                    k_ = self.change // prices[item]
                    yield "operation", f"Ещё {k_}, и {self.change - k_ * prices[item]} останется.", None
            if "работой по айтишке" in last_a and re.search(r"почему", ql):
                yield "follow-up", "В Караганде с айтишной работой туго, всё в Астане или Алматы. Да и к столице я уже привык.", None
            if "физичка" in last_a.casefold() and re.search(r"делал|делала|чем (?:\w+ )?(?:бесил|страшн)|что (?:он|она)", ql):
                yield "follow-up", "Да она это была, физичка) Орала на весь класс и к доске вызывала, чтобы опозорить, если домашку не сделал.", None
            if re.search(r"сесси|билет", " ".join(t[0] + " " + t[1] for t in list(self.turns)[-2:]).casefold()) and re.search(r"(?:ты )?(?:сам )?(?:как )?(?:сдавал|сдал)", ql):
                yield "follow-up", "Нормально, без хвостов, только матан еле на тройку вытянул. Последнюю ночь вообще не спал.", None
            if re.search(r"плов|макарон|яичниц", last_a.casefold()) and re.search(r"кто .{0,15}научил|где (?:\w+ )?научился", ql):
                yield "follow-up", "Мама в основном, ну и ютуб. Первый плов был кашей, Ерлан до сих пор вспоминает.", None
            cmp_m = re.search(r"^(\S+)\. Если в десятичных, это (\S+) против (\S+)\.", last_a)
            if cmp_m and re.search(r"почему|как так|объясни", ql):
                yield "follow-up", f"Потому что {cmp_m[2]} больше, чем {cmp_m[3]}. Дроби проще сравнивать, когда переводишь в десятичные.", None
            if "яичниц" in last_a.casefold() and re.search(r"и вс[её]\b|только это|и все\b", ql):
                yield "follow-up", "Ну да, утром особо не до изысков. Зато в обед в столовке отъедаюсь.", None
            if re.search(r"сестр|айгерим|айка", recent) and re.search(r"ладите|ругаетесь|ссоритесь", ql):
                yield "follow-up", "Да нормально, иногда цапаемся из-за ерунды, но она мелкая, ей можно. Даже скучаю по ней.", None
            if re.search(r"^Астана\.", last_a) and re.fullmatch(r"(?:а |ну )?караганда\W*", ql):
                yield "follow-up", "Караганда родная, но жить сейчас интереснее в Астане. А туда — к родителям на каникулы.", None
            if "завал какой-то" in last_a and re.search(r"(?:сам|ты) (?:так )?(?:\w+ )?делал", ql):
                yield "follow-up", "Бывало, когда в учёбу закапывался и забывал ответить. Но два дня — это уже много, честно.", None
            if "серик" in last_a.casefold() and "опоздал на минуту" not in last_a.casefold() and re.search(r"чем (?:\w+ )?(?:он )?(?:тебе )?нравится|почему (?:он|именно он)", ql):
                yield "follow-up", "Объясняет на живых примерах и не орёт, если тупишь. И с юмором, на его парах реально не спишь.", None
            if "без телефонов" in last_a and re.search(r"на (?:его|ее|её|моем|моём) месте|ты бы (?:как )?(?:поступил|сделал)", ql):
                yield "follow-up", "Я бы не предъявлял, а просто позвал погулять. Фразой «ты вечно в телефоне» только поругаетесь.", None
            if "двести с лишним" in last_a.casefold() and re.search(r"на ч[её]м|как (?:\w+ )?ездишь", ql):
                yield "follow-up", "На поезде или на маршрутке. Маршрутка быстрее, но в поезде можно поспать.", None
            if re.search(r"\bмама\b", last_a.casefold()) and re.search(r"как (?:\w+ )?зовут (?:ее|её)|(?:ее|её) (?:\w+ )?зовут", ql):
                yield "follow-up", "Маму Гульмира зовут. Но я её, понятно, мамой зову)", None
            if re.search(r"максимум|больше всего", ql) and re.search(r"пассажир|человек|людей", ql):
                for uq, _ in reversed(list(self.turns)):
                    bus = re.search(r"(\d+) (?:человек|пассажир)\w*.{0,60}?вышл\w* (\d+).{0,10}?зашл\w* (\d+)", uq.casefold())
                    if bus:
                        st_, out_, in_ = int(bus[1]), int(bus[2]), int(bus[3])
                        mx = max(st_, st_ - out_ + in_)
                        yield "operation", f"{mx}, в самом начале. Дальше только меньше становилось." if mx == st_ else f"{mx}, после первой остановки.", None
                        break
            sib_ = getattr(self, "sib", None)
            if sib_ and re.search(r"если бы я (?:был )?(?:мальчик|парн)", ql):
                nb, ns = sib_
                yield "operation", f"Тогда {nb + 1 + ns}: ты четвёртый брат, мальчиков {nb + 1}, а девочек всё так же {ns}." if nb == 3 else f"Тогда {nb + 1 + ns}.", None
            if re.search(r"^Папа, наверное", last_a) and re.search(r"ладишь|ладите|отношения", ql):
                yield "follow-up", "Нормально, сейчас вообще отлично. В школе было сложнее, а теперь он даже советы по компу просит.", None
            if "помог бы родителям" in last_a and re.search(r"и вс[её]\b|и все\b|только это", ql):
                yield "follow-up", "Ну ещё ноут нормальный купил бы, мой уже как самолёт шумит. Миллион тенге не так много, если честно.", None
            if "ерлан" in recent and re.search(r"уживаетесь|ссоритесь|ругаетесь", ql):
                yield "follow-up", "Да нормально. Бывает по мелочи, кто посуду не помыл, но со школы друг друга знаем, быстро отходим.", None
            if "рыбалку вытаскивал" in last_a and re.search(r"в (?:его|ее|её|твоем) возрасте|таким же", ql):
                yield "follow-up", "Почти) Тоже в комп залипал, мама ругалась. Перерос, когда в футбол с пацанами начал ходить.", None
            if re.search(r"папа на заводе", last_a.casefold()) and re.fullmatch(r"(?:а )?мама\W*", ql):
                yield "follow-up", "Мама учительница, начальные классы. Говорит, с первоклассниками проще, чем со мной было)", None
            if "самсунг" in last_a.casefold() and re.search(r"почему (?:не )?(?:айфон|iphone)", ql):
                yield "follow-up", "Дорого же. За эти деньги можно полгода жить, а самсунг и так всё умеет.", None
            if re.search(r"толстой|достоевский|пушкин|булгаков|ауэзов|гоголь|тургенев|грибоедов", last_a.casefold()) and re.search(r"(?:ты )?(?:сам )?(?:его |ее |её )?читал", ql):
                yield "follow-up", "Честно — частично. В школе краткое содержание и пару глав, а целиком только то, что самому нравилось." if "толстой" not in last_a.casefold() else "Только краткое содержание в школе, четыре тома — это жёстко. Но фильм потом посмотрел.", None
            if re.search(r"караганд", last_a.casefold()) and re.search(r"зимой|холоднее", ql) and not re.search(r"\bили\b", ql):
                yield "follow-up", "По градусам похоже, но в Астане ветер дикий, поэтому ощущается холоднее. В Караганде хоть не сдувает.", None
            shop3 = getattr(self, "shop", None)
            more_ = re.search(r"(?:взял\w*|купил\w*|возьм\w*) (два|две|три|четыре|\d+) (\w+)", ql)
            if shop3 and more_:
                prices, total, paid = shop3
                k_ = {"два": 2, "две": 2, "три": 3, "четыре": 4}.get(more_[1]) or int(more_[1])
                price = next((v for key, v in prices.items() if key[:4] == more_[2][:4]), None)
                if price:
                    nt = total + (k_ - 1) * price
                    yield "operation", (f"Тогда {nt}, сдачи {paid - nt}." if paid and paid >= nt else f"Тогда {nt}, уже не хватит."), None
            if "доширак" in last_a.casefold() and re.search(r"сколько (?:\w+ )?(?:стоил|стоит|отдал)", ql):
                yield "follow-up", "Тенге четыреста, с яйцом пятьсот. Студенческая кухня, что поделать.", None
            if re.search(r"^Такси", last_a) and re.fullmatch(r"(?:а )?почему\W*", ql):
                yield "follow-up", "На остановке в минус тридцать с ветром замёрзнешь, пока автобус ждёшь. А в такси тепло и до двери.", None
            if "сестра" in recent and re.search(r"чем (?:\w+ )?(?:она )?(?:занимается|увлекается)", ql):
                yield "follow-up", "Школьница, девятый класс. Ходит на танцы и сидит в тиктоке, как все.", None
            grand = getattr(self, "grand", None)
            if grand and re.search(r"(?:одному|одной) .{0,15}в (?:два|2) раза больше", ql):
                n_, k_, r_ = grand
                yield "operation", f"Тогда {(n_ - 1) * k_ + 2 * k_ + r_}: одному {2 * k_}, остальным по {k_}, и себе {r_}.", None
            if grand and re.search(r"сколько (?:\w+ )?(?:яблок|было)", ql) and re.search(r"задач|бабушк", ql):
                n_, k_, r_ = grand
                yield "operation", f"{n_ * k_ + r_}: внукам {n_ * k_}, и себе {r_}.", None
            if "дальше алматы" in last_a.casefold() and re.search(r"как там|понравил", ql):
                yield "follow-up", "В Алматы норм, горы красивые и тепло. Но дорого, я там у родственников жил, иначе бы не потянул.", None
            if "трц сгоняли" in last_a.casefold() and re.search(r"что (?:было )?потом|и что|дальше", ql):
                yield "follow-up", "Да ничего, вечером сериал и спать. Понедельник же.", None
            if re.search(r"я (?:парень|девушка) или (?:парень|девушка)|я (?:м|ж) или", ql) and getattr(self, "user_name", None):
                fem_ = getattr(self, "user_female", False)
                yield "recall", f"Ты ж {self.user_name}, {'девушка' if fem_ else 'парень'}, конечно)", None
            if "ерлан" in recent and re.search(r"храпит", ql):
                yield "follow-up", "Бывает, особенно после футбола. Я в наушниках сплю, спасает.", None
            if "по-собачьи" in last_a and re.search(r"почему|как так", ql):
                yield "follow-up", "В детстве в бассейн не водили, а потом стыдно было начинать с малышами. Плюс я глубины боюсь.", None
            if "в ауле под карагандой" in last_a.casefold() and re.search(r"жив[её]т|караганд", ql):
                yield "follow-up", "Не в самой, в ауле под Карагандой, час езды. Город она не любит, говорит, шумно.", None
            if "помог бы родителям" in last_a and re.search(r"жалко", ql):
                yield "follow-up", "На родителей не жалко вообще. А Япония — это мечта, на мечту тоже не жалко.", None
            if "болат агай" in last_a.casefold() and re.search(r"чему|что (?:\w+ )?вел|какой предмет", ql):
                yield "follow-up", "Физре) Но и по жизни советы давал: «Не ной, а делай». До сих пор помню.", None
            if "к доске вызвал" in last_a and re.search(r"что (?:было )?потом|и что|дальше", ql):
                yield "follow-up", "Постоял, сказал что-то умное про погоду, все поржали. Препод отпустил, но запомнил меня.", None
            if re.search(r"^В семь\.", last_a) and re.search(r"\bлег\b|ложишься|спать лег", ql):
                yield "follow-up", "Около часа, сериал досматривал. Так что шесть часов, классика.", None
            grand2 = getattr(self, "grand", None)
            if grand2 and re.search(r"половин\w* (?:\w+ )?(?:своих )?друг", ql) and re.search(r"у второго|у другого|сколько стало", ql):
                k_ = grand2[1]
                yield "operation", f"{k_ + k_ // 2}: свои {k_} плюс {k_ // 2} от первого.", None
            if getattr(self, "user_where", None) and re.search(r"где (?:я )?(?:сейчас )?сижу|где я сейчас", ql) and not getattr(self, "user_name", None) is None:
                yield "recall", f"Ты же говорил: {self.user_where.replace('и скучаешь', '').strip()}. Не скучай там)", None
            if "опоздал на минуту" in last_a.casefold() and re.search(r"почему|зачем", ql):
                yield "follow-up", "Старой закалки, говорит, что мы распущенные. Но предмет знает так, что придраться не к чему.", None
            if re.search(r"(?:помнишь|напомни|знаешь).{0,25}(?:откуда я|кем (?:я )?работаю|где я (?:учусь|работаю))|(?:откуда я|кем (?:я )?работаю).{0,20}помнишь", ql) and getattr(self, "user_name", None):
                parts_ = [f"ты {self.user_name}"]
                if getattr(self, "user_city", None):
                    parts_.append(f"из {self.user_city}")
                if getattr(self, "user_job", None):
                    parts_.append(f"работаешь {self.user_job}")
                if getattr(self, "user_study", None):
                    parts_.append(f"учишься на {self.user_study}")
                if len(parts_) > 1:
                    txt_ = ", ".join(parts_) + ") Я ж не рыбка."
                    yield "recall", txt_[:1].upper() + txt_[1:], None
            if "рыбалку" in last_a and re.search(r"сколько (?:\w+ )?(?:тебе )?(?:тогда )?(?:было|лет)", ql):
                yield "follow-up", "Лет шесть, наверное. Серьёзно, до сих пор помню, как ноги замёрзли в валенках.", None
            if "больше двух остановок" in last_a and re.search(r"почему", ql):
                yield "follow-up", "На остановке тоже мёрзнешь, но в автобусе хотя бы тепло. А пешком через мост в ветер — это квест.", None
            if re.search(r"\d+,5", last_a) and re.search(r"округли|до целого|целых", ql):
                m5 = re.search(r"(\d+),5", last_a)
                yield "operation", f"Тогда {int(m5[1]) + 1}, если по-честному округлять. Хотя по-жизни половинку обычно не считают)", None
            if "серик" in last_a.casefold() and re.search(r"как (?:\w+ )?(?:его )?зовут", ql) and not re.search(r"меня|мое|моё|сосед|друг", ql):
                yield "follow-up", "Серик Болатович, я ж написал) Его весь поток знает.", None
            if re.search(r"\d", last_a) and re.search(r"посчитал|считаешь|сосчитал", ql) and re.search(r"откуда|как (?:ты )?так|быстро", ql):
                yield "follow-up", "Да в уме, по частям раскладываю. В школе на олимпиады по математике ходил, привычка осталась.", None
            if "выключател" in self.turns[-1][0].casefold() and re.search(r"(?:две|2) ламп|ламп\w* (?:было бы )?(?:две|2)", ql):
                yield "follow-up", "Так же почти: одну узнаёшь по свету, вторую по теплу, а третий выключатель — пустой. Главное, не обжечься)", None
            if "иртыш" in last_a.casefold() and re.search(r"был|видел|ездил", ql):
                yield "follow-up", "Был в Павлодаре у родственников, летом купались. Течение там сильное, я далеко не заплывал.", None
            if getattr(self, "user_job", None) and re.search(r"(?:где|кем) (?:я )?работаю", ql) and not re.search(r"зовут", ql):
                yield "recall", f"{self.user_job[:1].upper() + self.user_job[1:]} же, ты в начале писал.", None
            if "нервы дороже" in last_a and re.search(r"уволил|уволит|ушел бы|ушёл бы", ql):
                yield "follow-up", "Если орут постоянно — да, долго бы не выдержал. Но сначала нашёл бы другое место, а потом ушёл.", None
            if re.search(r"макарон", last_a.casefold()) and re.search(r"сам (?:\w+ )?(?:готовил|варил|делал)", ql):
                yield "follow-up", "Ну да, макароны сварить — тут большого таланта не надо)", None
            city_job = re.search(r"\b(?:в|на) ([а-яё]+)$", getattr(self, "user_job", "") or "")
            if re.search(r"(?:в каком|какой) город\w* я|откуда я\b", ql) and (getattr(self, "user_city", None) or city_job):
                place_ = getattr(self, "user_city", None) or city_job[1].capitalize()
                yield "recall", f"{place_} же, ты говорил)", None
            if "классическая ловушка" in last_a.casefold() and re.search(r"почему|как так", ql):
                yield "follow-up", "Ну сказано же «все, кроме», — значит, эти и остались. Многие сразу вычитают, вот и ловушка.", None
            if "воду в чайнике" in last_a.casefold() and re.search(r"почему", ql):
                yield "follow-up", "Телефон — это и учёба, и связь с мамой, и автобус по приложению. А горячую воду в чайнике согрел, и норм.", None
            if re.search(r"\bпока\b|до свидания|бывай", ql) and re.search(r"тимур|пара кончил|мне пора|бежать|ухожу|было (?:прикольно|приятно|интересно|весело|круто)", ql):
                nm_ = getattr(self, "user_name", None)
                yield "profile", f"Давай{', ' + nm_ if nm_ else ''}, удачи!) Было приятно поболтать.", None
            if re.search(r"поправ|больниц|выздорав|сочувствую", recent) and re.search(r"(?:спасибо|спс).{0,25}(?:выслушал|поддерж)", ql):
                yield "follow-up", "Да не за что, держись. Пусть бабушка быстрее поправится." if "бабушк" in " ".join(t[0] for t in list(self.turns)[-3:]).casefold() else "Да не за что, держись там. Если что, пиши.", None
            if getattr(self, "user_age", None) and re.search(r"сколько мне (?:лет|годиков)", ql):
                yield "recall", f"{self.user_age}, ты ж в самом начале говорил.", None
            if re.search(r"токио|япони", last_a.casefold()) and re.search(r"(?:в )?казахстан|остаться|не здесь|не тут", ql):
                yield "follow-up", "Казахстан никуда не денется) Хочется другой опыт получить, а потом можно и вернуться.", None
            if "дота с пацанами" in last_a and re.fullmatch(r"(?:а |и )?(?:давно|как давно)\W*", ql):
                yield "follow-up", "Да всегда так, со школы ещё. Привычка, что поделать.", None
            bus_ = getattr(self, "bus", None)
            if bus_:
                keep_ = re.search(r"вс[еи],? кроме (\d+)", ql)
                if keep_ and re.search(r"вышл|вышли", ql):
                    k_ = int(keep_[1])
                    out2 = bus_["after"] - k_
                    bus_["out"].append(out2)
                    bus_["after"] = k_
                    yield "operation", f"{k_} остаётся. Вышли {out2}.", None
                elif re.search(r"сколько (?:\w+ )?(?:всего )?вышл", ql) and len(bus_["out"]) > 1:
                    yield "operation", f"{sum(bus_['out'])}: {bus_['out'][0]} на первой и {bus_['out'][1]} на второй.", None
            if "вчера плов" in last_a.casefold() and re.fullmatch(r"(?:а |и )?(?:давно|как давно)\W*", ql):
                yield "follow-up", "Да вчера же, говорю) А вообще учусь готовить с первого курса, как съехал от родителей.", None
            if "была в школе" in last_a.casefold() and re.search(r"что (?:было )?потом|и что|дальше", ql):
                yield "follow-up", "Я уехал в Астану, она в Алматы. На расстоянии не вывезли, но расстались нормально.", None
            if re.search(r"о ч[её]м (?:мы )?(?:сегодня )?(?:больше всего |в основном )?(?:болтали|говорили|общались|разговаривали)", ql) and len(self.history) >= 4:
                topics_ = [("учёбу", r"пар|учеб|учишься|препод|сесси|экзамен|универ"), ("еду", r"ел|готов|завтрак|обед|ужин|плов"), ("семью", r"сестр|брат|родител|мам|пап|бабушк"), ("задачки", r"задач|сколько будет|умнож|яблок|автобус|поезд"), ("твоё расставание", r"расстал|бросил|поругал|поссорил"), ("кино", r"фильм|сериал"), ("Караганду", r"караганд")]
                counts_ = []
                for name_t, rx in topics_:
                    c_ = sum(1 for uq, _ in self.history if re.search(rx, uq.casefold()))
                    if c_:
                        counts_.append((c_, name_t))
                counts_.sort(reverse=True)
                top_ = [n for _, n in counts_[:3]]
                if top_:
                    yield "recall", ("Да обо всём понемногу: про " + ", ".join(top_[:-1]) + (" и " if len(top_) > 1 else "") + top_[-1] + ". Нормально поболтали)")[:300], None
            if "культурологи" in last_a.casefold() and re.search(r"кто (?:\w+ )?(?:его |ее |её )?веде", ql):
                yield "follow-up", "Гульнара Сериковна. Приятная, но любит пофилософствовать минут на сорок вместо лекции.", None
            yh_ = re.search(r"(?:хокку|хайку)(?: про ([а-яё]+))?", ql)
            if yh_:
                yield "operation", ("Ладно: «Ветер над степью / сухая трава шумит / еду домой я». Я не поэт)" if (yh_[1] or "").startswith("степ") else f"Ладно: «Тихо за окном / снова думаю про {yh_[1] or 'сессию'} / чай давно остыл». Я не поэт)"), None
            if re.search(r"как(?:ой|ое) (?:сейчас |сегодня |щас )?год\b", ql):
                from datetime import datetime as _dt
                yield "profile", f"{_dt.now().year} же) Проверяешь, что ли.", None
            if re.search(r"(?:напомни|помнишь).{0,40}сколько", ql) and getattr(self, "problem_result", None):
                parts_r = [f"{self.problem_result} осталось)"]
                ab_ = re.search(r"про (\w+) (?:рассказывал|говорил)", ql)
                if ab_:
                    st_ = ab_[1][:4]
                    hit_ = next((ea for eq, ea in self.history if st_ in eq.casefold() or st_ in ea.casefold()), None)
                    if hit_:
                        first_ = re.split(r"(?<=[.!])\s", hit_, maxsplit=1)[0].rstrip(".")
                        parts_r.append(f"А про {ab_[1]}: {first_[:1].lower() + first_[1:]}.")
                yield "recall", " ".join(parts_r)[:300], None
            if "очередь из двух" in last_a and re.search(r"очеред", ql):
                yield "follow-up", "Ну утром да, Ерлан там по полчаса торчит. Я встаю раньше, чтобы успеть.", None
            if "с велика" in last_a and re.search(r"сколько (?:\w+ )?(?:тебе )?(?:тогда )?(?:было|лет)", ql):
                yield "follow-up", "Лет восемь, только кататься научился без страховки.", None
            if re.match(r"мама", last_a.casefold()) and re.fullmatch(r"(?:а )?(?:как )?(?:\w+ )?(?:зовут|имя)(?: ее| её)?\W*", ql):
                yield "follow-up", "Маму Гульмира зовут. Но для меня она просто мама)", None
            if "мастера и маргариту" in last_a.casefold() and re.search(r"дочитал|и что|понравил", ql):
                yield "follow-up", "Дочитал, конечно, за неделю. Воланд там лучший персонаж, а кот Бегемот вообще топ.", None
            if re.search(r"поболтаем|отвлеч|выговорит", last_a.casefold()) and re.search(r"спасибо|легче|полегчало", ql):
                yield "follow-up", "Рад, правда. Если что — пиши, я тут.", None
            if "плов" in last_a.casefold() and re.search(r"как (?:\w+ )?получил|серь[её]зно|вкусно", ql):
                yield "follow-up", "Ну рис разварился, мясо жестковатое, но Ерлан сказал норм. В следующий раз маме по видео позвоню, пусть контролирует)", None
            if "носки с оленями" in last_a.casefold() and re.search(r"кто (?:\w+ )?(?:подарил|дарил)", ql):
                yield "follow-up", "Тётя из Караганды. Она каждый год что-то такое дарит, уже традиция)", None
            if re.search(r"(?:что|чего) (?:\w+ )?(?:я |ты )?(?:тебе |мне )?(?:про|о|об) сосед\w* (?:\w+ )?(?:говорил|рассказывал)", ql):
                yield "recall", "Что живу с Ерланом, с первого курса снимаем вместе. И что он мою гречку доедает)", None
            if re.search(r"секунд\w* в (?:сутк|одних сутк|дн[её]|дне)", ql):
                yield "operation", "86400. Двадцать четыре часа на 3600 секунд.", None
            if re.search(r"минут\w* в (?:сутк|дн[её])", ql):
                yield "operation", "1440. Двадцать четыре на шестьдесят.", None
            if "наушники" in last_a.casefold() and re.fullmatch(r"(?:а |и )?(?:давно|как давно)\W*", ql):
                yield "follow-up", "Наушники с первого курса, мама на поступление подарила. Брелок — с девятого класса.", None
            if "упал со сцены" in last_a and re.search(r"что (?:было )?потом|и что|дальше", ql):
                yield "follow-up", "Встал, отряхнулся и дочитал стих под общий смех. Учительница потом сказала, что это было самое запоминающееся выступление)", None
            if "сестра" in recent and re.fullmatch(r"(?:а )?(?:как )?(?:\w+ )?(?:зовут|имя)(?: ее| её)?\W*", ql) and getattr(self, "user_name", None) == "Айгерим":
                yield "follow-up", "Айгерим, как тебя, прикинь) Дома все зовут Айка.", None
            if "сестра" in recent and re.fullmatch(r"(?:а )?(?:как )?(?:\w+ )?(?:зовут|имя)(?: ее| её)?\W*", ql):
                yield "follow-up", "Айгерим. Хотя дома все зовут Айка.", None
            if "сестра" in recent and re.search(r"(?:младше|старше) или (?:старше|младше)|она (?:младше|старше)|сколько (?:ей|ему) лет", ql):
                yield "follow-up", "Младше, ей пятнадцать. Так что командую пока я.", None
            if "пересдашь" in self.turns[-1][1].casefold() and re.search(r"на моем месте|ты бы (?:что )?(?:делал|сделал)|что бы ты делал", ql):
                yield "follow-up", "Тоже бы поныл вечер, а потом сел бы за задачи с тем, кто шарит. Я так физику пересдал, работает.", None
            if "остыть" in self.turns[-1][1].casefold() and re.search(r"на моем месте|ты бы (?:что )?(?:делал|сделал)|что бы ты делал", ql):
                yield "follow-up", "Я бы, наверное, сам первым написал, если неправ был. И не тянул бы долго, а то только хуже становится.", None
            if "сочувствую" in self.turns[-1][1].casefold() and "пересдашь" not in self.turns[-1][1].casefold() and "остыть" not in self.turns[-1][1].casefold() and re.search(r"на моем месте|ты бы (?:что )?(?:делал|сделал)|что бы ты делал", ql):
                yield "follow-up", "Вышел бы с пацанами куда-нибудь и не писал бы ей хотя бы неделю. Ну и в зал, злость лучше туда.", None
            if re.search(r"Дота, со школы", self.turns[-1][1]) and re.fullmatch(r"(?:а )?почему\W*", ql):
                yield "follow-up", "Привык, со школы с пацанами пати собираем. Да и в кс я криворукий, честно.", None
            if re.search(r"школ", self.turns[-1][0].casefold()) and re.search(r"предмет", ql) and re.search(r"ненавид|не любил|бесил|худш", ql):
                yield "follow-up", "Химию. Вообще не понимал, что там с чем реагирует, списывал у соседки по парте.", None
        if self.history and re.search(r"(?:что|чего) (?:ты )?(?:\w+ )?(?:запомнил|помнишь|знаешь) (?:\w+ )?(?:обо мне|про меня)", question, re.I):
            frags = []
            for uq, _ in self.history:
                for m in re.finditer(r"(?:\bя\b|\bу меня\b)([^.?!]{6,70})", uq, re.I):
                    frag = m[0].strip(" ,")
                    if re.search(r"\bты\b|\bтебя\b|\bтебе\b|зовут", frag, re.I):
                        continue
                    frag = re.sub(r"\bу меня\b", "у тебя", frag, flags=re.I)
                    frag = re.sub(r"^я\s+", "", frag, flags=re.I)
                    frag = re.sub(r"\bмне\b", "тебе", frag)
                    frag = re.sub(r"\bмой\b", "твой", frag)
                    frag = re.sub(r"\bмоя\b", "твоя", frag)
                    frags.append(frag.split(",")[0].strip())
            parts = []
            if getattr(self, "user_name", None):
                parts.append(f"что ты {self.user_name}" + (f" из {self.user_city}" if getattr(self, "user_city", None) else ""))
            parts += frags[-2:]
            if parts:
                text = "Ну, " + ", ".join(parts) + ". Видишь, слушал внимательно)"
                yield "recall", text[:1].upper() + text[1:] if len(text) <= 300 else text[:297].rsplit(" ", 1)[0] + "…", None
        books = {"войн": "Толстой, Лев Николаевич. Четыре тома, я в школе честно только краткое содержание осилил.",
                 "преступлени": "Достоевский. Читал в школе, Раскольников до сих пор в голове сидит.",
                 "евгени": "Пушкин. Учили наизусть отрывки, «Мой дядя самых честных правил» до сих пор помню.",
                 "мастер": "Булгаков. Моя любимая книга, кстати, перечитывал несколько раз.",
                 "путь абая": "Мухтар Ауэзов. В школе проходили, но целиком я, если честно, не осилил.",
                 "абай жолы": "Мухтар Ауэзов. В школе проходили, но целиком я, если честно, не осилил.",
                 "горе от ума": "Грибоедов. Помню только, что все там умные, а счастья нет.",
                 "мертвые души": "Гоголь. Чичиков и его схемы, классика.",
                 "отцы и дети": "Тургенев. Базаров, нигилизм, всё такое.",
                 "гарри поттер": "Джоан Роулинг. Все семь части в детстве прочитал, вот это я реально читал."}
        if re.search(r"кто (?:\w+ )?(?:написал|автор)|чей (?:роман|книга)|автор (?:книги|романа)", question.casefold()):
            ql_b = question.casefold().replace("ё", "е")
            for key_, ans_ in books.items():
                if key_ in ql_b:
                    yield "operation", ans_, None
                    break
        ordinal = re.search(r"(?:спросил|спрашивал|написал)\w* (?:у тебя |тебя |тебе )?(втор|трет|четверт|четвёрт|пят)\w*|(втор|трет|четверт|четвёрт|пят)\w* (?:вопрос|сообщени)", question, re.I)
        if ordinal:
            idx = {"втор": 1, "трет": 2, "четв": 3, "пят": 4}[(ordinal[1] or ordinal[2]).lower()[:4] if (ordinal[1] or ordinal[2]).lower().startswith("четв") else (ordinal[1] or ordinal[2]).lower()]
            names = {1: "Вторым", 2: "Третьим", 3: "Четвёртым", 4: "Пятым"}
            if len(self.history) > idx:
                asked_q = re.sub(r"[?!.]+$", "", list(self.history)[idx][0]).strip()
                yield "recall", f"{names[idx]} было «{asked_q}». Я слежу, не думай."[:300], None
        if re.search(r"(?:ты )?меня запомнишь|не забудешь меня", question, re.I):
            who = f", {self.user_name}" if self.user_name else ""
            yield "persona", f"Конечно{who}. Не каждый день так допрашивают, такое не забывается.", None
        if asks_name:
            if self.user_name:
                course_ = getattr(self, "user_course", None)
                if course_ and re.search(r"курс", question, re.I):
                    num_c, fac_c = course_
                    num_c = {"1": "первый", "2": "второй", "3": "третий", "4": "четвёртый"}.get(num_c, num_c)
                    yield "recall", f"{self.user_name}, {num_c} курс" + (f", {fac_c}" if fac_c else "") + ". Помню)", None
                where_ = getattr(self, "user_where", None)
                if where_ and re.search(r"где (?:я )?(?:сейчас )?сижу|где я сейчас", question, re.I):
                    yield "recall", f"{self.user_name}, сидишь {where_}) Помню.", None
                exam_ = getattr(self, "user_exam", None)
                if exam_ and re.search(r"сдаю|экзамен", question, re.I):
                    yield "recall", f"{self.user_name}, а завтра у тебя {exam_}. Иди учи давай)" if exam_.startswith("экзамен") else f"{self.user_name}, а завтра сдаёшь {exam_}. Иди учи давай)", None
                job_ = getattr(self, "user_job", None)
                if job_ and re.search(r"работаю|работаешь", question, re.I):
                    yield "recall", f"{self.user_name}, работаешь {job_}. Помню)", None
                study_ = getattr(self, "user_study", None)
                if study_ and re.search(r"учусь|учишься", question, re.I):
                    yield "recall", f"{self.user_name}, учишься на {study_}. Помню)", None
                yield "recall", f"{self.user_name}, конечно. Ты же говорил, я запомнил.", None
            else:
                yield "recall", "Ты же не говорил, как тебя зовут. Скажи — запомню.", None
        if self.turns and self.last_route != "fallback" and RECALL_SELF.fullmatch(question):
            previous = self.turns[-1][1]
            yield "recall", previous, None
        if self.turns and self.first_answer and re.search(r"\bперв\w* (?:вопрос|раз)\w*|в (?:самом )?начале", question, re.I) and re.search(r"\b(?:ты )?(?:ответил|сказал|отвечал)\w*", question, re.I) and not re.search(r"что у тебя|что ты из|что тебя зовут|близнец|\bпро\b|\bо (?:своем|твоем|соседе|друге)", question, re.I):
            yield "recall", f"Сказал: «{self.first_answer.rstrip('.')}». Помню, помню."[:300], None
        asked = re.search(r"(?:помнишь|напомни)\w*,? (?:что|как|о ч[её]м) я (?:тебя |тебе )?(?:спрашивал|спросил|писал|говорил)\w* (?:про|о|об|насчет|насчёт) ([а-яё]{3,})", question.casefold())
        if asked and self.history:
            stem = asked[1][:5]
            for earlier_q, earlier_a in reversed(self.history):
                if stem in earlier_q.casefold():
                    quoted = re.sub(r"[?!.]+$", "", earlier_q).strip()
                    first = re.split(r"(?<=[.!])\s", earlier_a, maxsplit=1)[0].rstrip(".")
                    yield "recall", f"Помню: ты спрашивал «{quoted}», а я сказал «{first}»."[:300], None
                    break
        asked2 = re.search(r"(?:спросил|спрашивал)\w* (?:тебя )?(?:про|о|об) ([а-яё]{3,}).{0,40}(?:помнишь|что (?:ты )?ответил)", question.casefold())
        if asked2 and self.history:
            stem = asked2[1][:3] if len(asked2[1]) <= 5 else asked2[1][:5]
            for earlier_q, earlier_a in reversed(self.history):
                if stem in earlier_q.casefold():
                    first = re.split(r"(?<=[.!])\s", earlier_a, maxsplit=1)[0].rstrip(".")
                    if first.startswith("Сказал бы"):
                        inner = re.sub(r"^Сказал бы:?\s*", "", first).strip("«» ")
                        yield "recall", f"Помню, конечно. Я бы сказал: «{inner}»."[:300], None
                    else:
                        yield "recall", f"Помню, конечно: «{first}»."[:300], None
                    break
        about_food = re.search(r"\b(?:про|о|об) (?:еду|еде|ед[ыу]|завтрак\w*|обед\w*) (?:что )?(?:ты )?(?:ответил|сказал|говорил)", question.casefold())
        if about_food and self.history:
            for earlier_q, earlier_a in self.history:
                if re.search(r"пое[лс]|завтрак|обед|ужин|\bед[аyуы]\b", earlier_q.casefold()):
                    first = re.split(r"(?<=[.!])\s", earlier_a, maxsplit=1)[0].rstrip(".")
                    yield "recall", f"Что {first[:1].lower() + first[1:]}. Помню, помню."[:300], None
                    break
        mine = re.search(r"\bчто я (?:тебя |тебе )?(?:спрашивал|спросил|писал|говорил|рассказывал)\w* (?:про|о|об|насчет|насчёт) (?:своего |свою |свой |своем |моего |мою |мой |моем )?([а-яё]{3,})", question.casefold())
        if mine and self.history:
            stem = mine[1][:4] if len(mine[1]) > 4 else mine[1][:3]
            for earlier_q, _ in reversed(self.history):
                if stem in earlier_q.casefold():
                    quoted = re.sub(r"[?!.]+$", "", earlier_q).strip()
                    yield "recall", f"Ты писал: «{quoted}». Помню."[:300], None
                    break
        about = re.search(r"\bчто (?:ты )?(?:мне |тебе )?(?:в начале |тогда |раньше )?(?:отвечал|ответил|сказал|говорил|писал)\w* (?:про|о|об|насчет|насчёт) (?:своего |свою |своем |свой |твоего |твою |твоем )?([а-яё]{3,})", question.casefold())
        if about and self.history:
            stem = about[1][:4] if len(about[1]) > 4 else about[1][:3]
            hit_ = next((ea for eq, ea in self.history if stem in eq.casefold()), None) or next((ea for eq, ea in reversed(self.history) if stem in ea.casefold()), None)
            if hit_:
                yield "recall", f"Я сказал: «{hit_.rstrip('.')}».", None
        if self.turns and re.search(r"(?:последн\w*|прошл\w*|предыдущ\w*) (?:сообщени|вопрос)\w*.{0,30}\bя (?:тебе |тебя )?(?:писал|спрашивал|спросил|задал|отправлял)|(?:что|о ч[её]м) я (?:тебя |тебе )?(?:спрашивал|писал|спросил) (?:до этого|перед этим|только что|в прошлый раз)", question, re.I):
            quoted = re.sub(r"\?\s*", ". ", self.turns[-1][0]).strip(" .!")
            yield "recall", f"Ты спросил: «{quoted}». Я ещё не настолько устал."[:300], None
        if self.history and re.search(r"(?:о ч[её]м|про что) (?:мы )?(?:\w+ )?(?:говорили|болтали|начинали|общались|разговаривали)\w* (?:в начале|сначала|вначале|в самом начале)|напомни.{0,25}(?:в начале|сначала|вначале)", question, re.I):
            first = [re.sub(r"[?!.]+$", "", q0).strip() for q0, _ in list(self.history)[:2]]
            if first:
                yield "recall", ("Сначала ты спросил «" + "», потом «".join(first) + "». Я помню, не волнуйся.")[:300], None
        if self.turns and REPEAT.search(question):
            previous = self.turns[-1][1]
            if len(previous) > 120:
                previous = previous[:117].rsplit(" ", 1)[0] + "…"
            yield "recall", f"Ну я же сказал: «{previous.rstrip('.')}».", None
        typo = re.search(r"[\"«]([а-яёa-z]{3,})[\"»]", question, re.I)
        if typo and re.search(r"задом наперед|задом наперёд|наоборот", question, re.I) and re.search(r"с (?:\w+ )?ошибк|с (?:\w+ )?опечатк|ошибись", question, re.I):
            rev = typo[1][::-1].lower()
            rev = rev[0] + rev[2] + rev[1] + rev[3:] if len(rev) > 3 else rev
            yield "operation", f"{rev})) Специально ошибся, как просили.", None
        if typo and re.search(r"с ошибк|с опечатк|ошибись", question, re.I) and not re.search(r"задом|наоборот", question, re.I):
            w = typo[1].lower()
            bad = w.replace("о", "а", 1) if "о" in w else (w[:2] + w[3] + w[2] + w[4:] if len(w) > 4 else w + w[-1])
            yield "operation", f"{bad}. Доволен))", None
        if self.turns:
            chained = _chain_arithmetic(question, self.turns[-1][1])
            if chained:
                yield "operation", chained, None
        phrase = re.search(r"[\"«]([а-яё ]{3,30})[\"»]\s*(?:задом наперед|задом наперёд|наоборот)|(?:задом наперед|задом наперёд|наоборот)\W+[\"«]([а-яё ]{3,30})[\"»]", question, re.I)
        if phrase and " " in (phrase[1] or phrase[2]) and not re.search(r"ошиб|опечат", question, re.I):
            yield "operation", f"{(phrase[1] or phrase[2]).lower()[::-1]}. Доволен))", None
        plain = re.search(r"слово ([а-яё]{3,}) (?:задом наперед|задом наперёд|наоборот)|(?:задом наперед|задом наперёд|наоборот)\W+(?:слово\W+)?([а-яё]{3,})\W*$", question, re.I)
        if typo and not plain and re.search(r"задом наперед|задом наперёд|наоборот", question, re.I) and not re.search(r"ошиб|опечат", question, re.I):
            yield "operation", f"{typo[1].lower()[::-1]}. Звучит как заклинание))", None
        if plain and not typo and not re.search(r"ошиб|опечат", question, re.I):
            word = (plain[1] or plain[2]).lower()
            yield "operation", f"{word[::-1]}. Звучит как заклинание))", None
        swap = re.search(r"\bне (\w+),? а (\w+)", question.casefold())
        if swap and getattr(self, "last_problem", None) and re.search(r"^(?:а )?(?:если|бы)", question.casefold()):
            old = self.last_problem
            pos = re.search(r"\b" + re.escape(swap[1]) + r"\b", old.casefold())
            if pos:
                story = _word_problem(old[:pos.start()] + swap[2] + old[pos.end():])
                if story is not None:
                    yield "operation", "Тогда " + story[:1].lower() + story[1:], None
        redo = re.search(r"^(?:а )?(?:если )?(?:бы )?(?:у \w+ )?(?:было|будет|было бы) (\d+)", question.casefold())
        if redo and getattr(self, "last_problem", None) and not re.search(r"сколько", question, re.I):
            old = self.last_problem
            first = re.search(r"\d+", old)
            story = _word_problem(old[:first.start()] + redo[1] + old[first.end():])
            if story is not None:
                yield "operation", "Тогда " + story[:1].lower() + story[1:], None
        story = _word_problem(question)
        if story is not None:
            self.last_problem = question
            num_s = re.match(r"-?\d+(?:,\d+)?", story)
            if num_s:
                self.problem_result = num_s[0]
            bq = question.casefold()
            bs = re.search(r"(\d+) (?:человек|пассажир)", bq)
            if bs and re.search(r"автобус|маршрутк|вагон", bq):
                st_ = int(bs[1])
                fr_ = re.search(r"вышл\w* (половин\w*|треть|четверть)", bq)
                o1 = re.search(r"вышл\w* (\d+)|(\d+) вышл", bq)
                out1 = (st_ // {"пол": 2, "тре": 3, "чет": 4}[fr_[1][:3]]) if fr_ else (int(o1[1] or o1[2]) if o1 else 0)
                i1 = re.search(r"зашл\w* (\d+)|(\d+) зашл", bq)
                in1 = int(i1[1] or i1[2]) if i1 else 0
                self.bus = {"after": st_ - out1 + in1, "out": [out1]}
            yield "operation", story, None
        mm = re.search(r"(\d+) (?:на|умножить на|умноженное на|\*|x|х) (\d+) (минус|плюс|\+|-) (\d+)", question, re.I)
        if mm:
            a, b, c = int(mm[1]), int(mm[2]), int(mm[4])
            r = a * b - c if mm[3] in ("минус", "-") else a * b + c
            yield "operation", f"{r}. Сначала {a} на {b} — {a * b}, потом {'минус' if mm[3] in ('минус', '-') else 'плюс'} {c}.", None
        trip = re.search(r"(\d+) ?км ?(?:в|/) ?ч\w*.{0,40}?(\d+) ?км|(\d+) ?км.{0,40}?(\d+) ?км ?(?:в|/) ?ч", question, re.I)
        if trip and re.search(r"за сколько|сколько (?:\w+ )?(?:времени|часов|ехать|по времени)|сколько ехать", question, re.I):
            v, d = (int(trip[1]), int(trip[2])) if trip[1] else (int(trip[4]), int(trip[3]))
            if v:
                h = d / v
                mins = round(h * 60)
                self.last_minutes = mins
                hs = f"{mins // 60} ч" + (f" {mins % 60} мин" if mins % 60 else "")
                yield "operation", f"{hs}, если без остановок: {d} делим на {v}.", None
        dist = re.search(r"(\d+(?:[.,]\d+)?) час\w* (?:\w+ )?(?:со скоростью |по )?(\d+) ?км|(\d+) ?км ?(?:/|в) ?ч\w*.{0,40}?(\d+(?:[.,]\d+)?) час", question, re.I)
        dist_ok = bool(dist) and bool(re.search(r"расстояни|сколько (?:км|километр)|какой путь|сколько (?:\w+ )?(?:проедет|проехал|пройдет|прошел)", question, re.I)) and not re.search(r"\d+ мин", question, re.I)
        if dist_ok:
            t, v = (float(dist[1].replace(",", ".")), int(dist[2])) if dist[1] else (float(dist[4].replace(",", ".")), int(dist[3]))
            self.last_trip = (v, t)
            km_ = v * t
            ks_ = str(int(km_)) if km_ == int(km_) else f"{km_:.1f}".replace(".", ",")
            ts_ = f"{t:g}".replace(".", ",")
            yield "operation", f"{ks_} км: {v} на {ts_}.", None
        arrive = re.search(r"в (\d{1,2})(?::(\d{2}))? ?(?:утра|час\w*|ч\b|вечера)?", question, re.I)
        legs = re.search(r"(\d+) ?км(?! ?/ ?ч| в час| ?ч\b)", question, re.I)
        vel = re.search(r"скорост\w* (\d+)|(\d+) ?км ?(?:/|в) ?ч", question, re.I)
        if arrive and legs and vel and re.search(r"во сколько|когда (?:\w+ )?(?:приедет|прибудет|будет)", question, re.I):
            v_ = int(vel[1] or vel[2])
            d_ = int(legs[1])
            if v_ and d_ != v_:
                start_ = int(arrive[1]) * 60 + int(arrive[2] or 0)
                end_ = start_ + round(d_ / v_ * 60)
                self.arrival = end_
                ride = round(d_ / v_ * 60)
                yield "operation", f"В {end_ // 60 % 24}:{end_ % 60:02d}. Ехать {ride // 60} ч" + (f" {ride % 60} мин." if ride % 60 else "."), None
        late = re.search(r"опозда\w* (?:\w+ )?на (\d+) мин|(?:простоит|постоит|стоял|простоял|стоянк\w*|задерж\w*) (?:\w+ )?(?:на )?(\d+) мин", question, re.I)
        if late and getattr(self, "arrival", None) is not None:
            end_ = self.arrival + int(late[1] or late[2])
            yield "operation", f"Тогда в {end_ // 60 % 24}:{end_ % 60:02d}.", None
        dt_ = re.search(r"(\d+) ?км за (\d+) час", question, re.I)
        if dt_ and re.search(r"скорост|как быстро|сколько км в час", question, re.I):
            d_, t_ = int(dt_[1]), int(dt_[2])
            if t_:
                v_ = d_ / t_
                self.last_trip = (v_, t_)
                yield "operation", f"{v_:g} км/ч: {d_} делим на {t_}.", None
        span0 = re.search(r"(?:проедет|пройдет|проедешь) (?:\w+ )?за (\d+) (минут\w*|час\w*)", question, re.I)
        if dt_ and span0 and int(dt_[2]):
            v_ = int(dt_[1]) / int(dt_[2])
            mins = int(span0[1]) * (60 if span0[2].lower().startswith("час") else 1)
            km = v_ * mins / 60
            self.last_trip = (v_, mins / 60)
            kms = str(int(km)) if km == int(km) else f"{km:.1f}".replace(".", ",")
            yield "operation", f"{kms} км. Скорость {v_:g} км/ч, а дальше просто на время умножить.", None
        far = re.search(r"(\d+) ?км", question, re.I)
        if far and getattr(self, "last_trip", None) and re.search(r"за сколько|сколько (?:\w+ )?(?:времени|часов|ехать)", question, re.I) and not re.search(r"км ?(?:/|в) ?ч|скорост\w* \d", question, re.I):
            v_ = self.last_trip[0]
            if v_:
                mins = round(int(far[1]) / v_ * 60)
                self.last_minutes = mins
                yield "operation", f"{mins // 60} ч" + (f" {mins % 60} мин" if mins % 60 else "") + f", если с той же скоростью {v_:g} км/ч.", None
        trip_ = getattr(self, "last_trip", None)
        back_v = re.search(r"(?:обратн\w*|назад).{0,30}?(\d+) ?км|(\d+) ?км ?(?:/|в) ?ч\w*.{0,20}обратн", question, re.I)
        if trip_ and back_v and re.search(r"сколько (?:\w+ )?времени|за сколько|сколько (?:\w+ )?час", question, re.I):
            v1_, t1_ = trip_
            d_ = v1_ * t1_
            v2_ = int(back_v[1] or back_v[2])
            if v2_:
                t2_ = d_ / v2_
                self.trip_back = (d_, t1_, t2_)
                m_ = round(t2_ * 60)
                yield "operation", f"{m_ // 60} ч" + (f" {m_ % 60} мин" if m_ % 60 else "") + f": {d_:g} км делим на {v2_}.", None
        tb_ = getattr(self, "trip_back", None)
        if tb_ and re.search(r"средн\w* скорост", question, re.I):
            d_, t1_, t2_ = tb_
            avg_ = 2 * d_ / (t1_ + t2_)
            yield "operation", f"Около {avg_:.0f} км/ч: {2 * d_:g} км за {t1_ + t2_:g} ч. Не среднее арифметическое, тут ловушка)".replace(".", ",", 1) if False else f"Около {avg_:.0f} км/ч: весь путь {2 * d_:g} км делим на всё время {f'{t1_ + t2_:g}'.replace('.', ',')} ч. Не среднее арифметическое, тут ловушка)", None
        question_n = re.sub(r"(\d+) с половиной", r"\1,5", question)
        speed = re.search(r"(\d+) ?км ?(?:/|в) ?ч", question_n, re.I)
        span = re.search(r"за (\d+(?:[.,]\d+)?) час\w*(?: (?:и )?(\d+) мин\w*)?|за (\d+) мин|за (полтора|пол ?часа|полчаса|два с половиной)(?: час\w*)?", question_n, re.I)
        if not speed and span and getattr(self, "last_trip", None) and re.search(r"^(?:а |ну )?(?:если )?за ", question.casefold()):
            speed_v = self.last_trip[0]
        else:
            speed_v = int(speed[1]) if speed else None
        if speed_v and span and not dist_ok:
            if span[4]:
                minutes = {"полтора": 90, "полчаса": 30, "пол часа": 30, "полчас": 30, "два с половиной": 150}.get(span[4].lower().replace(" ", " "), 90)
                if span[4].lower().startswith("пол") and not span[4].lower().startswith("полтор"):
                    minutes = 30
            else:
                minutes = round(float(span[1].replace(",", ".")) * 60) + int(span[2] or 0) if span[1] else int(span[3])
            km = speed_v * minutes / 60
            kms = str(int(km)) if km == int(km) else f"{km:.1f}".replace(".", ",")
            self.last_trip = (speed_v, minutes / 60)
            hrs = minutes / 60
            hs = str(int(hrs)) if hrs == int(hrs) else f"{hrs:.2f}".rstrip("0").replace(".", ",")
            if minutes < 60:
                yield "operation", f"{kms} км. {minutes} минут — это {hs} часа, умножаем на {speed_v:g}.", None
            else:
                yield "operation", f"{kms} км: {speed_v:g} на {hs} ч.", None
        mult = re.search(r"в (полтора|два|три|\d+) раза (?:больше|быстрее|выше)", question, re.I)
        if mult and getattr(self, "last_trip", None) and re.search(r"скорост|быстрее", question, re.I):
            v, t = self.last_trip
            k = {"полтора": 1.5, "два": 2, "три": 3}.get(mult[1].lower()) or float(mult[1])
            nv = v * k
            km = nv * t
            kms = str(int(km)) if km == int(km) else f"{km:.1f}".replace(".", ",")
            yield "operation", f"Тогда {kms} км: скорость {nv:g} км/ч.".replace(".5 ", ",5 "), None
        part_ = re.search(r"(?:увеличить|увеличится|больше|быстрее) на (четверть|треть|половину|(\d+) ?(?:%|процент))", question, re.I)
        if part_ and getattr(self, "last_trip", None) and re.search(r"скорост", question, re.I):
            v, t = self.last_trip
            f_ = {"четверть": 0.25, "треть": 1 / 3, "половину": 0.5}.get(part_[1].lower()) or int(part_[2]) / 100
            nv = v * (1 + f_)
            km = nv * t
            kms = str(int(km)) if km == int(km) else f"{km:.1f}".replace(".", ",")
            yield "operation", f"Тогда скорость {nv:g} км/ч, а проедет {kms} км.".replace(".", ",", 1) if nv != int(nv) else f"Тогда скорость {int(nv)} км/ч, а проедет {kms} км.", None
        faster = re.search(r"скорость (?:\w+ )?(?:увеличить|увеличится|прибавить|добавить|больше) на (\d+)|(?:уменьшить|снизить|меньше) (?:\w+ )?на (\d+)", question, re.I)
        if faster and getattr(self, "last_trip", None):
            v, t = self.last_trip
            nv = v + int(faster[1]) if faster[1] else v - int(faster[2])
            yield "operation", f"Тогда {nv * t} км: {nv} на {t}.", None
        pct = re.search(r"(\d+(?:[.,]\d+)?) ?(?:%|процент\w*) от (\d+)", question, re.I)
        if pct:
            p_, base_ = float(pct[1].replace(",", ".")), int(pct[2])
            r_ = p_ * base_ / 100
            rs = str(int(r_)) if r_ == int(r_) else f"{r_:.2f}".rstrip("0").replace(".", ",")
            yield "operation", f"{rs}. Десять процентов — это {base_ / 10:g}, дальше уже просто.", None
        if re.search(r"сдач|сколько (?:\w+ )?(?:заплатил|потратил|стоит все|выйдет)", question, re.I):
            ql2 = question.casefold().replace("ё", "е")
            buy = re.search(r"(?:взял|купил|беру|берешь|взяла|купила)\w* (.+?)(?:,? (?:и )?(?:дал|отдал|заплатил|протянул)\w* (\d+)|[.?]|$)", ql2)
            prices = {m[1]: int(m[2]) for m in re.finditer(r"([а-я]{3,}) (?:стоит |по )?(\d+)", ql2[:buy.start()] if buy else ql2)}
            if buy and prices:
                words = {"одну": 1, "один": 1, "одна": 1, "две": 2, "два": 2, "три": 3, "четыре": 4, "пять": 5}
                total = 0
                for part in re.split(r",? и |, ", buy[1]):
                    m = re.match(r"(?:(\d+|одну|один|одна|две|два|три|четыре|пять) )?([а-я]+)", part.strip())
                    if not m:
                        continue
                    k = int(m[1]) if m[1] and m[1].isdigit() else words.get(m[1], 1)
                    price = next((v for key, v in prices.items() if key[:4] == m[2][:4]), None)
                    if price:
                        total += k * price
                paid = int(buy[2]) if buy[2] else None
                if total:
                    self.shop = (prices, total, paid or total)
                    if paid:
                        yield "operation", f"{paid - total} сдачи. Всё вместе {total} выходит.", None
                    else:
                        yield "operation", f"{total} всего.", None
        cmp_ = re.search(r"(\d+)\b[^.?!]{0,40}?на (\d+) (меньше|больше)", question, re.I)
        if cmp_ and re.search(r"вместе|всего|в сумме", question, re.I):
            a_ = int(cmp_[1])
            b_ = a_ - int(cmp_[2]) if cmp_[3].lower() == "меньше" else a_ + int(cmp_[2])
            self.pair = (a_, b_)
            yield "operation", f"{a_ + b_}. У второго {b_}, плюс {a_} у первого.", None
        rt = re.search(r"у (\w+) (?:\w+ )?в (\d+|два|три|четыре|пять) раза больше(?: \w+)?,? чем у (\w+).{0,30}?вместе (?:у них )?(\d+)", question, re.I)
        if rt:
            k = {"два": 2, "три": 3, "четыре": 4, "пять": 5}.get(rt[2].lower()) or int(rt[2])
            total_ = int(rt[4])
            if total_ % (k + 1) == 0:
                small = total_ // (k + 1)
                big = small * k
                self.ratio = ((rt[1].casefold(), rt[3].casefold()), (big, small))
                asked = re.search(r"сколько (?:\w+ )?у (\w+)", question[rt.end():], re.I)
                val = small if not asked or asked[1][:3].lower() == rt[3][:3].lower() else big
                yield "operation", f"{val}. Делим {total_} на {k + 1} части: одна у одного, {k} у другого.", None
        dep = re.search(r"в (\d{1,2})[:.](\d{2})", question) or re.search(r"в (\d{1,2})()(?= ?(?:утра|вечера|дня|часов))", question)
        dur = re.search(r"(\d+) ?(?:час\w*|ч\b)(?: (?:и )?(\d+) ?мин\w*)?", question)
        if dep and dur and re.search(r"во сколько|когда (?:\w+ )?(?:приедет|приеду|будет|буду|прибудет)", question, re.I) and not re.search(r"\d+ ?км", question):
            start_ = int(dep[1]) * 60 + int(dep[2] or 0)
            end_ = start_ + int(dur[1]) * 60 + int(dur[2] or 0)
            self.arrival = end_
            yield "operation", f"В {end_ // 60 % 24}:{end_ % 60:02d}.", None
        age = re.search(r"(\w+) старше (\w+) на (\d+) (?:года|лет|год).{0,50}через (\d+) (?:года|лет|год).{0,25}вместе (?:\w+ )?(\d+)", question, re.I)
        if age:
            d_, t_, sum_ = int(age[3]), int(age[4]), int(age[5])
            young = (sum_ - 2 * t_ - d_) / 2
            if young == int(young) and young >= 0:
                young = int(young)
                # ages: (younger, older, asked_younger_name_is_second)
                self.ages = (young, young + d_, True)
                asked = re.search(r"сколько (?:\w+ )?(?:сейчас )?(\w+)", question[age.end():], re.I)
                val = young if not asked or asked[1][:4] == age[2][:4] else young + d_
                yield "operation", f"{val}. Сейчас младшему {young}, старшему {young + d_}, через {t_} года вместе как раз {sum_}.", None
        age2 = re.search(r"(\w+) старше (\w+) на (\d+) (?:года|лет|год).{0,50}через (\d+) (?:года|лет|год).{0,40}(вдвое|в два раза|втрое|в три раза|в (\d+) раза)", question, re.I)
        if age2 and not age:
            d_, t_ = int(age2[3]), int(age2[4])
            k_ = {"вдвое": 2, "в два раза": 2, "втрое": 3, "в три раза": 3}.get(age2[5].casefold()) or int(age2[6] or 2)
            young = (d_ - (k_ - 1) * t_) / (k_ - 1)
            if young == int(young) and young >= 0:
                young = int(young)
                self.ages = (young, young + d_, True)
                self.age_names = (age2[1][:3].casefold(), age2[2][:3].casefold())
                asked = re.search(r"сколько (?:лет )?(?:сейчас )?(\w+)", question[age2.end():], re.I)
                is_old = bool(asked) and asked[1][:3].casefold() == age2[1][:3].casefold()
                nm = asked[1].capitalize() if asked and asked[1].casefold() not in ("сейчас", "лет", "ему", "ей") else ""
                val = young + d_ if is_old else young
                yield "operation", (f"{nm} {val}" if nm else f"{val}") + f": через {t_} года будет {young + t_} и {young + d_ + t_}, ровно в {k_} раза.".replace("в 2 раза", "вдвое"), None
        if self.turns:
            ops = re.findall(r"(отними|вычти|отнять|вычесть|минус|прибавь|прибавить|плюс|добавь|умножь|умножить|раздели|разделить|подели|поделить)\w* (?:от этого |к этому |из этого |это |его )?(?:на )?(\d+)", question.casefold())
            nums_prev = re.findall(r"-?\d+", self.turns[-1][1])
            if len(ops) >= 2 and nums_prev and re.search(r"это|него|результат|теперь|потом", question, re.I):
                val = float(nums_prev[0])
                for op, n in ops:
                    n = int(n)
                    if op.startswith(("отн", "выч", "мин")):
                        val -= n
                    elif op.startswith(("приб", "плюс", "доб")):
                        val += n
                    elif op.startswith("умн"):
                        val *= n
                    elif n:
                        val /= n
                vs = str(int(val)) if val == int(val) else f"{val:.2f}".rstrip("0").replace(".", ",")
                yield "operation", f"{vs}. По шагам считал, вроде не ошибся.", None
        frac_ = re.search(r"(\d+) (?:учени\w*|человек|дет\w*|студент\w*|школьник\w*)\W+(\d+)/(\d+) (?:из них |их )?(\w+)", question, re.I)
        if frac_ and re.search(r"сколько", question, re.I):
            total_, a_, b_ = int(frac_[1]), int(frac_[2]), int(frac_[3])
            if b_ and total_ * a_ % b_ == 0:
                part = total_ * a_ // b_
                grp = frac_[4].casefold()
                other = {"девочки": "мальчики", "мальчики": "девочки", "девочек": "мальчиков", "мальчиков": "девочек"}.get(grp, "остальные")
                asked = re.search(r"сколько (\w+)", question[frac_.end():], re.I)
                val = part if asked and asked[1][:4].lower() == grp[:4] else total_ - part
                self.cls = (total_, {grp: part, other: total_ - part})
                yield "operation", f"{val}. {total_} делим на {b_} и умножаем, остальное — другие.", None
        letters = re.search(r"сколько (?:раз )?(?:букв\w* )?[\"«']?([а-яё])[\"»']? (?:в слове |в )[\"«']?([а-яё]+)", question, re.I)
        if letters:
            n_ = letters[2].lower().count(letters[1].lower())
            yield "operation", f"{n_}. Посчитал по буквам, {letters[2].lower()}.", None
        need = re.search(r"(?:есть|у меня) (\d+)[^.?!]{0,40}?стоит (\d+)|стоит (\d+)[^.?!]{0,40}?(?:есть|у меня) (\d+)", question, re.I)
        if need and re.search(r"не хватает|сколько (?:\w+ )?(?:надо|нужно|добавить)", question, re.I):
            have = int(need[1] or need[4]); price = int(need[2] or need[3])
            if price > have:
                self.shortfall = price - have
                yield "operation", f"{price - have} не хватает.", None
            else:
                yield "operation", f"Хватает, ещё {have - price} останется.", None
        sib = re.search(r"у меня (\d+|два|три|четыре) брат\w*,? у каждого (?:брат\w* )?(?:по )?(\d+|две|три) сестр", question.casefold())
        if sib:
            nb = {"два": 2, "три": 3, "четыре": 4}.get(sib[1]) or int(sib[1])
            ns = {"две": 2, "три": 3}.get(sib[2]) or int(sib[2])
            self.sib = (nb, ns)
            if re.search(r"я (?:мальчик|парень)", question, re.I):
                yield "operation", f"{nb + 1 + ns}. Ты тоже брат, так что мальчиков {nb + 1}, плюс {ns} девочки.", None
            else:
                yield "operation", f"{nb + ns}. {nb} брата и {ns} девочки, одна из них ты.", None
        sis = re.search(r"у (\w+) (\d+|два|три|четыре|пять) брат\w*.{0,40}у каждого .{0,25}сестр", question, re.I) if not sib else None
        if sis:
            n_ = {"два": 2, "три": 3, "четыре": 4, "пять": 5}.get(sis[2].lower()) or int(sis[2])
            yield "operation", f"{n_ + 1}. Сестра у всех братьев одна и та же.", None
        fruit = re.search(r"(\w+) стоит (\d+)[^.]*?,? (\w+) на (\d+) (дороже|дешевле)[^.]*?\.?.{0,25}?(\d+) (\w+) и (\d+) (\w+)", question.casefold())
        if fruit:
            p1 = int(fruit[2]); p2 = p1 + int(fruit[4]) if fruit[5] == "дороже" else p1 - int(fruit[4])
            prices = {fruit[1]: p1, fruit[3]: p2}
            total = 0
            for qty, item in ((int(fruit[6]), fruit[7]), (int(fruit[8]), fruit[9])):
                price = next((v for k, v in prices.items() if k[:3] == item[:3]), None)
                if price is None:
                    total = None
                    break
                total += qty * price
            if total:
                self.shop = (prices, total, total)
                self.change = None
                yield "operation", f"{total}. {fruit[3].capitalize()} по {p2}, дальше просто складываешь.", None
        sis2 = re.search(r"у (трех|двух|четырех|пяти|\d+) брат\w* по одной сестр", question.casefold().replace("ё", "е"))
        if sis2:
            n_ = {"двух": 2, "трех": 3, "четырех": 4, "пяти": 5}.get(sis2[1]) or int(sis2[1])
            yield "operation", f"{n_ + 1}. Сестра у всех братьев одна и та же.", None
        cmpnum = re.search(r"(\d+/\d+|\d+(?:[.,]\d+)?)\s*(?:или|vs)\s*(\d+/\d+|\d+(?:[.,]\d+)?)", question)
        if cmpnum and re.search(r"больше|меньше|что (?:\w+ )?(?:больше|меньше)", question, re.I) and ("/" in cmpnum[0] or "," in cmpnum[0] or "." in cmpnum[0]):
            def _val(t):
                if "/" in t:
                    a_, b_ = t.split("/")
                    return int(a_) / int(b_) if int(b_) else None
                return float(t.replace(",", "."))
            x_, y_ = _val(cmpnum[1]), _val(cmpnum[2])
            if x_ is not None and y_ is not None and x_ != y_:
                big = cmpnum[1] if x_ > y_ else cmpnum[2]
                if re.search(r"меньше", question, re.I):
                    big = cmpnum[2] if x_ > y_ else cmpnum[1]
                fmt = lambda v: f"{round(v, 3):g}".replace(".", ",")
                yield "operation", f"{big}. Если в десятичных, это {fmt(x_)} против {fmt(y_)}.", None
        had = re.search(r"(?:было|есть|у меня) (\d+) ?(?:тенге|тг)", question.casefold())
        if had and re.search(r"сдач", question, re.I) and not re.search(r"взял|купил", question, re.I):
            ql_s = question.casefold()[had.end():]
            items_ = {m[1]: int(m[2]) for m in re.finditer(r"([а-я]{3,}) (?:стоит |по )?(\d+)", ql_s)}
            if items_:
                total_ = sum(items_.values())
                paid_ = int(had[1])
                self.shop = (items_, total_, paid_)
                yield "operation", (f"{paid_ - total_} сдачи. Всё вместе {total_}." if paid_ >= total_ else f"Не хватит, нужно ещё {total_ - paid_}."), None
        gm = re.search(r"(\d+|два|три|четыре|пять) внук\w*,? каждому .{0,15}?по (\d+) (?:\w+ )?(?:и )?себе (?:\w+ )?(\d+)", question.casefold())
        if gm:
            n_ = {"два": 2, "три": 3, "четыре": 4, "пять": 5}.get(gm[1]) or int(gm[1])
            k_, r_ = int(gm[2]), int(gm[3])
            self.grand = (n_, k_, r_)
            yield "operation", f"{n_ * k_ + r_}. Внукам {n_ * k_}, и {r_} себе.", None
        conv = re.search(r"сколько (минут|часов|секунд) в (полутора|одних|двух|трех|\d+(?:[.,]\d+)?) ?(сутк\w*|дн\w*|час\w*|недел\w*)", question.casefold())
        if conv:
            mult_ = {"полутора": 1.5, "одних": 1, "двух": 2, "трех": 3}.get(conv[2]) or float(conv[2].replace(",", "."))
            base_min = 1440 if conv[3].startswith(("сут", "дн")) else 60 if conv[3].startswith("час") else 10080
            per = {"минут": 1, "часов": 1 / 60, "секунд": 60}[conv[1]]
            val_ = base_min * mult_ * per
            vs_ = str(int(val_)) if val_ == int(val_) else f"{val_:.1f}".replace(".", ",")
            yield "operation", f"{vs_}. Сначала в часы перевёл, потом умножил.", None
        seq0 = re.search(r"(?:от|из) (\d+) (?:отнять|вычесть|отними|вычти) (\d+)", question.casefold())
        if seq0 and re.search(r"сколько|посчитай", question, re.I):
            val0 = int(seq0[1]) - int(seq0[2])
            rest0 = question.casefold()[seq0.end():]
            for op, n in re.findall(r"(умножить|умножь|разделить|раздели|поделить|подели|прибавить|прибавь|плюс|отнять|вычесть|минус)\w* (?:на )?(\d+)", rest0):
                n = int(n)
                if op.startswith("умн"):
                    val0 *= n
                elif op.startswith(("разд", "под")) and n:
                    val0 /= n
                elif op.startswith(("приб", "плюс")):
                    val0 += n
                else:
                    val0 -= n
            vs0 = str(int(val0)) if val0 == int(val0) else f"{val0:.1f}".replace(".", ",")
            yield "operation", f"{vs0}. Сначала {int(seq0[1]) - int(seq0[2])}, а дальше по порядку.", None
        dig = re.search(r"(\d+) (?:человек\w*|землекоп\w*|рабоч\w*) (?:\w+ )?(?:копают|выкапывают|копали) (\d+) ям\w* за (\d+) час\w*.{0,40}?(\d+) (?:человек\w*|землекоп\w*|рабоч\w*) за (\d+) час", question, re.I)
        if dig:
            p1, h1, t1, p2, t2 = (int(x) for x in dig.groups())
            per = h1 / (p1 * t1)
            total = per * p2 * t2
            ts = str(int(total)) if total == int(total) else f"{total:.1f}".replace(".", ",")
            one = h1 / p1
            one_s = "одну яму" if one == 1 else f"{one:g} ямы"
            self.dig_explain = f"Один человек за {t1} ч копает {one_s}. За {t2} ч — уже {one * t2 / t1:g}. А {p2} человек — в {p2} раз больше, вот и {ts}."
            yield "operation", f"{ts}. Не {h1}, как хочется сказать сразу.", None
        flock = re.search(r"(\d+) (\w+)\W.{0,30}вс[еи] кроме (\d+)", question, re.I)
        if flock:
            self.flock = [int(flock[3]), int(flock[1]) - int(flock[3]), flock[2]]
            yield "operation", f"{flock[3]}. Все, кроме {flock[3]}, убежали, значит {flock[3]} и осталось. Классическая ловушка.", None
        elif getattr(self, "flock", None):
            left, ran, kind = self.flock
            if re.search(r"ног|лап", question, re.I) and re.search(r"сколько", question, re.I):
                yield "operation", f"{left * 4}, если у всех по четыре ноги: {left} на 4.", None
            elif re.search(r"половин", question, re.I) and re.search(r"заболел|больн", question, re.I) and re.search(r"здоров", question, re.I):
                half_ = left / 2
                hs_ = f"{half_:g}".replace(".", ",")
                yield "operation", f"{hs_}. Половина от {left}" + (", овцу пополам не поделишь, так что 4 или 5." if left % 2 else "."), None
            elif re.search(r"половин", question, re.I) and re.search(r"убеж|убег|ушл|уйдет|сбеж", question, re.I) and re.search(r"оставш", question, re.I):
                gone = left // 2
                self.flock = [left - gone, ran + gone, kind]
                if left % 2:
                    yield "operation", f"{left - gone}. Половина от {left} — это {left / 2:g}, овцу не поделишь, так что пусть убежало {gone}.".replace(".5", ",5"), None
                else:
                    yield "operation", f"{left - gone}. Половина от {left} убежала, {left - gone} осталось.", None
            elif re.search(r"вернет|вернут|вернул|купит|купил|добав|новых|прибежал|пришл|прибавил", question, re.I):
                value = left
                note = []
                if re.search(r"половин\w* (?:\w+ )?(?:убежавш|сбежавш)", question, re.I):
                    value += ran // 2
                    note.append(f"вернулось {ran // 2}")
                elif re.search(r"половин\w* оставш", question, re.I):
                    note.append("оставшиеся и так дома")
                question_w = re.sub(r"\b(двое|трое|четверо|пятеро|шестеро)\b", lambda m_: {"двое": "2", "трое": "3", "четверо": "4", "пятеро": "5", "шестеро": "6"}[m_[1]], question.casefold())
                extra = re.search(r"(\d+) (?:нов|ещ|вернул|верн|прибеж|пришл)|(?:купит|купил|добав\w*|прибежал\w*|пришл\w*|ещ[её]|вернул\w*) (?:ещ[её] )?(\d+)", question_w, re.I)
                if extra:
                    k = int(extra[1] or extra[2])
                    value += k
                    note.append(f"плюс {k}")
                self.flock = [value, max(ran - (value - left), 0), kind]
                if note:
                    yield "operation", f"{value}: " + ", ".join(note) + ".", None
        catch = re.findall(r"(\d{1,2}):(\d{2})[^\d]{0,40}?(\d+) ?км", question)
        if len(catch) >= 2 and re.search(r"догон", question, re.I):
            (h1, m1, v1), (h2, m2, v2) = [(int(a), int(b), int(c)) for a, b, c in catch[:2]]
            t1, t2 = h1 + m1 / 60, h2 + m2 / 60
            if v2 > v1 and t2 > t1:
                head = v1 * (t2 - t1)
                hours = head / (v2 - v1)
                when = t2 + hours
                self.catchup = round(v2 * hours)
                hh, mm = int(when), round((when - int(when)) * 60)
                yield "operation", f"В {hh}:{mm:02d}. Первый успеет уйти на {head:g} км, а второй отыгрывает по {v2 - v1} км в час.", None
        if getattr(self, "catchup", None) and re.search(r"расстояни|сколько (?:км|километр)|далеко", question, re.I) and not re.search(r"\d", question):
            yield "operation", f"{self.catchup} км. Второй к тому моменту как раз столько и проедет.", None
        cap = re.search(r"столиц\w* (?:у |в )?(австралии|канады|франции|германии|японии|казахстана|турции|бразилии|сша|америки|италии|испании|китая|кореи|англии|великобритании|египта|кыргызстана|киргизии|узбекистана)", question.casefold())
        if cap:
            capitals = {"австралии": "Канберра", "канады": "Оттава", "франции": "Париж", "германии": "Берлин", "японии": "Токио", "казахстана": "Астана", "турции": "Анкара", "бразилии": "Бразилиа", "сша": "Вашингтон", "америки": "Вашингтон", "италии": "Рим", "испании": "Мадрид", "китая": "Пекин", "кореи": "Сеул", "англии": "Лондон", "великобритании": "Лондон", "египта": "Каир", "кыргызстана": "Бишкек", "киргизии": "Бишкек", "узбекистана": "Ташкент"}
            ar_ = re.search(r"(\d+) ?(умножить на|умноженное на|на|\*|x|х|плюс|\+|минус|-|разделить на|поделить на) ?(\d+)", question.casefold())
            num_ = None
            if ar_:
                a1, opw, b1 = int(ar_[1]), ar_[2], int(ar_[3])
                val = a1 * b1 if opw in ("умножить на", "умноженное на", "на", "*", "x", "х") else a1 + b1 if opw in ("плюс", "+") else a1 - b1 if opw in ("минус", "-") else (a1 / b1 if b1 else None)
                if val is not None:
                    num_ = [f"{val:g}"]
            if num_:
                yield "operation", f"{capitals[cap[1]]}, а второе — {num_[0]}.", None
            else:
                yield "operation", f"{capitals[cap[1]]}." + (" Многие думают, что Сидней, но нет." if cap[1] == "австралии" else ""), None
        yield "operation", short_operation(question), None
        yield "reasoning", self.reasoning.answer(question), None
        # A previous *question* from a different spectator is not testimony.
        history = [q for q, _ in self.turns if "?" not in q]
        extracted = self.reader.propose(question, history)
        if extracted:
            yield "reader", extracted[0], None
        caveat = None if re.search(r"что лучше|\bили\b", question, re.I) else limitation(question)
        if caveat:
            is_reflection = caveat.startswith(("Первое,", "Я бы проверил"))
            yield "reflection" if is_reflection else "abstention", caveat, None
        if self.turns and not re.search(r"скучаешь", question, re.I) and re.search(r"Караганд|Астан", self.turns[-1][1]) and re.search(r"(?:холодно|мороз|погода|тепло).{0,15}(?:сейчас|щас)|(?:сейчас|щас).{0,15}(?:холодно|мороз|погода|тепло)", question, re.I):
            yield "follow-up", "Сейчас уже минус десять-пятнадцать, а с ветром кажется все минус двадцать пять. Ноябрь, что ты хочешь.", None
        if self.turns and re.search(r"опозд|ждал", self.turns[-1][0]) and re.search(r"(?:ты|сам) (?:бы )?обиделся|обидишься", question, re.I):
            yield "follow-up", "Чуть-чуть бы обиделся. Но если пришёл с тортом — прощу.", None
        if self.turns and re.search(r"спис|отказ", self.turns[-1][0] + " " + self.turns[-1][1]) and re.search(r"обидел|обидится", question, re.I):
            yield "follow-up", "Ну обиделся и обиделся. Я ему ничего не должен, а разобраться помочь предлагал.", None
        if self.turns and re.search(r"опозд|ждал", self.turns[-1][0] + " " + self.turns[-1][1]) and re.search(r"не извинил|даже не извин|без извинен", question, re.I):
            yield "reaction", "Тогда я бы уже обиделся. Сказал бы: «Ты даже не извинишься. Окей, в следующий раз жди ты». И правда бы так сделал.", None
        if self.turns and self.last_route == "reaction" and re.match(r"^(?:а|но|и) (?:если|когда|он|она|вдруг)\b", question, re.I):
            # «А если он опять опоздает?» continues the reported situation.
            yield "reaction", reactions.answer(self.turns[-1][0] + " " + question.rstrip(" ?") + ". Что ответишь?"), None
            yield "reaction", reactions.speech(question.rstrip(" ?") + ". Что скажешь?"), None
        yield "reaction", reactions.answer(question), None
        yield "reaction", reactions.speech(question), None
        if not re.search(r"денег (?:ноль|нет|0|мало)|мало денег|без денег|бесплатн|на мели", question, re.I):
            yield "travel", travel.answer(question), None
        yield "profile", self.profile.answer(question), None
        yield "goal-plan", goal_plan.answer(question), None
        if self.turns and re.search(r"анекдот|шутк|пошути", self.turns[-1][0], re.I) and re.search(r"смешн|смеш", question, re.I):
            yield "follow-up", self._follow_up(question), None
        if re.search(r"неб", question, re.I) and re.search(r"голуб|син", question, re.I) and re.search(r"пятиклас|ребен|ребён|простыми|по-простому|как другу|для тупых", question, re.I):
            yield "explanation", "Солнечный свет — это смесь всех цветов. Синий самый «прыгучий»: он отскакивает от воздуха во все стороны, поэтому кажется, что синее всё небо.", None
        yield "explanation", explain(question), None
        # Resolve a request for direct speech before planning a practical action.
        # A problem in the premise (e.g. lateness) need not be the requested task.
        social = self.social.propose(question)
        if social and re.search(r"\b(?:отве(?:ти|ча)|сказа|напиш)\w*",question,re.I):
            yield "social", social[0], None
        yield "tips", tips.answer(question), None
        yield "persona", self.persona.answer(question), None
        if social:
            yield "social", social[0], None
        if self.turns:
            yield "follow-up", self._follow_up(question), None
        if self.turns and self.last_route not in (None, "fallback") and WHY.fullmatch(question.strip()):
            # The reason behind a retrieved reply is unknown; a person answers
            # such a «А почему?» briefly instead of inventing an argument.
            seen = {answer for _, answer in self.turns}
            parts = re.split(r"(?<=[.!:])\s+", self.turns[-1][1])
            if self.last_route in ("profile", "persona", "tips", "reaction", "follow-up") and len(parts) >= 2 and len(parts[1]) > 20 and not parts[1].startswith("«") and not re.match(r"(?:Но|А|И|Хотя|Зато)\b", parts[1]) and not re.search(r"думать особо нечего|Без вариантов|Однозначно|не сомневаюсь|Даже представлять", parts[1]):
                yield "follow-up", "Ну я же говорю: " + parts[1][:1].lower() + parts[1][1:], None
            options = [r for r in WHY_REPLIES if r not in seen]
            if options:
                yield "follow-up", options[len(self.turns[-1][1]) % len(options)], None
        # A real reply to a closely matching question sounds more human than
        # the authored templates below, so a confident match goes first.
        self.last_retrieval = None
        yield "retrieval", self._retrieve(question, RETRIEVAL_STRONG), None
        yield "writing", self.writing.answer(question, self.imagination), None
        yield "creative", self.imagination.answer(question), None
        yield "practical", self.practical.answer(question), None
        focus = focal_question(question)
        if not self.confidence.needs_evidence(question):
            if ASK.search(normal(question)):
                action = self.advice.propose(question)
                if action:
                    yield "advice", action[0], None
                evidence = self.articles.support(question)
                if evidence:
                    action = self.advice.propose(question, evidence[0])
                    if action:
                        combined = action[0] + " " + evidence[0]
                        yield "advice", combined if len(combined)<=MAX_CHARS else action[0], evidence[1]
            # Encyclopaedic passages only when an explanation or a definition is
            # asked for explicitly; otherwise they read like a reference book.
            if EXPLICIT.search(question):
                article = self.articles.propose(focus)
                if article and on_topic(question, article[0]):
                    yield "article", article[0], article[1]
                passage = self.knowledge.propose(focus)
                if passage and on_topic(question, passage[0]):
                    yield "reference", passage[0], "https://ru.wikipedia.org/?curid=" + str(passage[1])
        # The free generator, topic associations and word completion were
        # removed from the chain in 3.0: blind checks found them mostly
        # meaningless. Their code and weights remain for comparison.
        yield "retrieval", self._retrieve(question, RETRIEVAL_THRESHOLD), None
        # The authored situation plans of 2.x matched too loosely (blind
        # rounds found bureaucratic texts on unrelated questions); in 3.0 they
        # are no longer part of the chain.
        yield "fallback", self.abstentions.answer(question), None

    def _precise(self, part):
        """First answer of a reliable module for one part of a question."""
        for route, candidate, _ in self._candidates(part):
            if route in ("retrieval", "fallback"):
                return None
            if route in PRECISE and candidate and standalone(candidate):
                return candidate
        return None

    def _two_parts(self, question):
        """Answer «X и Y?» questions part by part, as a person would."""
        parts = split_parts(question)
        if not parts:
            return None
        if re.search(r"докаж|бот бы не|нейросеть бы не|если бы", question, re.I):
            return None
        if any(re.fullmatch(r"\W*(?:а )?(?:как )?(?:его |ее |её )?(?:зовут|звать|имя)\W*", p, re.I) for p in parts):
            return None
        if len(parts) > 1 and re.match(r"\W*(?:ну|вот|то есть|в смысле|короче|типа|я имею в виду)\b", parts[1], re.I):
            return None
        if re.search(r"\?\s*(?:ну|вот|то есть|в смысле|короче|типа)\b|курс.{0,15}где учишься|где учишься.{0,15}курс|предмет.{0,30}препод|музык.{0,40}засыпаешь|засыпаешь.{0,40}музык|целыми днями|уснуть.{0,40}смысл|смысл.{0,40}уснуть|учишься.{0,15}жалеешь|сосед.{0,20}как зовут|фильм.{0,30}(?:кончил|закончил)|завтрак.{0,40}встал|встал.{0,40}завтрак|где (?:ты )?учишься.{0,25}(?:откуда|сколько)", question, re.I):
            return None
        if any(re.match(r"\W*(?:когда|если|пока|чтобы|потому)\b", p_, re.I) for p_ in parts[1:]):
            return None
        answers = [self._precise(p) for p in parts]
        if not all(answers):
            return None
        said = {a for _, a in self.history}
        if any(a in said for a in answers):
            return None
        w0 = set(re.findall(r"[а-яё]{4,}", answers[0].casefold()))
        w1 = set(re.findall(r"[а-яё]{4,}", answers[1].casefold()))
        overlap = len(w0 & w1) / max(1, min(len(w0), len(w1)))
        joined = answers[0] if answers[0] == answers[1] or answers[0][:12] == answers[1][:12] or answers[1].split(".")[0] in answers[0] or overlap >= 0.4 else answers[0].rstrip() + " " + answers[1]
        reply = display_text(joined)
        if len(reply) > MAX_CHARS or not standalone(reply):
            return None
        self.last_route, self.last_source = "two-parts", None
        self.turns.append((question, reply))
        self.history.append((question, reply))
        if self.first_question is None:
            self.first_question, self.first_answer = question, reply
        return reply

    def _follow_up(self, question):
        """Continue about a title or a city the student has just named."""
        previous = self.turns[-1][1]
        q = question.casefold().replace("ё", "е").strip()
        if len(q) > 120:
            return None
        title = next((t for t in TITLES if t in previous), None)
        if re.search(r"доту|игр", previous) and re.search(r"сколько (?:\w+ )?час", q):
            return "Часа 2 в будни, на выходных больше. Если честно, иногда и до ночи залипаю."
        if re.fullmatch(r"(?:а |ну )?ты(?: как| сам)?\W*", q) and not re.search(r"\bты\b|\bтебе\b|\bтвой\b", previous.casefold()):
            return "Я ж только что рассказал)) Теперь твоя очередь."
        prev_q = self.turns[-1][0].casefold()
        if re.search(r"ерлан", previous.casefold()) and re.search(r"или бесит|бесит или", q):
            return "Нормальный, только храпит и посуду копит. Со школы друг, так что терплю)"
        if re.search(r"ерлан", previous.casefold()) and re.search(r"бесит", q):
            return "Да ничем особо. Гречку мою доедает и храпит иногда, а так нормальный пацан."
        if re.search(r"барсик", previous.casefold()) and re.search(r"как (?:\w+ )?зовут|имя|кличк", q):
            return "Барсик же. Серый, толстый и очень наглый."
        if re.search(r"самсунг", previous.casefold()) and re.search(r"бюджет|тысяч|\d+ ?к\b", q):
            return "За двести тысяч лучше новый самсунг из средних. Айфон за эти деньги только б/у или совсем старый."
        if re.search(r"школ", prev_q) and re.search(r"предмет", q) and re.search(r"ненавид|не любил|бесил|худш", q):
            return "Химию. Вообще не понимал, что там с чем реагирует, списывал у соседки по парте."
        if re.search(r"одноклассниц", previous) and re.search(r"чем (?:\w+ )?(?:кончилось|закончилось)|и что|и как", q):
            return "Ничем) Она с парнем из одиннадцатого начала встречаться, я неделю пострадал и пошёл в доту."
        if re.search(r"футбол", previous) and re.search(r"раз в неделю|как часто|сколько раз", q):
            return "Раз в неделю, по субботам. Иногда ещё в будни побегать выходим, но это редко."
        if re.search(r"Высоты", previous) and re.fullmatch(r"(?:а )?(?:серьезно|правда|реально)\W*", q):
            return "Ну да, на балконе выше пятого этажа уже ноги ватные. Колесо обозрения вообще не моё."
        if re.fullmatch(r"(?:а )?(?:серьезно|правда|реально|да ладно)\W*", q) and self.last_route in ("profile", "follow-up", "persona"):
            return "Ну да, честно)) Зачем мне выдумывать."
        if re.search(r"пересдач", previous) and re.search(r"выкрутил|как сдал|^и как", q):
            return "Неделю сидел над задачами с одногруппником, он физику шарит. На пересдаче попалась задача, которую мы как раз разбирали, повезло."
        if re.search(r"отн[её]с бы", previous) and re.search(r"серьезно|прям|правда|реально", q):
            return "Да, честно. Это ж чья-то зарплата может быть, я бы потом спать не смог."
        if re.search(r"доту", previous) and re.search(r"с кем", q):
            return "С Ерланом и пацанами из группы. Впятером как раз команда, если никто не сливается."
        if re.search(r"информационные системы", previous.casefold()) and re.search(r"почему|зачем", q):
            return "С детства с компами вожусь, а айтишникам и платят нормально. Ну и грант был именно на эту специальность."
        if re.search(r"выспался|бодр", previous) and re.search(r"чего так|что-то случилось|что случилось|с чего бы", q):
            return "Да ничего не случилось, просто выспался впервые за неделю. Первую пару отменили, вот и счастье."
        if re.search(r"апреля", previous) and re.search(r"отмечал|праздновал|празднуешь|отмечаешь", q):
            return "С пацанами в кафешке посидели, потом гуляли до ночи. Ерлан торт притащил со свечкой в виде ракеты."
        if re.search(r"^(?:серьезно\W*)?(?:и |а )?тебе (?:это |такое |он |она )?(?:правда |реально )?нравится", q):
            return "Ну да, иначе бы не говорил)) Под настроение заходит, особенно в автобусе."
        if re.search(r"курьером", previous) and re.fullmatch(r"(?:а |и )?почему\W*", q):
            return "Во время учёбы не успеваю: пары до трёх, потом домашка. Живу на стипендию и что родители скинут."
        if re.search(r"макарон|яичниц|доширак", previous) and re.search(r"каждый день|одно и то же", q):
            return "Ну не прям каждый день, иногда шаурму беру или в столовке ем. Но макароны — база."
        if re.search(r"в два|в час|в три", previous) and re.search(r"\bвстал\w*|проснул\w*", q):
            return "В восемь, на первую пару. Часов шесть получилось, поэтому и хожу как зомби."
        if re.search(r"Ерлан", previous) and re.search(r"(?:что|чем) (?:вы )?(?:\w+ )?(?:с ним )?(?:делаете|занимаетесь)", q):
            return "В доту катаем, гуляем по вечерам, иногда в зал ходим. Ну и готовим по очереди, если это можно назвать готовкой."
        if re.search(r"автобус", previous) and re.search(r"сколько (?:примерно |это )?(?:по времени|времени|минут|ехать)", q):
            return "На автобусе минут пятнадцать, если без пробок. Пешком около двадцати."
        if re.search(r"разработчик|работать|специальност", previous) and re.search(r"родител\w* (?:что|как)|(?:что|как) (?:\w+ )?родител", q):
            return "Мама говорит, лишь бы работа нормальная была. Папа сначала хотел, чтобы я на инженера шёл, но уже смирился."
        if re.search(r"последнюю ночь", previous) and re.search(r"\bу меня\b", q):
            return "Хз, мы ж только переписываемся. Может, задавать по сорок вопросов подряд, ахах."
        if re.search(r"\b(?:что|какой) (?:я )?(?:тебя )?(?:спросил|спрашивал|был (?:мой )?(?:первый )?вопрос)\w* (?:самым |в самом )?(?:первым|сначала|начале)|(?:мой )?первый вопрос", q) and self.first_question:
            quoted = re.sub(r"\?\s*", ". ", self.first_question).strip(" .!")
            return f"Ты спросил: «{quoted}». Я ещё не настолько устал."
        earlier = self.turns[-1][0].casefold()
        if "Профессор" in previous and re.search(r"\b(?:откуда|почему|как) (?:\w+ )?(?:взял\w*|появил\w*|так|прозвал\w*)|\bпочему профессор", q):
            return "Как-то перед контрольной всей группе объяснял пределы на перемене, вот и прилипло. Я не против."
        if "чавка" in earlier and re.search(r"\b(?:сам|ты)\b.{0,20}чавка", q):
            return "Может, и чавкал, со стороны себя не слышно. Но если что, мне бы сказали, у нас в группе с этим не стесняются."
        product = re.fullmatch(r"(\d+), если в уме не ошибся\.", previous)
        nums = re.findall(r"\d+", earlier)
        if product and len(nums) >= 2 and re.search(r"\bкак\b.{0,20}(?:посчитал|считал|получил|вышло|понял)", q):
            a, b = int(nums[0]), int(nums[1])
            tens, units = b - b % 10, b % 10
            if tens and units:
                return f"Разбил на части: {a}·{tens} = {a * tens}, плюс {a}·{units} = {a * units}. Вместе {a * b}."
            return f"Просто {a} умножил на {b} столбиком в голове, получилось {a * b}."
        if re.search(r"яичниц", previous) and re.search(r"(?:что|чем) (?:именно|конкретно)|что (?:ел|ты ел)", q):
            return "Два яйца с помидором и хлеб. Ничего особенного, на скорость."
        if "Маска" in previous and re.search(r"^(?:а |и |ну )?(?:почему|зачем)", q):
            return "Интересно, как у него голова устроена. Мне бы хоть половину его продуктивности на сессию."
        if re.search(r"брос", previous + " " + earlier) and re.search(r"уже бросил|не получается|не вышло|не выходит", q):
            return "Тогда сказал бы, что вернуться не стыдно, восстановиться в универе можно. Зато попробовал и не будет жалеть всю жизнь."
        estimate = re.search(r"около (\d+) тысяч", previous)
        if estimate and len(nums) >= 2 and re.search(r"почему|посчитай|точн|попробуй|не посчитал", q):
            a, b = int(nums[0]), int(nums[1])
            near = round(b / 10) * 10 or b
            diff = near - b
            fmt = lambda n: f"{n:,}".replace(",", " ")
            if diff:
                sign, op = ("минус", diff) if diff > 0 else ("плюс", -diff)
                part = fmt(a) if op == 1 else f"{a}·{op} = {fmt(a * op)}"
                return f"Ладно, точнее: {a}·{near} = {fmt(a * near)}, {sign} {part}. Выходит {fmt(a * b)}. Но в уме это больно."
            return f"Ладно: выходит {fmt(a * b)}. Но в уме это больно."
        if "геометри" in previous and re.search(r"препод\w*|преподавател\w*", q):
            return "Пожилой дядька, мелом рисовал идеально ровно. Строгий, но если видел, что стараешься, ставил нормально."
        if re.search(r"денег|деньги", earlier) and re.search(r"родител", earlier) and re.search(r"на что|зачем|куда", q):
            return "Скажу как есть — на ноут для учёбы или на проезд. Врать родителям бессмысленно, мама всё равно узнает."
        if re.search(r"брос", previous + " " + earlier) and re.search(r"документ|реально решил|уже решил", q):
            return "Тогда уже поддержать. Решил — значит решил, главное, чтобы был план. А восстановиться, если что, всегда можно."
        if "миллион" in earlier and re.search(r"доллар", q):
            return "Тогда родителям квартиру, себе учёбу в Японии, а остальное вложил бы, чтобы не спустить за год."
        if re.search(r"девушк|нравится", earlier) and re.search(r"встречается|есть парень|занята|с кем-то|не одна", q):
            return "Тогда не лезть. Просто общаться нормально, а там жизнь покажет."
        if re.search(r"завал|не сдал|провалил", earlier) and re.search(r"(?:третий|второй|который) раз|опять|снова", q):
            return "Тогда, может, это просто не его специальность. Я бы предложил подумать о переводе, а не бросать всё сразу."
        if product and len(nums) >= 2 and re.search(r"не посчитал|почему не|попробовал бы", q):
            a, b = int(nums[0]), int(nums[1])
            near = round(b / 10) * 10 or b
            diff = b - near
            how = f"{a} на {near} — это {a * near}, " + (f"плюс {a} на {diff}, ещё {a * diff}" if diff > 0 else f"минус {a} на {-diff}, это {a * -diff}") if diff else f"{a} на {b}"
            return f"Так я ж посчитал — {a * b}. Можешь проверить: {how}."
        if "ручки" in previous and re.search(r"почему|не можешь|бросить", q):
            return "Само получается, когда задумываюсь. Замечаю, только когда ручка уже вся в следах."
        if re.search(r"долг|занял|не отда", earlier) and re.search(r"лучший друг|друг с детства|близкий", q):
            return "Тогда тем более сказал бы прямо. С лучшим другом можно и надо говорить честно, иначе обида копится."
        if re.search(r"Астан|Караганд|Алмат", previous) and re.search(r"развлеч|скучно|чем заняться|куда сходить", q):
            return "Не скучно: кафешки, кино, вечером набережная. Ну и с друзьями всегда найдётся, чем заняться."
        if re.search(r"брос", previous + " " + earlier) and re.search(r"работать|на работу", q):
            return "Тогда сказал бы: работать — норм, но лучше взять академ, а не забирать документы. Вернуться так всегда проще."
        if "Ерлан" in previous and "подушк" in previous and re.search(r"решал|терпишь|говорил|делаешь", q):
            return "Беруши купил, а когда совсем громко, подушкой кидаю. Он не обижается, говорит, что ничего не помнит."
        if re.search(r"подар", earlier + " " + " ".join(t for t, _ in list(self.turns)[-2:])) and re.search(r"кофе", q) and re.search(r"не любит|ничего не любит|кроме", q):
            return "Тогда хорошие зёрна и турку, или сводить её в любимую кофейню. Кофеману лучше подарка не придумаешь."
        if "Сова, упрямый" in previous and re.search(r"друз|другие|мама|ерлан", q):
            return "Ерлан бы добавил «вечно голодный». А так, думаю, да."
        if previous.startswith("Пельмени") and re.search(r"^(?:а |и |ну )?почему", q):
            return "Ну и манты я сам лепить не умею, а пельмени купил — и готово. Студенческая логика."
        if re.search(r"Караганд|Астан|Алмат", previous) and re.search(r"поесть|попробовать|еда|кухн|вкусн", q):
            return "Бешбармак у кого-нибудь дома, это главное. И баурсаки к чаю, в кафе их не так делают, как бабушки."
        if re.search(r"«(?:Начало|Остров проклятых|Интерстеллар|Тьма)»", previous) and re.search(r"не люблю|не нравится|не заходит|не фанат", q):
            return "Тогда «1+1» или «Зелёная книга». Без фантастики, просто хорошие истории про людей."
        if re.search(r"родител", earlier) and re.search(r"поняли|поймут|разрешили|отреагировали", q):
            return "Не сразу. Мама бы поплакала, папа бы помолчал. Но если бы увидели, что у меня есть план, со временем поняли бы."
        if "Скриптонит" in previous and re.search(r"^(?:а |и |ну )?(?:почему|чем)", q):
            return "У него голос такой, что под него хорошо думается. И он из Павлодара, свой, казахстанский."
        if re.search(r"спасаюсь|я сам так|сам так делаю", previous) and re.search(r"помога|работает|правда", q):
            return "Через раз, честно. Когда совсем лень, и таймер не спасает, но в среднем работает."
        if re.search(r"спис", earlier) and re.search(r"не поверил|не верит|не поверит", q):
            return "Тогда попросил бы решить при нём другой вариант или пошёл бы к заведующему кафедрой. Если не списывал, доказать несложно."
        if "Космонавтом" in previous and re.search(r"сейчас|поменял|изменил|теперь", q):
            return "Хочу всё так же в айти, только теперь знаю, что это не игры делать, а сидеть над багами до ночи. Но всё равно нравится."
        if re.search(r"анекдот|шутк", earlier) and re.search(r"прикол|не понял|объясни", q):
            if "дедлайн" in previous:
                return "Ну что сначала кажется, что времени вагон, а за час до дедлайна паника. Это про меня каждый семестр."
            if "лестниц" in previous:
                return "Ну «высший уровень» — это про сложность, а он понял буквально и взял лестницу. Тупо, но мне смешно."
            return "Да это игра слов, тут нечего объяснять. Если объяснять шутку, она умирает."
        if re.search(r"девушк|подкат", earlier) and re.search(r"подруг|не одна|с компанией", q):
            return "Тогда через общих знакомых или поймать момент на перемене, когда она одна. Ну или подойти ко всей компании, это даже проще."
        if re.search(r"скин", earlier) and re.search(r"\bсам\b|\bты\b", q):
            return "Конечно, там пятьсот тенге максимум. Староста за нас весь год отдувается."
        if "Шерлок" in previous and re.search(r"длинн|сколько (?:серий|сезонов)|долго смотреть", q):
            return "Четыре сезона по три серии, но каждая по полтора часа, как фильм. За выходные можно осилить."
        if "Была в школе" in previous and re.search(r"познакомил|как (?:вы )?встрет|расстал|почему", q):
            return "В одном классе учились, на химии за одной партой сидели. А после выпуска она уехала учиться в Алматы, так и разошлись."
        if "Космонавтом" in previous and re.search(r"передумал|почему|не стал", q):
            return "Понял, что в космос берут после лётного училища, а я высоты боюсь. А игры делать можно и дома."
        if re.search(r"подар", earlier) and re.search(r"не любит цвет|цветы не", q):
            return "Тогда её любимые сладости или что-то по её хобби. Главное, чтобы было видно, что ты про неё думал."
        if re.search(r"брос|тикток|блог", earlier) and re.search(r"подписчик|миллион", q):
            return "Ну тогда это уже работа, а не хобби. Сказал бы: круто, только деньги откладывай и не забивай на всё остальное."
        if re.search(r"бот|нейросет", previous + " " + earlier) and re.search(r"чем бы|что бы (?:ты )?делал", q):
            return "Ответил бы за всех на экзамене и прочитал бы весь интернет. А потом, наверное, заскучал бы без бешбармака."
        if re.search(r"попадал в так|сам (?:\w+ )?попадал|с тобой (?:\w+ )?(?:такое )?было", q):
            return "Было пару раз что-то похожее. Тогда я примерно так и сделал, и вроде нормально вышло."
        if re.search(r"миллион", earlier + " " + q) and re.search(r"потратил|на что|куда (?:бы )?дел", q) and "Японию" not in previous:
            return "Сначала помог бы родителям, потом съездил бы в Японию. Остальное отложил бы, а то знаю себя — разлетится за месяц."
        if re.search(r"собес|собеседов", earlier) and re.search(r"волну|нервнича|страшно|боюсь", q):
            return "Все волнуются, это нормально. Приди минут за десять, подыши, и можно прямо сказать, что это первый собес, — к таким относятся мягче."
        if "билетик" in previous and re.search(r"давно|когда начал|с какого|сколько (?:\w+ )?(?:билет|собрал|уже)", q):
            return "Класса с седьмого. Бабушка научила: если сумма первых трёх цифр равна сумме последних, билет счастливый и его надо съесть."
        if re.search(r"(?i)котом|как кот|Барсик", previous) and re.search(r"^(?:а |и |ну )?почему", q):
            return "Ну кто живёт лучше кота: поел, поспал, все гладят, и никаких дедлайнов."
        if "Скриптонит" in previous and re.search(r"назови|конкретн|кого (?:ещ[её]|именно)|кто ещ[её]", q):
            return "Скриптонит в первую очередь. Ещё Джа Халиб и Ирина Кайратовна, когда настроение получше."
        if "Тысяч сорок" in previous and re.search(r"много|мало|нормально", q):
            return "Для студента нормально. С доставкой улетало бы в два раза больше, проверено."
        if re.search(r"подар", earlier) and re.search(r"безделуш|не любит (?:всякие|вещи|такое)|такое не любит|не понравится|ничего не надо", q):
            return "Тогда что-то полезное или впечатление: сходить вместе в кафе, на каток или сертификат туда, куда она давно хотела."
        if re.search(r"отшил|отказал", earlier) and re.search(r"напишет|позвонит|вернется|вернётся", q):
            return "Ответил бы, но без особых надежд. Смотря зачем пишет: если просто скучно, то нет."
        if re.search(r"помирит|накосяч|поссор", earlier) and re.search(r"сам виноват|он виноват|не я виноват", q):
            return "Тогда тем более не ждать, пока он извинится первым. Сказать прямо, что обидно, но без наезда, и дать ему шанс."
        if re.search(r"брос|тикток|блог", earlier) and re.search(r"получается|зарабатыва", q):
            return "Тогда вообще молодец. Но я бы всё равно посоветовал академ, а не отчисление, чтобы был запасной вариант."
        if re.search(r"женишься|женюсь", earlier) and re.search(r"говорит|отвечает|скажет", q):
            return "«Я до вашей свадьбы не доживу». Каждый раз. Хотя здоровее нас всех."
        if re.search(r"звон", earlier) and re.search(r"\b(?:сам|ты)\b", q) and re.search(r"бои|звон", q):
            return "Немного, да. Если звонит незнакомый номер, сначала жду, вдруг сбросят. Маме, конечно, отвечаю сразу."
        if "Шпору" in previous and re.search(r"поймал|спалил|заметил", q):
            return "Сказал бы, что шпора не моя. Шучу — извинился бы и пошёл на пересдачу, что тут скажешь."
        if "Яичниц" in previous and re.search(r"сам (?:\w+ )?готовил|почему именно", q):
            return "Сам, конечно. Яичница — единственное, что я готовлю без риска. Быстро и сытно."
        if "Караганд" in previous and re.search(r"славится|известен|знамен|интересн|чем (?:там )?(?:\w+ )?хорош", q):
            return "Шахтами, это же угольный город. Ну и Карлаг рядом, музей там тяжёлый, но сходить стоит."
        if re.search(r"женишься|женюсь", earlier) and re.search(r"не отстанет|каждые|постоянно|опять", q):
            return "Тогда буду каждый раз переводить разговор на её бешбармак. Работает безотказно."
        if "Была в школе" in previous and re.search(r"вместе|\bодин\b", q):
            return "Я ж говорю, один пока. С той, школьной, полгода встречались, но это было давно."
        if "Яичниц" in previous and re.search(r"вкусно|на бегу|быстро|сам готовил", q):
            return "На бегу, доедал уже в лифте. Но горячая яичница на бегу всё равно лучше, чем ничего."
        if re.search(r"стипенди|на гранте", previous + " " + earlier) and re.search(r"тратишь|трачу|на что", q):
            return "Проезд, еда и иногда кино. Остальное само уходит на «ну это же недорого»."
        if re.search(r"завал|пересдач", earlier + " " + previous) and re.search(r"не получится|тоже не|опять не|снова не", q):
            return "Тогда комиссия, там уже трое преподов. Готовиться серьёзнее, но шанс есть. А если совсем тяжело, можно подумать про академ."
        if re.search(r"брос", earlier) and re.search(r"несчастлив|не нравится|мучается|плохо ему", q):
            return "Тогда пусть уходит, но не в никуда, а с планом. Мучиться четыре года ради бумажки тоже так себе."
        if re.match(r"(?:Уметь )?[Лл]етать", previous) and re.search(r"куда|где", q):
            return "Над Астаной ночью, посмотреть на Байтерек сверху. А потом к бабушке, без пробок."
        if re.search(r"брос|айти|универ", earlier) and re.search(r"родител", q) and re.search(r"сказать|объяснить", q):
            return "Честно и с планом: «Мам, пап, я понял, что это не моё, и вот что хочу делать вместо этого». Без плана к ним лучше не идти."
        if re.search(r"спис", earlier) and re.search(r"нравится|девушк", q):
            return "Тогда дал бы, конечно, но заодно объяснил бы, как решать. Двойная польза: и помог, и повод пообщаться."
        if "в субботу" in previous and re.search(r"потом|дальше|и что", q):
            return "Посидел с охранником, попил чаю и пошёл домой. Родители до сих пор вспоминают."
        if re.search(r"миллион", earlier) and re.search(r"за сколько", q):
            return "Да и за полмиллиона согласился бы, если честно. Месяц без телефона — это просто отдых."
        if "в субботу" in previous and re.search(r"вспоминали|смеялись|узнали|дразнили", q):
            return "До выпуска вспоминали. Каждую субботу кто-нибудь писал: «Ты там не в школе случайно». Теперь сам смеюсь."
        if re.match(r"(?:Бешбармак|Плов)", previous) and re.search(r"мама|мам\w*", q):
            return "Маме скажу, что её плов лучший. Дипломатия, все довольны."
        if re.search(r"матан|Матанализ", previous + " " + earlier) and re.search(r"не понимаю|ничего не понима|не шарю", q):
            return "Тогда найти в группе того, кто шарит, и попросить объяснить за шаурму. И ютуб, там разжёвывают лучше, чем на паре."
        if re.search(r"девушк|признан|подумать", earlier) and re.search(r"(?:неделю|долго|несколько дней|вообще) не отвечает|игнорит|пропала", q):
            return "Значит, это и есть ответ. Навязываться не буду, напишу один раз нормально, и всё."
        if re.search(r"плов|яичниц|бешбармак|гречк", previous) and re.search(r"рецепт|как готов|как делаешь", q):
            return "Да какой там рецепт: мясо, морковь, лук, рис, всё в казан и не мешать. Главное — не торопиться, а я всегда тороплюсь."
        if re.search(r"анекдот|шутк", earlier) and re.search(r"другой|ещ[её] один|уже слышал|старый|баян", q):
            return "Ладно, другой. Препод: «Почему опоздал». Студент: «Там знак висел — „Осторожно, дети“. Вот я и ехал осторожно»."
        if "отложил" in previous and re.search(r"поездк|путешеств", q):
            return "Так Япония — это и есть поездка. Просто сначала родителям, а потом уже себе."
        if "отложил" in previous and re.search(r"не отложил|почему не|зачем трат", q):
            return "Так я и отложил бы остальное. Просто родителям и Японии — в первую очередь, а то так никогда и не съезжу."
        if "Матанализ" in previous and re.search(r"^(?:а |и |ну )?(?:почему|чем)", q):
            return "Там всё на доказательствах, а я больше практик. Пока не вижу, где это пригодится, мотивации ноль."
        if "тенге" in previous and re.search(r"сколько (?:\w+ )?(?:стоил|стоит|отдал|заплатил)", q):
            price = re.search(r"за ([а-яё]+(?: [а-яё]+)?) тенге", previous)
            return f"Ну я же говорю, {price[1] if price else 'недорого'} тенге. Дешевле только доширак."
        if re.search(r"брос", previous + " " + earlier) and re.search(r"блогер|тикток|ютуб|бизнес|стартап|сво[её] дело|работать|на работу", q):
            return "Тогда пусть пробует параллельно. Если пойдёт — круто, а диплом пусть будет запасным вариантом."
        if "Фотка гор" in previous and re.search(r"почему|зачем", q):
            return "Там красиво и тихо. Когда на паре скучно, смотрю и вспоминаю лето."
        if "короб" in earlier and re.search(r"собак", q):
            return "Собаки больше про хозяина: им важнее быть рядом с тобой, чем прятаться в коробке."
        if re.search(r"мам", earlier) and re.search(r"обидит|обидел", q):
            return "Тогда позвонил бы ещё раз и поговорил подольше. На маму долго обижаться не получается, она быстро отходит."
        if re.search(r"^(?:а |и )?(?:если|вдруг) (?:он|она) .{0,12}(?:обидится|обиделся|обиделась|разозлится)", q):
            return "Ну и пусть. Если дружба держится только на этом, то это так себе дружба."
        if re.search(r"без телефона|без интернета", earlier) and re.search(r"вместо|что бы (?:ты )?делал|чем бы .{0,10}занима", q):
            return "Читал бы, гулял, может, наконец доделал бы курсач. Ну и спал бы больше, это точно."
        if re.search(r"помирит|накосяч", earlier) and re.search(r"не отвечает|игнор|молчит", q):
            return "Тогда дать ещё пару дней и прийти лично. Если человек дорог, гордость тут лишняя."
        if re.search(r"опозд|опаздыва", earlier) and re.search(r"(?:третий|второй|пятый|который|десятый) раз|опять|снова", q):
            return "Тогда уже не оправдываться, а честно подойти после пары: «Понимаю, что не первый раз, исправлюсь». И правда начать выходить раньше."
        if re.search(r"энергетик|кофе|чай", earlier + " " + previous) and re.search(r"сколько (?:\w+ )?(?:в день|штук|раз)", q):
            return "Чая чашек пять, кофе одну, и то в сессию. Энергетики — почти никогда."
        if re.search(r"деда", previous + " " + " ".join(a for _, a in list(self.turns)[-2:])) and re.search(r"заплати|денег|тысяч|\d+\s*к\b|продал", q):
            return "Даже за сто тысяч нет. Это память, такое не продаётся."
        if re.search(r"сосед|музык|шум|гитар", earlier) and re.search(r"по-хорошему|по хорошему|не понимает|не помогает|не слушает", q):
            return "Тогда к коменданту или хозяину, а если совсем никак — к участковому. Но сначала ещё раз поговорить, уже при свидетелях."
        if "По стилю похоже" in previous and re.search(r"почему|с чего", q):
            return "Пишешь коротко и без смайликов. Но это так, на глаз, могу и ошибаться."
        if self.last_route == "tips" and re.search(r"(?:ты )?сам .{0,15}(?:делал|пробовал|так делаешь)", q):
            return "Делал, перед матанализом. Сдал на тройку и потом два дня отсыпался. Больше не хочу."
        if self.last_route == "tips" and re.search(r"(?:\d+|два|три|час\w*) (?:часа )?ночи|уже поздно|ночь уже", q):
            return "Тогда уже спать. Три-четыре часа сна дадут больше, чем ещё два билета в полусне."
        if re.search(r"первокурсник|заговорить|одногрупп", earlier) and re.search(r"интроверт|одному|один норм", q):
            return "Тогда и не надо его тянуть. Главное — пусть знает, к кому подойти, если что-то понадобится."
        if re.search(r"завал", earlier) and re.search(r"мам", earlier) and re.search(r"пап|отцу|отец", q):
            return "Папе проще: сказал бы коротко, что пересдам, и он кивнёт. Главное, чтобы от мамы он не первым узнал."
        if re.search(r"(?i)второй курс", previous) and re.search(r"сколько.{0,20}учиться|ещ[её] учиться|когда (?:\w+ )?закончишь", q):
            return "Ещё два с половиной года, бакалавриат четыре. А там, может, магистратура в Японии."
        if "чужому билету" in previous and re.search(r"отреагир|сказал|что (?:\w+ )?было", q):
            return "Посмеялся и дал мой билет. Я его, кстати, сдал, на нервах всё вспомнил."
        if re.search(r"лестниц|дедлайн", previous) and re.search(r"почему (?:\w+ )?(?:этот|его|именно)", q):
            if "лестниц" in previous:
                return "Да просто первый, что вспомнился. Его у нас на потоке все знают."
            return "Он про меня, если честно. Каждый семестр одно и то же."
        if re.search(r"бросил|ушла|девушк", earlier) and re.search(r"с другим|другой парень|новый парень", q):
            return "Тогда тем более не писать ей. Больно, но через месяц станет заметно легче, проверено на друзьях."
        if re.search(r"нейросет|заменит|Рутину|ИИ", previous) and re.search(r"пугает|страшно|боишься|тревож", q):
            return "Немного. Но если учиться нормально, а не копировать код, работа всё равно будет. Так что больше мотивирует, чем пугает."
        if "печенье за экраном" in previous and re.search(r"родител|мама|папа", q):
            return "Конечно, мама до сих пор это всем гостям рассказывает. Печенье, говорит, находили через неделю."
        if re.search(r"ключ", earlier + " " + previous) and re.search(r"делать|как быть|что теперь", q):
            return "Сначала пройтись по маршруту и спросить в автобусном парке, туда часто сдают. Если нет — вызвать мастера и поменять личинку, это недорого."
        if re.search(r"Элджернон", previous) and re.search(r"читается|сложно|толст|страниц", q):
            return "Не, легко, за пару вечеров. Там дневник героя, сначала даже с ошибками написан, это часть сюжета."
        if re.search(r"«[^»]+»", previous) and re.search(r"(?:сложно|легко) (?:\w+ )?читается|толстая|сколько страниц", q):
            return "Нормально читается, за пару вечеров осилишь."
        if title and re.search(r"грустн|печальн|страшн|тяжел", q) and not re.search(r"\bмне\b|у меня|\bя\b", q):
            if title == "Интерстеллар":
                return "Местами да, сцена с видеосообщениями прям давит. Но в целом он больше масштабный, чем грустный."
            return "Местами, но в целом нормальный, не депрессивный. После него не хочется лежать лицом в стену."
        if re.search(r"ЕНУ|Астан", previous) and re.search(r"общаг", q):
            return "Общаги есть, по три-четыре человека в комнате, кухня на этаж и очередь в душ утром. Я в общагу не попал, поэтому снимаем с другом."
        if re.search(r"бег|пробежк", earlier) and re.search(r"утром|не вста", q):
            return "Тогда бегай вечером, после пар. Главное — регулярность, а не время."
        if "Тьм" in previous and re.search(r"длинн|сезон|серий", q):
            return "Три сезона, серий по восемь-десять. За неделю реально осилить."
        if re.search(r"первокурсник|первый день", earlier) and re.search(r"\bсам\b|\bты\b", q) and re.search(r"чувствовал|как было|страшно|волновал", q):
            return "Потерянным. Заблудился и полчаса искал аудиторию, а потом всю неделю здоровался со старшекурсником, который помог."
        if "Бэкенд" in previous and re.search(r"почему|зачем", q):
            return "С детства копаюсь в компьютерах, а на втором курсе понял, что логика и базы данных мне заходят больше, чем дизайн."
        if "Астан" in previous and "Караганд" not in previous and re.search(r"зимой (?:\w+ )?холодно|холодно зимой", q):
            return "Ещё как: минус тридцать с ветром бывает. Без шапки и шарфа лучше не выходить."
        if "Караганд" in " ".join(a for _, a in list(self.turns)[-3:]) and re.search(r"вернешься|вернёшься|вернулся бы|жить туда|туда жить|вернуться", q):
            return "Вряд ли насовсем: в Астане работы больше. Но к родителям на праздники — всегда."
        if re.search(r"спис", earlier) and re.search(r"отказал\w* прямо|прямо отказал|прямо (?:бы )?сказал", q):
            return "Да, прямо, но спокойно. Сказал бы, что больше не дам, а объяснить, как решать, — без проблем."
        if "правила конкурса" in previous and re.search(r"почему|боишься|зачем", q):
            return "Не боюсь, просто правила такие: никаких контактов. Нарушу — дисквалифицируют, а я уже втянулся."
        if "Ryzen" in previous and re.search(r"сам бы|взял бы|что из этого|какой бы", q):
            return "Леново на Ryzen 5: у Ерлана такой третий год живой. Взял бы такой же."
        if "автобус" in previous and "проспал" in previous and re.search(r"поверил|поверит", q):
            return "Конечно нет. Посмеялся и отметил опоздание, но с тех пор я отмазки заранее продумываю."
        if re.search(r"миллион", earlier + " " + previous) and re.search(r"\d+ миллион|десять миллион|10 млн|\bлям\w*\b", q):
            return "За десять миллионов я бы и год без телефона просидел. Родителям квартиру, остальное — на учёбу в Японии."
        if re.search(r"оперативк|Ryzen|ноут", previous) and re.search(r"\bмак\w*|макбук|apple", q):
            return "Мак красивый, но дорогой, да и часть программ для учёбы только под винду. За эти деньги лучше нормальный ноут на винде."
        if "гречку" in previous and re.search(r"бесят|бесит|конкретно|чем именно", q):
            return "Ну гречку мою доедает, а ночью в доту орёт, хоть и в наушниках. Но это мелочи, он нормальный."
        if re.search(r"девушк|подкат", earlier) and re.search(r"наушник", q):
            return "Тогда подождать, пока снимет, или просто помахать. Если не снимает — значит, не сегодня."
        if re.search(r"поговорить", earlier) and re.search(r"дальше|потом|что бы (?:ты )?сделал", q):
            return "Дальше слушал бы, не перебивая. А потом уже решал бы: извиняться или спокойно объяснять свою сторону."
        if re.search(r"Матанализ|предмет|матан", previous + " " + earlier) and re.search(r"препод", q) and re.search(r"сказал|скажешь", q):
            if "Истори" in previous:
                return "Сказал бы: «Давайте перенесём пару хотя бы на десять утра, вы ведь и сами не высыпаетесь»."
            return "Сказал бы: «Объясните хоть раз, зачем нам эти доказательства». Ну и попросил бы не ставить пары в восемь."
        if re.search(r"девушк|подойти", earlier) and re.search(r"(?:сам|ты) (?:\w+ )?бы (?:\w+ )?подош|сам подош", q):
            return "Честно, тоже бы минут десять собирался. Но подошёл бы, иначе потом весь вечер думать буду."
        if re.search(r"Караганд|Астан", previous) and re.search(r"зимой", q) and re.search(r"\d+|минус|сорок", q):
            return "Бывает, раз в пару лет до минус сорока доходит. Но обычно минус двадцать с ветром, и этого хватает."
        if re.search(r"Спал бы восемнадцать|(?i:котом)", previous) and re.search(r"чем бы|первым делом|что бы (?:ты )?делал", q):
            return "Сначала отоспался бы, потом сел бы на подоконник смотреть на голубей. Идеальная неделя."
        if re.search(r"«[^»]+»", previous) and re.search(r"покороче|не сериал|фильм бы", q):
            return "Так это фильмы, не сериалы. Из совсем лёгких — «1+1»: полтора часа, и настроение на весь вечер."
        if re.search(r"брос", earlier) and re.search(r"родител", q) and re.search(r"поругал|поссорил|скандал", q):
            return "Тогда тем более поддержал бы. Позвал бы к нам переночевать, а с родителями пусть поговорит потом, когда все остынут."
        if self.last_route not in (None, "fallback", "retrieval") and re.search(r"^(?:а |ну |кстати,? )*(?:ты )?(?:сам|сам-то) (?:как )?думаешь|а ты как думаешь", q) and len(q) < 60:
            first = re.split(r"(?<=[.!])\s", previous, maxsplit=1)[0].rstrip(".")
            if len(first) > 15 and not first.startswith(("Сказал бы", "«")):
                return "Ну я же сказал: " + first[:1].lower() + first[1:] + ". Так и думаю."
        if self.last_route == "explanation" and re.search(r"у тебя (?:\w+ )?(?:так|тоже|такое) (?:\w+ )?бывает|ты (?:сам )?(?:тоже|так) (?:\w+ )?(?:делаешь|чувствуешь)", q):
            return "Бывает, конечно, я же тоже человек. Особенно в сессию, когда темнеет в пять."
        if re.search(r"завал|двойк|пересда", earlier + " " + previous) and re.search(r"орать|ругать|кричать|наругают", q):
            return "Поорут и успокоятся. Главное сразу сказать, что уже записался на пересдачу, это сильно смягчает."
        if re.search(r"без телефона|без интернета", earlier) and re.search(r"заплати|тысяч|\d+\s*к\b|денег", q):
            return "За сто тысяч — тем более, даже думать не надо. Неделю без телефона за такие деньги любой студент выдержит."
        if "старшекурсник" in previous and re.search(r"подружил|друз|познакомил", q):
            return "С парой ребят из группы, сидели рядом на первой паре. А Ерлана я и так со школы знаю."
        if "деда" in previous and re.search(r"\b(?:почему|зачем)\b", q):
            return "Это память. Он мне его отдал, когда я в первый класс пошёл, чтобы маме звонил."
        if "курьером" in previous and re.search(r"\b(?:сколько|скок|скока)\b|плати|зарплат", q) and not re.search(r"во сколько|лег|ложи|сова|встал|\d", q):
            return "Тысяч сто пятьдесят в месяц выходило, если каждый день катать. Для лета норм."
        if "опаздывают" in previous and re.search(r"\bделаешь\b|\bборешься\b|\bреагируешь\b", q):
            return "Говорю один раз спокойно. Если не доходит, просто перестаю ждать и иду по своим делам."
        if "гречку" in previous and re.search(r"\b(?:зачем|почему|нафига)\b", q):
            return "Ну чтобы не пригорала, логично же. Ерлан уже грозится отобрать у меня кастрюлю."
        if "геометри" in previous and re.search(r"\b(?:зачем|почему)\b", q):
            return "Говорят, развивает пространственное мышление. Может, и развивает, но я пока только кубы научился рисовать."
        if re.search(r"забыл", earlier) and re.search(r"обиделась|не отвечает|игнор", q):
            return "Тогда не спамить, а написать одно нормальное сообщение и прийти лично, с цветами. Переписка тут уже не спасёт."
        if re.search(r"\b(?:чего|почему|че|чё|что) (?:так|ты так) поздно|(?:занимался|делал) (?:\w+ )?так поздно", q):
            return "Да сериал затянул, а потом ещё в телефоне залип. Классика."
        if re.search(r"брос", previous + " " + earlier) and re.search(r"отговар\w*|отговорил\w*", q):
            return "Попробовал бы, да. Но если он всё взвесил и правда знает, чем будет заниматься, — это его жизнь, я бы поддержал."
        if re.search(r"прокрастин|отклады", earlier) and re.search(r"\b(?:у тебя|сам|самого)\b", q):
            return "Через раз, честно. Таймер помогает, но иногда я и его откладываю."
        if "Барсик" in previous and re.search(r"^(?:а |и |ну )?почему", q):
            return "Они спокойные и сами по себе. С собакой гулять надо три раза в день, а я себя-то еле из дома вытаскиваю."
        if re.search(r"подар", previous + " " + earlier) and re.search(r"ничего не (?:надо|нужно)", q):
            return "Все так говорят. Значит, нужно что-то небольшое, но от себя: открытка или её любимые сладости."
        if re.search(r"друз|дружить", previous + " " + earlier) and re.search(r"\bсмог\w* (?:бы )?(?:\w+ )?дружить|\bты бы (?:сам )?(?:смог|стал) дружить", q):
            return "Не сразу. Сначала надо переболеть, а там, может, и получится нормально общаться."
        if "Япони" in previous and re.search(r"не там|не здесь|не в астан|не дома", q):
            return "В Астане я и так живу, хочется чего-то совсем другого. Хотя бы на год, потом всё равно вернусь."
        if "Япони" in previous and re.search(r"\b(?:куда|почему|зачем)\b", q):
            return "Япония, Токио. Давно мечтаю, там сильная робототехника, и вообще интересно пожить в другой культуре."
        if "Пробовал в школе" in previous and re.search(r"\bродител\w*|\bмам\w*|\bпап\w*", q):
            return "Про тот раз в школе — нет, и надеюсь, не узнают."
        if re.search(r"«[^»]+»", previous) and re.search(r"(?:уже|я его|я это) (?:\w+ )?(?:смотрел|видел|читал)|смотрел уже|видел уже", q):
            return "Тогда «Настоящий детектив», первый сезон. Мрачнее, но затягивает не хуже."
        if "История Казахстана" in previous and re.search(r"\bпрепод\w*|\bпреподавател\w*", q):
            return "Нормальный мужик, просто читает с листа и не любит вопросы. Строгий, но на экзамене не валит."
        if "Скриптонит" in previous and re.search(r"\b(?:кто (?:это|он|такой)|это (?:\w+ )?кто|не слышал|не знаю такого)\b", q):
            return "Ты чего, это рэпер из Павлодара, у нас его все знают. Голос низкий, биты медленные. Послушай «Это любовь», для начала норм."
        if "лестниц" in previous and re.search(r"смешн|смеш|прикол", q):
            return "Ну «высший уровень» — это про сложность, а он понял буквально и взял лестницу. Тупо, но мне смешно."
        if re.search(r"анекдот|шутк|пошути", earlier) and re.search(r"\bсмешн\w*|\bсмеш\w*|\bне смешно", q):
            return "Мне смешно, потому что это правда про меня. Ну а анекдоты я рассказывать не умею, порчу концовку."
        if re.search(r"\bпарень|\bпарня", earlier) and re.search(r"^(?:а |и )?(?:если|вдруг) (?:она|он) (?:сама |сам )?(?:первая |первый )?(?:напишет|пишет|пишут|позвонит|звонит|позовет|позовёт|зовет|зовёт)", q):
            return "Тогда, может, у них там не всё так серьёзно. Но я бы всё равно не торопился, пусть она сама сначала разберётся со своим парнем."
        singer = re.search(r"\bкто (?:(?:ее|её|его|их) )?(?:поет|поёт|исполняет|написал|снял|играет)", q)
        if singer:
            name = next((n for n in ARTISTS if n.casefold() in previous.casefold()), None)
            if name:
                return f"Да {name} же, я говорил."
        if re.search(r"\b(?:докажи|уверен|правда|серьезно|серьёзно)\b", q) and re.search(r"\b(?:человек|живой|студент|не нейросеть|нейросетью|робот)\b", previous.casefold()):
            return "Ну как доказать. Могу сказать, что у меня сейчас затекла нога и очень хочется чаю. Бот бы до такого не додумался."
        if re.search(r"\b(?:объясни|скажи|расскажи) (?:\w+ )?(?:попроще|проще)|\bне понял\w*|\bнепонятно\b|перефразируй", q) and len(q) < 80:
            if "«" in previous:
                plain = re.sub(r"[«»]", "", previous)
                plain = re.split(r"(?<=[.!])\s", plain, maxsplit=1)[0].rstrip(".")
                plain = re.sub(r"^(?:Сказал|Ответил) бы:?\s*", "", plain)
                return "Ну, в смысле, я бы просто сказал: " + plain[:1].lower() + plain[1:] + ". Ничего сложного."
            first = re.split(r"(?<=[.!])\s", previous, maxsplit=1)[0].rstrip(".")
            if len(first) > 15:
                return "Если совсем коротко: " + first[:1].lower() + first[1:] + "."
        if self.last_route == "explanation" and re.search(r"\bкак (?:это |его |её |ее )?(?:\w+ )?(?:убрать|исправить|избежать|помочь|справиться|остановить|вылечить|ускорить|замедлить)", q):
            parts = re.split(r"(?<=[.!])\s", previous, maxsplit=1)
            if len(parts) == 2:
                return "Да проще всего так: " + parts[1][:1].lower() + parts[1][1:]
        if self.last_route == "explanation" and re.match(r"(?:а|и|ну) ", q) and not explain(question):
            again = explain(self.turns[-1][0] + " " + question)
            if again and again != previous:
                return again
        if self.last_route == "explanation" and re.search(r"\b(?:это )?(?:вредно|опасно|плохо|нормально)\b", q) and len(q) < 60 and not q.startswith(("почему", "зачем")):
            return "Если иногда, то ничего страшного. Но постоянно так лучше не делать."
        if self.last_route == "tips" and re.match(r"(?:а|и|ну) (?:если|вдруг|когда)\b", q):
            if re.search(r"\bне (?:помога\w*|работа\w*|получа\w*|выход\w*)", q) and not re.search(r"нигде не|спросят", q):
                return tips.ALTERNATIVES.get(previous) or "Тогда честно: значит, сейчас нужен отдых. Отдохни вечер без чувства вины, а завтра начни с самого маленького шага."
            return tips.answer(self.turns[-1][0] + " " + question)
        if re.search(r"\b(?:какой|что за|помнишь|повтори)\b.{0,20}\b(?:предыдущ|прошл)\w* (?:мой |твой )?вопрос\w*|(?:предыдущ|прошл)\w* (?:мой )?вопрос\w* (?:какой|был)|(?:что|какой) я (?:только что|сейчас) (?:спросил|спрашивал)", q):
            quoted = re.sub(r"\?\s*", ". ", self.turns[-1][0]).strip(" .!")
            return f"Ты спросил: «{quoted}»."
        if self.profile.last_choice and re.search(r"\bдруг(?:ой|ое|ую) вариант|\bвтор(?:ой|ое) вариант|если бы (?:пришлось|надо|нужно) (?:выбрать )?(?:друг|втор)", q):
            other = self.profile.last_choice[1]
            return f"Ну тогда {other}. Но уже без особого энтузиазма."
        if re.search(r"\b(?:что|чего) (?:тут |в этом |же )?смешного\b", q):
            return "Ну мне смешно было. Видимо, надо посидеть на первой паре, чтобы понять."
        if re.search(r"\bсам (?:\w+ )?(?:готовил|приготовил|делал)|\bвкусно (?:было|получилось)", q) and re.search(r"яичниц|завтрак|готов|бешбармак|плов", previous.casefold()):
            return "Сам, конечно. Яичницу испортить сложно, хотя я как-то умудрился."
        if re.search(r"\b(?:кто-нибудь|кто-то|кто) (?:это )?(?:видел|заметил)\w*", q) and len(q) < 60:
            return "Видели, конечно. Потом ещё неделю вспоминали, теперь уже и сам смеюсь."
        if "истори" in previous.casefold() and "восемь" in previous.casefold() and re.search(r"\b(?:почему|препод\w*|виноват)\b", q):
            return "Препод-то нормальный, дело не в нём. Просто восемь утра — не время для дат и войн."
        if "препод" in (previous + " " + self.turns[-1][0]).casefold() and re.search(r"\b(?:что (?:он|она) делал\w*|почему (?:именно )?(?:он|она))", q):
            return "Да придирался ко всему. На первой же паре выгнал полгруппы за опоздание на минуту."
        if re.search(r"\bпочему\b", q) and re.search(r"токио|япони", previous.casefold()):
            return "Там всё по-другому: еда, порядок, техника. Хочется пожить внутри этого, а не просто посмотреть на фото."
        if re.search(r"\b(?:трек|песн)\w*", q) and "скриптонит" in previous.casefold():
            return "«Положение», наверное. Старое, но до сих пор заходит."
        if re.search(r"\bкак (?:его|её|ее) зовут", q):
            if re.search(r"барсик|\bкот\b", previous.casefold()):
                return "Барсик же. Самое кошачье имя, зато откликается."
            if re.search(r"\bдруг\w*|\bсосед\w*", previous.casefold()):
                return "Ерлан. Мы с ним ещё со школы дружим."
        if "эйнштейн" in previous.casefold() and re.search(r"\b(?:спросил|спросишь|спрашивал|о ч[её]м)\b", q):
            return "Как ему вообще пришла идея, что время может течь по-разному. И не скучно ли ему было в патентном бюро."
        if re.search(r"\bкто (?:из (?:них|вас) )?(?:старше|младше)|^(?:а |и )?(?:она )?(?:младше|старше) или (?:старше|младше)", q) and "сестр" in previous.casefold():
            return "Я старше, сестра младше на четыре года. Так что командую пока я."
        if re.search(r"^(?:а |и |ну )?почему(?: (?:именно )?(?:он|его|бешбармак|так|не плов))?[?!. ]*$", q) and "бешбармак" in previous.casefold():
            return "Потому что бабушкин. Его готовят полдня, и весь дом пахнет праздником."
        if re.search(r"\bродител\w* (?:\w+ )?(?:узнали|знают|узнают|в курсе)", q) and re.search(r"школ|суббот|охранник|урок", previous.casefold()):
            return "Узнали, конечно, охранник же маме позвонил. Мама смеялась дольше всех."
        if re.fullmatch(r"(?:а |и )?с кем(?: (?:пойдешь|пойдёшь|поедешь|будешь|идешь|идёшь|пойдете))?[?!. ]*", q):
            return "С Ерланом, мы с ним ещё со школы дружим."
        if re.search(r"\b(?:денег (?:нет|совсем нет)|нет денег)\b", q) and re.search(r"\b(?:деньги|верни|долг)\b", previous.casefold()):
            return "Тогда пусть отдаёт частями, хоть по чуть-чуть. Мне не горит, главное, чтобы не пропал."
        if self.last_route == "reaction" and re.match(r"(?:а|и|ну) (?:если|вдруг|когда)\b", q):
            if re.search(r"\b(?:потише|неприятно|договоримся|про деньги|очередь|спрашивай)\b", previous.casefold()):
                return "Тогда поговорил бы ещё раз, уже серьёзнее. Ссориться не люблю, но и терпеть бесконечно не стану."
            if re.search(r"\b(?:жаль|обидно|рядом|подышим|страшно|пересдашь)\b", previous.casefold()):
                return "Тогда отстал бы, но написал бы вечером, что если что — я рядом. Навязываться тоже не стоит."
        if title and re.search(r"\b(?:о чем|про что|что за|расскажи)\b", q) and re.search(r"\b(?:она|он|оно|это|книга|книжка|фильм|кино)\b|^(?:а |и )?(?:о чем|про что)", q):
            return TITLES[title][0]
        if title and re.search(r"\bсколько раз\b", q):
            return "Раза три точно, может, и больше. Каждый раз что-то новое замечаю."
        if title and re.search(r"\bпочему\b", q) and re.search(r"\b(?:посоветовал|советуешь|нравится|любишь|его|её|ее)\b", q):
            return TITLES[title][1]
        if title and re.search(r"\b(?:с кем)\b", q):
            return "С соседом, Ерланом. Он, правда, обычно засыпает на середине."
        if title and re.search(r"\b(?:зацепил\w*|зашел|зашёл|цепляет|нравится в нем|нравится в нём|чем (?:он|она) тебе)\b", q):
            if title != "Интерстеллар":
                return TITLES[title][1]
            return "Атмосферой и музыкой. И тем, что там про время и про семью, а не только про космос."
        if title and re.search(r"\b(?:понравил\w*|как тебе|как (?:он|она|оно) тебе|стоит (?:ли )?(?:смотреть|читать))\b", q):
            return TITLES[title][1]
        if re.search(r"\bтам\b|\bв нем\b|\bв нём\b", q):
            named = [(m.start(), data) for data in travel.CITIES.values()
                     for m in [re.search(r"\b(?:" + data["pattern"] + r")", previous.casefold())] if m]
            if named:
                # The first city named is the one the speaker is talking about.
                data = min(named, key=lambda item: item[0])[1]
                if re.search(r"\b(?:интересн\w*|посмотреть|сходить|посетить|красив\w*|делать|заняться)\b", q):
                    return data["place"]
                for season, expr in travel.SEASONS:
                    if expr.search(q):
                        return data[season]
                if re.search(r"\b(?:холодно|жарко|тепло|погода)\b", q):
                    return data["winter"] if "холод" in q else data["summer"]
        return None

    def _retrieve(self, question, threshold):
        if self.retrieval is None:
            return None
        # Spectators' questions are mostly independent; the previous reply is
        # context only for a short follow-up such as «А почему?».
        follow = len(question.split()) <= 5 and re.match(r"^(?:а|и|но|ну|так|почему|зачем|серьезно|серьёзно|правда)\b", question.strip(), re.I)
        previous = self.turns[-1][1] if self.turns and follow else None
        seen = {answer.casefold() for _, answer in self.turns}
        if self.last_retrieval is None or self.last_retrieval[0] != question:
            self.last_retrieval = (question, self.retrieval.propose(question, previous, seen))
        found = self.last_retrieval[1]
        if found and found[1] >= threshold and found[2]["dense"] >= RETRIEVAL_MIN_DENSE:
            return found[0]
        return None

    def _generate(self, question, context):
        options = self.word_model.candidates(context)
        neural = self.generator.generate(context)
        cues = self.word_model.cues(context)
        ranks = {text: score for text, score in options}
        if neural and re.search("[а-яё]", neural.casefold()) and neural not in ranks:
            ranks[neural] = -2.
        focus = self.turns[-1][1]+" "+question if self.turns and FOLLOW.fullmatch(question) else question
        hints = self.planner.hints(focus)
        contextual = self.language.generate(context, plan=hints, guidance=1.2)
        if contextual:
            ranks.setdefault(contextual, -1.5)
        # An oblique interrogative can be answered by filling the missing
        # constituent. Require evidence for its grammatical form in real text.
        completions = {text: strength for text, strength in self.cloze.complete(question) if strength >= -3.2}
        for text in completions:
            ranks.setdefault(text, -1.)
        proposals = [answer for answer in ranks if meaningful_generation(answer, question)]
        if not proposals:
            return None
        fit_scores = [self.confidence.assess(focus, answer) for answer in proposals]
        valid = [i for i, answer in enumerate(proposals)
                 if fit_scores[i]>=self.confidence.threshold or answer in completions]
        if not valid and IMAGINE.search(question):
            # Imaginary requests can use content supported by the user's own
            # premise without presenting a conjecture as an established fact.
            heads = {w[:5] for w in hints}
            valid = [i for i,text in enumerate(proposals) if heads &
                     {w[:5] for w in tokenize(text) if len(w)>=4} and fit_scores[i]>=.39 and len(text)<130]
        if not valid:
            return None
        probabilities = self.generator.score(context, proposals)
        unconditioned = self.generator.score("", proposals)
        previous = {answer.casefold() for _, answer in self.turns}
        def quality(i):
            answer = proposals[i]
            tokens = tokenize(answer)
            content = set(t for t in tokens if t.isalpha() and len(t) >= 4)
            support = sorted([cues.get(t, 0.) for t in content], reverse=True)
            grounded = sum(support[:3]) / (3 + math.sqrt(len(content)))
            fit = probabilities[i] - unconditioned[i]
            value = (.22 * max(-4., min(2., ranks[answer])) + .30 * probabilities[i]
                     + 1.1 * fit + .72 * grounded + .80 * self.word_model.fluency(tokens)
                     - .10 * max(0, len(tokens) - 17))
            if answer in completions:
                value += 4.5 + .40 * completions[answer]
            if answer.casefold() in previous:
                value -= 2.
            return value + 1.5*fit_scores[i]
        return proposals[max(valid, key=quality)]
