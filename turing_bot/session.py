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
                    if a and b and len(a & b) / min(len(a), len(b)) < 0.5:
                        continue
                if reply in said and route != "recall":
                    # A person does not repeat a whole sentence word for word.
                    fresh = [d for d in DENIALS if d not in said]
                    if reply in DENIALS and fresh:
                        reply = fresh[0]
                    elif route == "explanation":
                        reply = display_text("Да по той же причине. " + re.split(r"(?<=[.!])\s", reply, maxsplit=1)[0])
                    elif route in ("profile", "persona", "travel", "tips", "reaction", "follow-up"):
                        reply = display_text("Я же уже говорил: " + reply[:1].lower() + reply[1:])
                    else:
                        continue
                    if reply in said:
                        continue
                self.last_route = route
                self.last_source = source
                if route != "persona":
                    self.persona.reset()
                self.turns.append((question, reply))
                self.history.append((question, reply))
                if self.first_question is None:
                    self.first_question = question
                    self.first_answer = reply
                return reply

    def _candidates(self, question):
        asks_name = re.search(r"как меня зовут|мо[её] имя|помнишь,? как меня", question, re.I)
        intro = None if asks_name else (re.search(r"\b(?:меня зовут|мо[её] имя|зови меня)\s+([А-ЯЁA-Zа-яё][а-яёa-z]{2,})", question, re.I)
                                       or re.search(r"\bменя\s+(?!все|так|тоже|никак|вообще)([а-яё]{3,})\s+зовут", question, re.I))
        if intro:
            self.user_name = intro[1][:1].upper() + intro[1][1:].lower()
            yield "persona", f"Запомнил, {self.user_name}. Приятно познакомиться, я Тимур.", None
        if self.turns and re.search(r"около \d+ тысяч", self.turns[-1][1]) and re.search(r"долго|не считают|так не", question, re.I):
            yield "follow-up", "Так я и не считал точно, прикинул на глаз. Если надо до единицы — дай минуту и листочек.", None
        if self.first_question and re.search(r"перв\w* (?:что|сообщени|вопрос|фраз)\w*.{0,25}(?:написал|спросил|писал|сказал)|(?:написал|спросил) (?:тебе )?(?:самым )?первым", question, re.I):
            quoted = re.sub(r"\?\s*", ". ", self.first_question).strip(" .!")
            yield "recall", f"Ты написал: «{quoted}». Помню, помню."[:300], None
        if asks_name:
            if self.user_name:
                yield "recall", f"{self.user_name}, конечно. Ты же говорил, я запомнил.", None
            else:
                yield "recall", "Ты же не говорил, как тебя зовут. Скажи — запомню.", None
        if self.turns and self.last_route != "fallback" and RECALL_SELF.fullmatch(question):
            previous = self.turns[-1][1]
            yield "recall", previous, None
        if self.turns and self.first_answer and re.search(r"\bперв\w* (?:вопрос|раз)\w*|в (?:самом )?начале", question, re.I) and re.search(r"\b(?:ты )?(?:ответил|сказал|отвечал)\w*", question, re.I):
            yield "recall", f"Сказал: «{self.first_answer.rstrip('.')}». Помню, помню."[:300], None
        about = re.search(r"\bчто (?:ты )?(?:отвечал|ответил|сказал|говорил|писал)\w* (?:про|о|об|насчет|насчёт) ([а-яё]{3,})", question.casefold())
        if about and self.history:
            stem = about[1][:5]
            for earlier_q, earlier_a in reversed(self.history):
                if stem in earlier_q.casefold() or stem in earlier_a.casefold():
                    yield "recall", f"Я сказал: «{earlier_a.rstrip('.')}».", None
                    break
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
        yield "operation", short_operation(question), None
        yield "reasoning", self.reasoning.answer(question), None
        # A previous *question* from a different spectator is not testimony.
        history = [q for q, _ in self.turns if "?" not in q]
        extracted = self.reader.propose(question, history)
        if extracted:
            yield "reader", extracted[0], None
        caveat = limitation(question)
        if caveat:
            is_reflection = caveat.startswith(("Первое,", "Я бы проверил"))
            yield "reflection" if is_reflection else "abstention", caveat, None
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
        answers = [self._precise(p) for p in parts]
        if not all(answers):
            return None
        said = {a for _, a in self.history}
        if any(a in said for a in answers):
            return None
        joined = answers[0] if answers[0] == answers[1] else answers[0].rstrip() + " " + answers[1]
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
        if re.search(r"яичниц", previous) and re.search(r"что (?:именно|конкретно)|что (?:ел|ты ел)", q):
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
        if re.search(r"списал", earlier) and re.search(r"не поверил|не верит|не поверит", q):
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
        if "Скриптонит" in previous and re.search(r"назови|конкретн|кого (?:ещ[её]|именно)|кто ещ[её]", q):
            return "Скриптонит в первую очередь. Ещё Джа Халиб и Ирина Кайратовна, когда настроение получше."
        if "Тысяч сорок" in previous and re.search(r"много|мало|нормально", q):
            return "Для студента нормально. С доставкой улетало бы в два раза больше, проверено."
        if re.search(r"подар", earlier) and re.search(r"безделуш|не любит (?:всякие|вещи)|ничего не надо", q):
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
        if re.search(r"^(?:а |и )?(?:если|вдруг) (?:он|она) .{0,12}(?:обидится|обиделся|обиделась|разозлится)", q):
            return "Ну и пусть. Если дружба держится только на этом, то это так себе дружба."
        if re.search(r"без телефона|без интернета", earlier) and re.search(r"вместо|что бы (?:ты )?делал|чем бы (?:ты )?занимал", q):
            return "Читал бы, гулял, может, наконец доделал бы курсач. Ну и спал бы больше, это точно."
        if re.search(r"помирит|накосяч", earlier) and re.search(r"не отвечает|игнор|молчит", q):
            return "Тогда дать ещё пару дней и прийти лично. Если человек дорог, гордость тут лишняя."
        if re.search(r"опозд|опаздыва", earlier) and re.search(r"(?:третий|второй|пятый|который|десятый) раз|опять|снова", q):
            return "Тогда уже не оправдываться, а честно подойти после пары: «Понимаю, что не первый раз, исправлюсь». И правда начать выходить раньше."
        if re.search(r"энергетик|кофе|чай", earlier + " " + previous) and re.search(r"сколько (?:\w+ )?(?:в день|штук|раз)", q):
            return "Чая чашек пять, кофе одну, и то в сессию. Энергетики — почти никогда."
        if "деда" in previous and re.search(r"\b(?:почему|зачем)\b", q):
            return "Это память. Он мне его отдал, когда я в первый класс пошёл, чтобы маме звонил."
        if "курьером" in previous and re.search(r"\b(?:сколько|скок|скока)\b|плати|зарплат", q) and not re.search(r"во сколько|лег|ложи|сова|встал", q):
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
        if re.search(r"анекдот|шутк|пошути", earlier) and re.search(r"\bсмешн\w*|\bсмеш\w*|\bне смешно", q):
            return "Мне смешно, потому что это правда про меня. Ну а анекдоты я рассказывать не умею, порчу концовку."
        if re.search(r"\bпарень|\bпарня", earlier) and re.search(r"^(?:а |и )?(?:если|вдруг) (?:она|он) (?:сама |сам )?(?:первая |первый )?(?:напишет|пишет|пишут|позвонит|звонит|позовет|позовёт|зовет|зовёт)", q):
            return "Тогда, может, у них там не всё так серьёзно. Но я бы всё равно не торопился, пусть она сама сначала разберётся со своим парнем."
        singer = re.search(r"\bкто (?:(?:ее|её|его|их) )?(?:поет|поёт|исполняет|написал|снял|играет)", q)
        if singer:
            name = next((n for n in ARTISTS if n.casefold() in previous.casefold()), None)
            if name:
                return f"Да {name} же, я говорил."
        if re.search(r"\b(?:докажи|уверен|правда|серьезно|серьёзно)\b", q) and re.search(r"\b(?:человек|живой|студент|не нейросеть)\b", previous.casefold()):
            return "Ну как доказать. Могу сказать, что у меня сейчас затекла нога и очень хочется чаю. Бот бы до такого не додумался."
        if re.search(r"\b(?:объясни|скажи|расскажи) (?:\w+ )?(?:попроще|проще)|\bне понял\w*|\bнепонятно\b", q) and len(q) < 60:
            first = re.split(r"(?<=[.!])\s", previous, maxsplit=1)[0].rstrip(".")
            if len(first) > 15:
                return "Если совсем коротко: " + first[:1].lower() + first[1:] + "."
        if self.last_route == "explanation" and re.search(r"\bкак (?:это |его |её |ее )?(?:\w+ )?(?:убрать|исправить|избежать|помочь|справиться|остановить|вылечить|ускорить|замедлить)", q):
            parts = re.split(r"(?<=[.!])\s", previous, maxsplit=1)
            if len(parts) == 2:
                return "Да проще всего так: " + parts[1][:1].lower() + parts[1][1:]
        if self.last_route == "explanation" and re.match(r"(?:а|и|ну) ", q):
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
            return "Да не, препод нормальный. Просто восемь утра — не время для дат и войн."
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
        if re.search(r"\bкто (?:из (?:них|вас) )?(?:старше|младше)", q) and "сестр" in previous.casefold():
            return "Я старше, сестра младше на четыре года. Так что командую пока я."
        if re.search(r"^(?:а |и |ну )?почему(?: (?:именно )?(?:он|его|бешбармак|так|не плов))?[?!. ]*$", q) and "бешбармак" in previous.casefold():
            return "Потому что бабушкин. Его готовят полдня, и весь дом пахнет праздником."
        if re.search(r"\bродител\w* (?:\w+ )?(?:узнали|знают|узнают|в курсе)", q) and re.search(r"школ|суббот|охранник|урок", previous.casefold()):
            return "Узнали, конечно, охранник же маме позвонил. Мама смеялась дольше всех."
        if re.fullmatch(r"(?:а |и )?с кем(?: (?:пойдешь|пойдёшь|поедешь|будешь|идешь|идёшь|пойдете))?[?!. ]*", q):
            return "С соседом по комнате, мы с ним ещё со школы дружим."
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
