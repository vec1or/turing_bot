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
from .profile import Profile
from .retrieval import Retrieval
from . import goal_plan, reactions, tips, travel
from .rounds import Abstentions, clean, focal_question, requested_format, short_operation, standalone

MAX_CHARS = 300
MAX_INPUT = 4000
RETRIEVAL_THRESHOLD = .72
RETRIEVAL_STRONG = .78
RETRIEVAL_MIN_DENSE = .5  # tuned on validation questions, see evaluation/RESULTS-3.0.md
RECALL_SELF = re.compile(r"^(?:а |напомни[, ]+)?(?:что|какой вариант|какое слово) (?:же )?ты (?:только что |тогда |в итоге )?(?:выбрал|назвал|ответил|предпоч[её]л)а?[?!. ]*$",re.I)
REPEAT = re.compile(r"(?:предыдущ|прошл|последн)\w* (?:вопрос|ответ)\w*.*(?:ответил|сказал|повтори)|(?:ответил|сказал) на (?:предыдущ|прошл)\w* вопрос|повтори,? (?:что|свой|свои|последн)\w*|что ты (?:только что|сейчас) (?:сказал|ответил)",re.I)
TITLES = {
    "Мастер": ("Про то, как дьявол со свитой приезжает в Москву тридцатых, а параллельно про Мастера, который пишет роман о Понтии Пилате. Ну и про любовь, конечно.",
               "Очень. Перечитываю, и каждый раз как в первый."),
    "Интерстеллар": ("Земля умирает, и команда летит через червоточину искать новую планету. А по сути это про отца и дочь и про время.",
                     "Да, очень. Пересматривал раза три, и каждый раз в конце сижу молча."),
}
EXPLICIT = re.compile(r"\b(?:объясни\w*|расскажи (?:про|о|об)|что такое|что значит|кто так(?:ой|ая)|как работает|как устроен\w*)\b", re.I)
WHY = re.compile(r"(?:а |но |и |ну )?(?:почему|зачем|с чего бы|отчего|как так)(?: именно)?(?: так| это| же| ты так думаешь| ты так считаешь)?[?!. ]*",re.I)
WHY_REPLIES = ("Ну, просто по опыту так.", "Сложно объяснить, просто так чувствую.", "Не знаю, мне так кажется.",
               "Просто так сложилось.", "Ну а как иначе, по-моему, очевидно.")
FOLLOW = re.compile(r"^(?:а |но )?(?:почему|зачем|как так|дальше|что потом|что дальше)(?: именно)?(?: так| туда| это| там)?[?!. ]*$",re.I)


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
        self.turns = deque(maxlen=3)

    def reset(self):
        self.turns.clear()
        self.last_source = None
        self.last_route = None
        self.last_retrieval = None
        self.persona.reset()
        self.abstentions.seen.clear()

    def answer(self, question):
        raw = question[:MAX_INPUT]
        if not raw.strip():
            raise ValueError("Введите вопрос.")
        question = clean(raw) or raw.strip()
        self.last_source = None
        for route, candidate, source in self._candidates(question):
            # The final fallback is already a complete, language-checked utterance.
            # Always keep an answer available even for impossible format demands.
            if route!="fallback":
                candidate=requested_format(candidate,question)
            if standalone(candidate):
                reply = display_text(candidate)
                if not standalone(reply):
                    continue
                self.last_route = route
                self.last_source = source
                if route != "persona":
                    self.persona.reset()
                self.turns.append((question, reply))
                return reply

    def _candidates(self, question):
        if self.turns and self.last_route != "fallback" and RECALL_SELF.fullmatch(question):
            previous = self.turns[-1][1]
            yield "recall", previous, None
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
        yield "reaction", reactions.answer(question), None
        yield "reaction", reactions.speech(question), None
        yield "travel", travel.answer(question), None
        yield "profile", self.profile.answer(question), None
        yield "goal-plan", goal_plan.answer(question), None
        yield "explanation", explain(question), None
        # Resolve a request for direct speech before planning a practical action.
        # A problem in the premise (e.g. lateness) need not be the requested task.
        social = self.social.propose(question)
        if social and re.search(r"\b(?:отве(?:ти|ча)|сказа|напиш)\w*",question,re.I):
            yield "social", social[0], None
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
        yield "tips", tips.answer(question), None
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
                if article:
                    yield "article", article[0], article[1]
                passage = self.knowledge.propose(focus)
                if passage:
                    yield "reference", passage[0], "https://ru.wikipedia.org/?curid=" + str(passage[1])
        # The free generator, topic associations and word completion were
        # removed from the chain in 3.0: blind checks found them mostly
        # meaningless. Their code and weights remain for comparison.
        yield "retrieval", self._retrieve(question, RETRIEVAL_THRESHOLD), None
        # Authored situation plans match loosely; they are used only when no
        # real reply is close enough.
        yield "situation", self.situations.answer(question), None
        yield "fallback", self.abstentions.answer(question), None

    def _follow_up(self, question):
        """Continue about a title or a city the student has just named."""
        previous = self.turns[-1][1]
        q = question.casefold().replace("ё", "е").strip()
        if len(q) > 120:
            return None
        title = next((t for t in TITLES if t in previous), None)
        if re.fullmatch(r"(?:а |и )?с кем[?!. ]*", q):
            return "С соседом по комнате, мы с ним ещё со школы дружим."
        if self.last_route == "reaction" and re.match(r"(?:а|и|ну) (?:если|вдруг|когда)\b", q):
            return "Тогда поговорил бы ещё раз, уже серьёзнее. Ссориться не люблю, но и терпеть бесконечно не стану."
        if title and re.search(r"\b(?:о чем|про что|что за|расскажи)\b", q) and re.search(r"\b(?:она|он|оно|это|книга|книжка|фильм|кино)\b|^(?:а |и )?(?:о чем|про что)", q):
            return TITLES[title][0]
        if title and re.search(r"\b(?:понравил\w*|как тебе|как (?:он|она|оно) тебе|стоит (?:ли )?(?:смотреть|читать))\b", q):
            return TITLES[title][1]
        if re.search(r"\bтам\b|\bв нем\b|\bв нём\b", q):
            named = [(m.start(), data) for data in travel.CITIES.values()
                     for m in [re.search(r"\b(?:" + data["pattern"] + r")", previous.casefold())] if m]
            if named:
                # The first city named is the one the speaker is talking about.
                data = min(named, key=lambda item: item[0])[1]
                if re.search(r"\b(?:интересн\w*|посмотреть|сходить|посетить|красив\w*|делать)\b", q):
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
