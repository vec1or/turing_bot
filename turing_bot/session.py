from __future__ import annotations

from collections import deque
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
from . import goal_plan, reactions, travel
from .rounds import Abstentions, clean, focal_question, requested_format, short_operation, standalone

MAX_CHARS = 300
MAX_INPUT = 4000
RECALL_SELF = re.compile(r"^(?:а |напомни[, ]+)?(?:что|какой вариант|какое слово) (?:же )?ты (?:только что |тогда |в итоге )?(?:выбрал|назвал|ответил|предпоч[её]л)а?[?!. ]*$",re.I)
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
        self.abstentions = Abstentions()
        self.last_source = None
        self.last_route = None
        self.turns = deque(maxlen=3)

    def reset(self):
        self.turns.clear()
        self.last_source = None
        self.last_route = None
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
        yield "travel", travel.answer(question), None
        yield "profile", self.profile.answer(question), None
        yield "goal-plan", goal_plan.answer(question), None
        yield "writing", self.writing.answer(question, self.imagination), None
        yield "creative", self.imagination.answer(question), None
        yield "explanation", explain(question), None
        yield "practical", self.practical.answer(question), None
        # Resolve a request for direct speech before planning a practical action.
        # A problem in the premise (e.g. lateness) need not be the requested task.
        social = self.social.propose(question)
        if social and re.search(r"\b(?:отве(?:ти|ча)|сказа|напиш)\w*",question,re.I):
            yield "social", social[0], None
        yield "situation", self.situations.answer(question), None
        yield "persona", self.persona.answer(question), None
        if social:
            yield "social", social[0], None
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
            article = self.articles.propose(focus)
            if article:
                yield "article", article[0], article[1]
            passage = self.knowledge.propose(focus)
            if passage:
                yield "reference", passage[0], "https://ru.wikipedia.org/?curid=" + str(passage[1])
            if re.search(r"^(?:что (?:такое|значит)|кто так(?:ой|ая)|кто (?:изобр[её]л|написал)|где находится|когда (?:родил|основ))\b",focus.casefold()):
                yield "fallback", self.abstentions.answer(question), None
            phrase = self.planner.adjective(question)
            if phrase:
                yield "word-completion", phrase, None
            yield "corpus-topic", self.associations.answer(question), None
            context = question
            if self.turns and FOLLOW.fullmatch(question):
                context = "\n".join((*self.turns[-1], question))
            yield "generator", self._generate(question, context), None
        yield "fallback", self.abstentions.answer(question), None

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
