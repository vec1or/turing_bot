"""Corpus-trained pair discriminator and a conservative clarification transducer."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import re
import unicodedata
import numpy as np

from .neural import Generator, tokenize


WORD = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*", re.I)
ENDING = re.compile(r"(?:иями|ами|ями|ыми|ими|ого|его|ому|ему|ую|юю|ая|яя|ое|ее|ые|ие|ый|ий|ой|ых|их|ам|ям|ах|ях|ом|ов|ев|ей|а|я|у|ю|е|и|ы|о|ь)$")
STOP = set("""а без был была были было быть бы в вам вами вас весь во вот все всего всею вы где да даже для до
его ее ей если есть еще же за здесь зачем и из или им именно их к как какая какие какой какого каком
когда которую который кто куда ли либо лишь мне меня может можно мы на над надо нам нас наш не него нее
нет ни них но ну нужно о об он она они от по под пока после потому поэтому почему при про просто раз с
сам саму свое свою себе себя со та так такой там твой тебе тебя тем то того тоже том тот ту тут ты
у уж уже чем чему что чтобы это этой этом этот я мой моя мою моей мои мое моих твоего вашего ваш
скажите скажи пожалуйста расскажи лучше подробнее обоих тогда""".split())
INTERROGATIVE = re.compile(r"^(?:а |ну )?(?:что|кто|почему|зачем|где|куда|когда|откуда|как\w*|чем|чей|сколько|знаешь ли|правда ли)\b")
FACT_PERSON = re.compile(r"\b(?:я|меня|мне|мной|мо[яйеюи]|моего|моей|моих|моем|у нас|моя семья)\b")
CURRENT = re.compile(r"\b(?:сейчас|ныне|сегодня|сегодняшн\w*|вчера|завтра|нынешн\w*|текущ\w*|последни[йехм]\w*|актуальн\w*|прямо сейчас)\b")
PRESUPPOSE = re.compile(r"\b(?:перестал\w*|снова|опять|скрыва\w*|притворя\w*)\b")
WHY = re.compile(r"^(?:почему|из-за чего|по какой причине|отчего|как так вышло,? что)\s+", re.I)
ADVICE = re.compile(r"\b(?:посовет\w*|совет\w*|стоит ли|как лучше|выбрать|что делать|что взять|как быть|помог\w*|помоч\w*|помочь)\b")
PROCEDURE = re.compile(r"^(?:как|чем) (?:мне |нам |лучше |можно |самостоятельно |проще )*([а-яё-]+(?:ть|ти|чь)(?:ся|сь)?(?:\s.+)?)$", re.I)
UNKNOWN_TERM = re.compile(r"\b(?:что (?:такое|значит|означает)|как (?:понимать|истолковать))\s+(.+)", re.I)
DIRECTIVE = re.compile(r"^(?:а )?(?:скажи(?:те)?|подскажи(?:те)?|уточни(?:те)?|напомни(?:те)?|расскажи(?:те)?)(?: мне)?\s*[,—: -]+", re.I)
PERSON = {
    "я":"ты", "меня":"тебя", "мне":"тебе", "мной":"тобой", "мой":"твой", "моя":"твоя",
    "моё":"твоё", "мое":"твоё", "мою":"твою", "мои":"твои", "моих":"твоих", "моего":"твоего",
    "моей":"твоей", "моем":"твоём", "моём":"твоём", "моему":"твоему", "моим":"твоим",
}


def normal(text):
    return unicodedata.normalize("NFC", text).casefold().replace("ё", "е")


def root(word):
    short = ENDING.sub("", word)
    return short if len(short) >= 3 else word


@dataclass(frozen=True)
class Tokens:
    words: tuple
    content: frozenset
    plain: str


@lru_cache(maxsize=1024)
def encode(text):
    string = normal(text[-500:].split("\n")[-1])
    ws = tuple(WORD.findall(string)[:36])
    return Tokens(ws, frozenset(root(w) for w in ws if len(w) > 2 and w not in STOP), " ".join(ws))


def features(q, a, lexicon, idf):
    overlap = q.content & a.content
    qa = [lexicon.get(w, {}) for w in q.content]
    strengths = sorted((max((row.get(w, 0.) for row in qa), default=0.) for w in a.content), reverse=True)
    shared_idf = sum(idf.get(w, 5.) for w in overlap)
    cover = len(overlap) / max(1, len(q.content))
    precision = len(overlap) / max(1, len(a.content))
    first = a.words[:1]
    return (
        min(20., shared_idf), cover, precision, cover*precision,
        max(strengths, default=0.), sum(strengths[:3])/3, sum(v > .9 for v in strengths)/max(1,len(strengths)),
        float("сколько" in q.words and bool(re.search(r"\d|\b(?:один|два|три|четыре|пять)\b", a.plain))),
        float(bool(WHY.match(q.plain+" ")) and ("потому" in a.words or "поэтому" in a.words or "из-за" in a.words)),
        float("ли" in q.words and first in {("да",),("нет",)}),
        float(bool({"я","ты","меня","тебя"} & set(q.words)) and bool({"я","ты","меня","тебя"} & set(a.words))),
        float(q.words[:1] == a.words[:1] and bool(q.words)), math.log1p(len(a.words)), math.log1p(len(q.words)),
    )


class Semantic:
    """Cosine from the GRU's in-batch contrastive objective; share its weights."""
    def __init__(self, gen):
        self.gen = gen

    @lru_cache(maxsize=1024)
    def query(self, text):
        v = self.gen.encode(text)[0]
        return v/(np.linalg.norm(v) or 1.)

    @lru_cache(maxsize=4096)
    def answer(self, text):
        ids = [self.gen.ids[t] for t in tokenize(text) if t in self.gen.ids]
        if not ids:
            return np.zeros(self.gen.config["hidden"], np.float32)
        v = self.gen.weights["topic.weight"][ids].sum(0)
        return v/(np.linalg.norm(v) or 1.)

    def fit(self, q, a):
        return float(self.query(q) @ self.answer(a))


def _clause(text):
    parts = [s.strip() for s in re.split(r"[.!?\n]+", text.strip()) if s.strip()]
    s = parts[-1] if parts else text.strip()
    return DIRECTIVE.sub("", re.sub(r"^(?:а|ну)\s+", "", s, flags=re.I)).strip(" \"«»")


def _fragment(s, idf):
    # Select a short contiguous source fragment around a specific noun/verb.
    matches = list(WORD.finditer(s))
    choices = []
    for i, m in enumerate(matches):
        w = normal(m[0])
        if w in STOP or len(w) < 3:
            continue
        score = idf.get(root(w), 5.) + .18*i
        if re.search(r"(?:ть|ти|чь)(?:ся|сь)?$", w):
            score -= 1.5
        if i and re.search(r"(?:ый|ий|ая|ое|ые|их|ую|ой|ого|ых|им|ыми)$", normal(matches[i-1][0])):
            score += 2.
        if m[0][0].isupper() and i:
            score += .5
        choices.append((score, i))
    if not choices:
        return ""
    i = max(choices)[1]
    left = i
    right = i
    for j in (i-1, i+1):
        if 0 <= j < len(matches) and j != i and normal(matches[j][0]) not in STOP:
            if s[matches[min(i,j)].end():matches[max(i,j)].start()].isspace():
                left, right = min(left,j), max(right,j)
    return s[matches[left].start():matches[right].end()][:62]


def _perspective(s):
    return WORD.sub(lambda m: PERSON.get(m[0].casefold(), m[0]), s)


def _lower_first(s):
    return s[:1].lower() + s[1:]


class Confidence:
    def __init__(self, asset, generator):
        if asset.get("format") != "arena-pair-contrast-1":
            raise ValueError("Неизвестная версия модели связи реплик.")
        self.lexicon, self.idf = asset["association"], asset["idf"]
        self.weights, self.bias = asset["coefficients"], asset["intercept"]
        self.threshold, self.parts = asset["threshold"], asset["discourse"]
        self.semantic = Semantic(generator)

    @classmethod
    def load(cls, directory=None, generator=None):
        path = (Path(directory) if directory else Path(__file__).with_name("assets")) / "confidence.json.gz"
        with gzip.open(path, "rt", encoding="utf8") as f:
            return cls(json.load(f), generator or Generator.load(directory))

    def assess(self, question, candidate):
        if not question.strip() or not candidate.strip():
            return 0.
        vals = (*features(encode(question), encode(candidate), self.lexicon, self.idf), self.semantic.fit(question, candidate))
        logit = self.bias + sum(w*x for w, x in zip(self.weights, vals))
        return 1. / (1. + math.exp(-max(-35., min(35., logit))))

    @staticmethod
    def needs_evidence(question):
        s = normal(_clause(question))
        guessing = re.search(r"\b(?:угада\w*|предска\w*|догада\w*)\b", s)
        return bool(guessing or (INTERROGATIVE.search(s) or "?" in question) and (
            CURRENT.search(s) or PRESUPPOSE.search(s) or (FACT_PERSON.search(s) and not ADVICE.search(s))
        ))

    def clarify(self, question):
        s = _clause(question)[:210].rstrip(" ,;:")
        plain = normal(s)
        p = self.parts
        term = UNKNOWN_TERM.search(s)
        if term:
            focus = term[1].strip(" «»\"")[:56].rstrip()
            return f"{p['request'].capitalize()} {p['example']} со словами «{focus}»?"
        method = PROCEDURE.match(s)
        if method:
            complement = _perspective(method[1])
            return f"{p['what_have'].capitalize()} уже есть, чтобы {complement}?"
        if not encode(plain).content:
            if re.search(r"\b(?:кого|них|двух|обоих|котор\w*)\b", plain):
                return f"Какие варианты {p['meaning'].removeprefix('что ')}?"
            return f"{p['more'].capitalize()} об этом подробнее?"
        why = WHY.match(s)
        if why:
            claim = _lower_first(_perspective(s[why.end():]).strip())
            if claim and not re.search(r"\b(?:ты|вы|я|мы)\b", normal(s[why.end():])):
                return f"{p['verify'].capitalize()}, что {claim}? {p['origin'].capitalize()} это {p['known']}?"
        if INTERROGATIVE.match(plain) and not ADVICE.search(plain):
            # Verbs marked for the first/second person cannot be embedded
            # after a pronoun swap without conjugating them. Quote instead.
            finite_person = re.search(r"\b(?:ты|вы|мы)\b|\b[а-я]+(?:ешь|ишь|ете|ите|емся|юсь)\b", plain)
            if not finite_person and (not re.search(r"\bя\b", plain) or
                                      re.search(r"\b[а-я]+(?:ла|ли|ло|ал|ил|ел)\b", plain)):
                embedded = _lower_first(_perspective(s))
                if CURRENT.search(plain):
                    return f"{p['unknown'].capitalize()}, {embedded}. Есть ли {p['information']} на этот момент?"
                if FACT_PERSON.search(plain):
                    return f"{p['unknown'].capitalize()}, {embedded}. {p['what'].capitalize()} тебе об этом {p['known']}?"
                return f"{p['unknown'].capitalize()}, {embedded}. {p['more'].capitalize()} об этом подробнее?"
        focus = _fragment(s, self.idf)
        if ADVICE.search(plain) and focus:
            return f"Если речь о словах «{focus}», {p['goal']}?"
        if focus:
            return f"{p['meaning'].capitalize()} под словами «{focus}»?"
        return f"{p['more'].capitalize()} об этом подробнее?"
