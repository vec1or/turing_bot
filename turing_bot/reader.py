"""Read an explicit attribute from a viewer's own short statement.

The training resource contains aggregate document frequencies, never sentences.
All extractable values come from the current conversation.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import re
import unicodedata
import zlib


WORD = re.compile(r"[\wЁё]+(?:-[\wЁё]+)*", re.UNICODE)
ADJ_END = re.compile(r"(?:ыми|ими|ому|ему|ого|его|ую|юю|ая|яя|ое|ее|ые|ие|ый|ий|ой|ом|ем|ым|им|ых|их|ей)$")
NOUN_END = re.compile(r"(?:иями|ями|ами|иям|ием|ьев|ьям|ах|ях|ом|ем|ам|ям|ов|ев|ей|ью|а|я|у|ю|е|и|ы|о|ь)$")
REL_VERB = re.compile(r"(?:ался|ился|лась|лись|лось|ила|или|ило|ает|яет|ала|али|ало|ешь|ишь|ать|ять|ить|ыл|ыла|ыли|ыло|ил|ал|ит|ены|ена)$")
# General Russian colour vocabulary. Stems serve as type checks, never as
# substitute answers. An explicitly named «цвета ...» can also be read.
COLOURS = set("бел черн красн син зелен желт голуб оранжев фиолетов розов сер коричнев бежев бирюзов лилов бордов ал малинов пурпурн сиренев серебрист золотист изумрудн лазурн янтарн салатов".split())
NUMBERS = set("ноль один одна одно два две три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split())
FRAME = set("""а в на с со по из под над к у от за о об около между для перед я ты он она оно мы вы они
мне меня мной моего моей моих мою мои мой нашу наша наши его ее их наш нас мне тебя тебе твой твоя
твой свою свой той это эта этот те том тут там же и или да как какого какой какая какое какие каким
какими какого какой чей кто что сколько количества где куда цвет цвета оттенка имя имени назван названа
зовут звали назывался называется было был была были будет есть стал стала лежит лежал лежала лежали
стоял стоит стояла положил положила поставил поставила подарил подарила мне подарили принес
принесла принесло зовется пожалуйста напомни скажи можешь теперь еще точно именно раньше назови
случае сообщение сообщении сказал сказала меня собственно делом сейчас который которую кличка оказалась оказался
было которую какая какие были указывал указывала написала написал писала писал""".split())
PREP = {"в", "во", "на", "под", "над", "около", "перед", "за", "у"}
BOUNDARY = {"и", "или", "потом", "тогда", "пока", "сразу", "где", "который", "которую", "вчера", "сегодня", "завтра", "сейчас", "снова", "теперь"}
QUESTION_START = re.compile(r"^(?:а )?(?:где|куда|кто|сколько|како\w*|как|что|чей|разве|почему|зачем|когда|представь\w*|пусть|допустим|найди|скажи|напомни)\b")
STATE_CHANGE = re.compile(r"\b(?:перекрас|переимен|перелож|перестав|перемест|передум|забра|убра|увез|забы)[а-я]*\b")
PLACING = re.compile(r"^(?:леж|лежа|стоя|стои|висе|висит|хран|наход|полож|кладу|клад|постав|спрят|убра|остав|помест|нашел|нашла|постав)")
COPULA = {"был", "была", "было", "были", "есть", "оказался", "оказалась", "оказалось"}
TIME = re.compile(r"^(?:понедельник|вторник|сред[уе]|четверг|пятниц|суббот|воскресень|январ|феврал|март|апрел|мае|мая|июн|июл|август|сентябр|октябр|ноябр|декабр|минут|недел|прошл|будущ|следующ|сутк|час)")
UNCERTAIN = re.compile(r"\b(?:не|ни|если|либо|возможно|наверно|наверное|кажется|может быть|вроде|неизвестно|предположим|вместо)\b")
IRREALIS = re.compile(r"\b(?:хочу|хотел[аи]?|мечта\w*|вдруг|будет|будут|буду|надеюсь|может|могла|могли|планир\w*|допустим|представ\w*)\b")


def normal(s):
    return unicodedata.normalize("NFC", s).casefold().replace("ё", "е")


def stem(s):
    word = normal(s)
    if len(word) <= 3:
        return word
    cut = NOUN_END.sub("", word)
    return cut if len(cut) >= 3 else word


def bucket(key, size):
    return zlib.crc32(key.encode("utf-8")) % size


def words(s):
    return WORD.findall(s)


def adj(s):
    return len(s) >= 4 and ADJ_END.search(normal(s)) is not None


def colour(s):
    base = ADJ_END.sub("", normal(s).split("-")[-1])
    return base in COLOURS


def number(s):
    return normal(s) in NUMBERS or bool(re.fullmatch(r"\d{1,9}", s))


def capitalized(s):
    return s[:1].isupper() and any(c.islower() for c in s[1:]) and s.isalpha()


def question_kind(q):
    if UNCERTAIN.search(q):
        return None
    if re.search(r"\b(?:какого (?:цвета|оттенка)|как(?:ой|им) (?:был |была )?цвет)\b", q):
        return "colour"
    if re.search(r"\bсколько\b", q):
        return "count"
    if re.search(r"\b(?:как (?:я |мы |его |ее |они )?(?:по)?звал\w*|как (?:я |мы )?назвал\w*|как (?:его |ее )?зовут|как звали|как называется|какое имя|какая кличка)\b", q):
        return "name"
    if re.search(r"\b(?:где|куда)\b", q):
        return "place"
    if re.search(r"\bкто\b", q):
        return "person"
    if re.search(r"\bкак(?:ой|ая|им|им[и]|ой|ие|ое) (?:у меня |был[аои]? |оказал\w* )", q):
        return "attribute"
    return None


@dataclass(frozen=True)
class Span:
    text: str
    key: str


def _quote(span):
    return Span(f"В сообщении — «{span}».", normal(span))


def _simple(span):
    return Span(span[:1].upper() + span[1:] + ".", normal(span))


def spans(kind, toks, anchors, query_tokens):
    low = [normal(s) for s in toks]
    hit = {i for i,t in enumerate(toks) if stem(t) in anchors}
    if kind in {"colour", "attribute"}:
        if kind == "colour" and any(low[i] == "и" and colour(toks[i-1]) and colour(toks[i+1])
                                    for i in range(1, len(toks)-1)):
            return
        for i, token in enumerate(toks):
            if kind == "colour" and low[i] == "цвета" and i + 1 < len(toks) and i - 1 in hit:
                value = toks[i+1:]
                if (1 <= len(value) <= 2 and (len(value) == 1 or adj(value[0]))
                        and all(t.isalpha() and normal(t) not in FRAME | BOUNDARY | {"нет"} for t in value)):
                    yield _simple("цвета " + " ".join(value))
            if low[i] in FRAME or not (colour(token) if kind == "colour" else adj(token)):
                continue
            # Simple adjective–noun group, or a nominal predicate.
            other = i + 1 if i + 1 in hit else i - 1 if i - 1 in hit else None
            if other is None and i >= 2 and low[i-1] in {"был","была","было","были","оказался","оказалась"} and i-2 in hit:
                other = i-2
            if other is not None:
                lo, hi = sorted((other, i))
                yield _quote(" ".join(toks[lo:hi + 1]))
    elif kind == "count":
        i = 0
        while i < len(toks):
            if not number(toks[i]):
                i += 1
                continue
            start = i
            while i < len(toks) and number(toks[i]):
                i += 1
            end = i
            while end < min(i + 3, len(toks)) and adj(toks[end]) and end not in hit:
                end += 1
            if end in hit:
                yield _simple(" ".join(toks[start:i]))
    elif kind == "name":
        for i, token in enumerate(low[:-1]):
            if token not in {"зовут", "звали", "назвал", "назвала", "назвали", "кличка", "имя", "имени", "кличке"}:
                continue
            end = i + 1
            while end < len(toks) and end in hit:
                end += 1
            if end < len(toks) and capitalized(toks[end]) and stem(toks[end]) not in anchors:
                yield _simple(toks[end])
    elif kind == "person":
        # Only an explicit name immediately followed by the questioned action.
        verbs = {normal(t)[:5] for t in query_tokens if len(t) >= 5 and REL_VERB.search(normal(t))}
        for name, verb in zip(toks, toks[1:]):
            if capitalized(name) and normal(verb)[:5] in verbs and stem(name) not in anchors:
                yield _simple(name)
    elif kind == "place":
        query_actions = {normal(t)[:5] for t in query_tokens
                         if len(t) >= 5 and normal(t) not in COPULA and REL_VERB.search(normal(t))}
        if query_actions and not query_actions <= {t[:5] for t in low}:
            return
        # «Куб на полке» and a predicate of placement are evidence for
        # location. «Она говорила о кубе на вокзале» is not.
        if not query_actions and not any(PLACING.match(t) or t in COPULA for t in low):
            if len(toks) > 5 or any(REL_VERB.search(t) for t in low):
                return
        for i, p in enumerate(low[:-1]):
            if p not in PREP:
                continue
            end = i + 1
            while (end < len(toks) - 1 and adj(toks[end]) and end - i <= 2
                   and low[end+1] not in BOUNDARY | FRAME | PREP and not REL_VERB.search(low[end+1])):
                end += 1
            if (end in hit or not hit.isdisjoint(range(i, end + 1)) or low[end] in FRAME | BOUNDARY
                    or TIME.match(low[end]) or REL_VERB.search(low[end]) or number(toks[end])):
                continue
            yield _simple(" ".join(toks[i:end + 1]))


class Reader:
    def __init__(self, resource):
        if resource.get("format") != "arena-reader-idf-1" or not resource.get("documents"):
            raise ValueError("Несовместимая статистика модуля чтения.")
        self.docs = resource["documents"]
        self.df = resource["document_frequencies"]
        if not self.df or any(not isinstance(x, int) or not 0 <= x <= self.docs for x in self.df):
            raise ValueError("Некорректные частоты признаков.")

    @classmethod
    def load(cls, directory=None):
        base = Path(directory) if directory else Path(__file__).with_name("assets")
        with gzip.open(base / "reader.json.gz", "rt", encoding="utf-8") as stream:
            return cls(json.load(stream))

    def propose(self, question: str, user_history: list[str]) -> tuple[str, float] | None:
        if not question.strip() or len(question) > 4000:
            return None
        parts = [s.strip() for s in re.findall(r"[^.!?\n]+[.!?]*", question) if s.strip()]
        if not parts:
            return None
        q = parts[-1]
        kind = question_kind(normal(q))
        if kind is None:
            return None
        qt = words(q)
        anchors = {stem(t) for t in qt if len(t) >= 3 and normal(t) not in FRAME and not adj(t)
                   and not REL_VERB.search(normal(t)) and not number(t)}
        if not anchors:
            return None
        # A common function word alone cannot establish the requested object.
        strongest = max(math.log((1+self.docs)/(1+self.df[bucket(k, len(self.df))])) for k in anchors)
        if strongest < 2:
            return None
        # In «Сколько кубиков на подносе?» the quantity belongs to кубиков.
        head = next(stem(t) for t in qt if stem(t) in anchors)
        modifiers = {ADJ_END.sub("", normal(t)) for t in qt if adj(t) and normal(t) not in FRAME}
        sources = list(user_history[-4:]) + parts[:-1]
        found = {}
        for source in sources:
            for sentence in re.findall(r"[^.!?\n]+[.!?]*", source[-4000:]):
                if "?" in sentence or QUESTION_START.match(normal(sentence.strip())):
                    continue
                if re.search(r"\b(?:если|предположим|возможно|допустим|наверное)\b", normal(sentence)):
                    if head in {stem(t) for t in words(sentence)}:
                        return None
                    continue
                for clause in re.split(r"[,;]", sentence):
                    toks = words(clause)
                    low = normal(clause)
                    clause_stems = {stem(t) for t in toks}
                    if found and STATE_CHANGE.search(low) and (head in clause_stems or re.search(r"\b(?:его|ее|их)\b", low)):
                        found.clear()
                    if not 2 <= len(toks) <= 36 or not anchors <= clause_stems:
                        continue
                    if UNCERTAIN.search(low) or IRREALIS.search(low) or re.search(r"\bили\b", low):
                        return None
                    if kind in {"person", "place", "name"} and re.search(r"\bи\b", low):
                        return None
                    if not modifiers <= {ADJ_END.sub("", normal(t)) for t in toks if adj(t)}:
                        continue
                    scope = {head} if kind in {"colour", "count", "attribute"} else anchors
                    for span in spans(kind, toks, scope, qt):
                        found[span.key] = span.text
        if len(found) != 1:
            return None
        return next(iter(found.values())), min(.98, .88 + strongest / 100)
