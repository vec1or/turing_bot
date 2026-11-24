"""Recognize a social speech act and decode a short corpus-trained response."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import re


WORD = re.compile(r"[а-яё]+(?:-[а-яё]+)*", re.I)
TIME = re.compile(r"\b(?:утр[оа]|вечер[а]?|день|дня|ноч[ьи])\b")
GREET = re.compile(r"^(?:здравствуй(?:те)?|здрасьте|привет(?:ик)?|салют|здор[оó]во)\b")
THANK = re.compile(r"^(?:(?:большое|огромное|большущее|сердечное) )?(?:спасибо|благодарю|благодарен|благодарна)\b")
APOLOGY = re.compile(r"^(?:ой[, ]+)?(?:извини(?:те)?|извиняюсь|прости(?:те)?|прошу прощения|виноват[а]?)\b")
LEAVE = re.compile(r"^(?:до (?:свидани[яй]|встречи|скорого|завтра)|увидимся|прощай(?:те)?|счастливо|всего (?:доброго|хорошего)|пока(?:-пока)?)\b")
CONGRAT = re.compile(r"^(?:поздравля(?:ю|ем)|с праздником|с наступающим|с дн[её]м рождения|с новым годом)\b")
CALL = re.compile(r"^(?:эй|ау|алло|алл[её]|ты (?:здесь|тут|меня слышишь)|вы (?:здесь|тут|меня слышите))\b")
VOCATIVE = {"всем","все","друзья","коллеги","ребята","собеседник","друг","здравствуйте","тебе","вам","тоже",
            "взаимно","вновь","снова","большое","огромное","большущее","большейшее","очень","ещё","раз","привет"}
ASK = re.compile(r"\b(?:ответи(?:л[аи]?|те|ть|шь|м)|отвеча(?:ть|ешь)|отв[её]тишь|отреагир\w*|"
                 r"(?:что|как) (?:бы )?(?:(?:ты|вы|я|ему|ей|мне|нам|тут) )*(?:бы )?ска(?:жешь|зать|зал[аи]?|жете)|"
                 r"(?:что|как) (?:бы )?ты (?:ему |ей )?сказал[аи]?|в ответ)\b")
REPORTS = {
    "greet": r"\b(?:поздоров|поприветств)\w*",
    "thanks": r"\b(?:поблагодар|благодарил)\w*",
    "sorry": r"\b(?:извинил(?:ся|ась|ись)|извиня(?:лся|лась|лись)|попроси(?:л|ла|ли) прощения)\b",
    "bye": r"\b(?:попроща|распроща)\w*",
    "celebrate": r"\b(?:поздравил[аи]?|поздравляет)\b",
    "call": r"\b(?:окликнул[аи]?|позвал[аи]?)\b",
}
REPORTS = {key:re.compile(expr) for key,expr in REPORTS.items()}
NEGATED = re.compile(r"\b(?:не|даже не|так и не|вовсе не)\s+(?:поздоров|попривет|поблагодар|благодарил|извин|попросил[аи]?|попрощ|поздравил|окликнул|позвал)\w*")
EXCEPTION = re.compile(r"\b(?:не готов[аы]?|не (?:могу|хочу|собираюсь) (?:его |е[её] |их )?(?:прощать|простить)|угрожа\w*|опасност\w*)\b")
META = re.compile(r"\b(?:произношени\w*|перевед\w*|перевод\w*|значени\w*|пишется|писать правильно|синоним\w*|антоним\w*|ошибк\w* в слове)\b")
SPEECH = re.compile(r"\b(?:сказал[аи]?|скажет|скажут|говорит|говорят|написал[аи]?|напишет|мне сказали)\s*[:,—-]?\s+")
MULTI = re.compile(r"\b(?:как|когда|куда|сколько|почему|котор\w*|зачем|где|что такое|подскажи\w*|расскажи\w*|науч\w*|объясни\w*|покажи\w*|помоги\w*|можешь|можно|скажите)\b")


def clean(text):
    return " ".join(text.casefold().strip(" \t\n—–«»„“”\"!?.,:").split())


def direct(text):
    """Label the utterance itself, refusing a salutation before another request."""
    q = clean(text)
    if not q or len(q) > 180 or META.search(q) or re.search(r"[.!?]\s+\w", q):
        return None
    if MULTI.search(q):
        return None
    words = WORD.findall(q)
    if not words:
        return None
    time = TIME.search(q)
    if time and any(w.startswith(("добр","спокойн")) for w in words):
        particular = time[0]
        if not all(w in VOCATIVE or w == particular or w.startswith(("добр","спокойн")) for w in words):
            return None
        if particular.startswith("ноч"):
            return "bye.night"
        subtype = "morning" if particular.startswith("утр") else "evening" if particular.startswith("вечер") else "day"
        return "greet." + subtype
    for expr, label in ((GREET,"greet"), (LEAVE,"bye"), (CALL,"call"), (THANK,"thanks"),
                        (CONGRAT,"celebrate"), (APOLOGY,"sorry")):
        found = expr.match(q)
        if not found:
            continue
        tail = WORD.findall(q[found.end():])
        if label in {"greet","bye"} and any(w not in VOCATIVE for w in tail):
            return None
        if label == "call" and tail and not ("здесь" in tail or "тут" in tail or "слышишь" in tail):
            return None
        if label == "sorry" and (re.search(r"\b(?:но|не могли бы|не подскаж\w*|не скаж\w*|что ты сказал)\b",q) or
                                  (text.rstrip().endswith("?") and tail)):
            return None
        if label == "thanks" and tail and not (tail[0] in VOCATIVE or tail[0] in {"за","тебя","вас","тебе","вам","от","огромнейшее"}):
            return None
        if label == "celebrate" and "не" in tail:
            return None
        if label == "greet" and found[0].endswith("йте"):
            return "greet.formal"
        if label == "bye" and found[0].startswith("прощ"):
            return "bye.final"
        return label
    return None


def detect(question):
    if len(question) > 650:
        return None
    q = question.casefold()
    if META.search(q) or EXCEPTION.search(q) or re.search(r"по-[а-яё]+ски\b|на [а-яё]+ языке",q):
        return None
    if ASK.search(q):
        if NEGATED.search(q):
            return None
        acts = []
        for quoted in re.findall(r"[«„“\"]([^«»„“”\"]+)[»“”\"]", q):
            label = direct(quoted)
            if label:
                acts.append(label)
        for speech in SPEECH.finditer(q):
            clause = re.split(r"[.!?;]|,\s*(?:что|как)\b",q[speech.end():],maxsplit=1)[0]
            label = direct(clause)
            if label:
                acts.append(label)
        for label, expression in REPORTS.items():
            reported = expression.search(q)
            if reported and re.search(r",?\s+но\b",q[reported.end():]):
                return None
            if reported and not any(act.split(".")[0] == label for act in acts):
                acts.append(label)
        unique = set(acts)
        if len(unique) == 1:
            return unique.pop(), True
        return None
    label = direct(question)
    return (label,False) if label else None


class Social:
    def __init__(self, data):
        self.models = data["models"]

    @classmethod
    def load(cls, directory=None):
        root = Path(directory) if directory is not None else Path(__file__).with_name("assets")
        with gzip.open(root/"social.json.gz","rt",encoding="utf-8") as stream:
            return cls(json.load(stream))

    @staticmethod
    def _decode(model):
        # Start, end, punctuation and word transitions are all observed.
        edges = model["edges"]
        pending = [(0., [], "^")]
        finished = []
        for _ in range(9):
            advanced = []
            for score, output, state in pending:
                next_counts = edges.get(state, {})
                total = sum(next_counts.values())
                for word, count in next_counts.items():
                    new_score = score + math.log(count / total)
                    if word == "$":
                        if output and output[-1] in {"!","."}:
                            finished.append((new_score / len(output)**.35, output))
                    elif word not in output:
                        seq = output + [word]
                        advanced.append((new_score,seq," ".join(seq[-2:])))
            pending = sorted(advanced, reverse=True)[:24]
        if not finished:
            return None
        _, words = max(finished)
        sentence = " ".join(words[:-1]) + words[-1]
        return sentence[:1].upper() + sentence[1:]

    def propose(self, question: str) -> tuple[str,float] | None:
        act = detect(question)
        if not act:
            return None
        label, reported = act
        model = self.models.get(label)
        if not model or model["observations"] < 2 and model["paired"] < 1:
            return None
        text = self._decode(model)
        if not text:
            return None
        confidence = .91 if model["paired"] else .78
        return text, confidence - (.04 if reported else 0)
