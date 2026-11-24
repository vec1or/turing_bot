"""Conditional first-person morphology and learned preference counts."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import re
import unicodedata


WORD = r"[а-яё]+(?:-[а-яё]+)*"
SINGLE = re.compile(WORD + r"$", re.I)
STOP = set("""я ты вы мы он она оно они мой твой свой ваш наш
    мои твои свои свою моего этот эта это эти тот та то те такой такие
    мне ли не бы в во на по у с со к за до без из о об от про для чем и а
    что его ее их какой какая какое какие какой-то иногда очень именно
    больше всего просто еще ещё лучше или сама сам здесь там вообще
    больше-то любой каждый всю весь разные разное""".split())
PREF = r"(?:любишь|любите|обожаешь|предпочитаешь|предпочитаете|предпочт[её]шь|выберешь|выбираешь|выберете|выбрала? бы|нравится|нравятся|ближе|приятнее)"
OPEN = re.compile(rf"^(?:а )?что (?:бы )?(?:ты |вы |тебе |вам )?(?:больше всего |особенно |обычно )?{PREF}(?: (?:делать|больше всего|больше))?$")
YES = re.compile(rf"^(?:а )?(?:(?:ты|вы) (?:и правда |вообще )?((?:не )?{PREF})|(?:тебе|вам) ((?:не )?нрав(?:ится|ятся))) (.+)$")
FEELING = re.compile(r"^(?:(?:а|ну) )?(?:ты|вы) ([а-яё-]+(?:ешь|ишь|ёшь))(?: ли)?$")
LEFT_FRAME = re.compile(
    rf"^(?:а )?(?:(?:что|чему) )?(?:бы )?(?:(?:ты|тебе|вам|вы) )?"
    rf"(?:(?:больше|обычно|лично|скорее) )?(?:(?:бы )?(?:выбрал[аи]?|предпоч(?:[её]л|ла|ли))|{PREF})(?: (?:больше|обычно|ты|лично))?"
    rf"(?: (?:из двух|для себя))?\s*[:,—-]?\s+",
    re.I)
CHOICE_TAIL = re.compile(rf"\s*[:,—-]\s*(?:что )?(?:ты |тебе |вам |вы )?{PREF}\s*$",re.I)
WORD_END = re.compile(r"(?:ешь|ишь|ёшь|ется|ётся|утся|ятся|ешься|ишься|ите|ете|ает|яет|ует|вают|ют)$")
INF = re.compile(r"(?:ть|ти|чь)(?:ся|сь)?$")
NP_REJECT = set("кто кому где когда почему зачем что как сколько ли нет да почему-то вместо если это значит можно нужно будет был".split())
ADJ = re.compile(r"(?:ый|ий|ая|яя|ое|ее|ые|ие|ых|их|ую|юю|ого|его|ому|ему)$")
NOMINAL_QUERY = re.compile(r"^(?:а )?ты (?:(?:точно|случайно|правда|вообще|действительно|всё-таки|все-таки) )?(?:не )?(.+)$")
NOUN_END = re.compile(r"(?:ник|ница|щица|тель|ист|ент|ант|арь|яр|ьер|[её]р|ор|ар|лог|граф|тив|щик|чик|ушка|юшка|щина)$")
STATE = set("""рад рада рады прав права правы готов готова готовы уверен уверена уверены
    болен больна занят занята занятно заняты согласен согласна согласны устал устала
    устали дома здесь там один одна вместе жив жива живы свой против над до
    скоро далеко рядом близко здоров здорова сердит сердита трезв трезва сыт сыта""".split())


def tokens(text):
    return re.findall(WORD, text.casefold())


def stem(word):
    word = word.casefold().replace("ё", "е")
    if len(word) < 5:
        return word
    # Conservative inflection folding; retain at least four letters.
    endings = ("ешься", "ишься", "ться", "иями", "ями", "ами", "ого", "его", "ому",
               "ему", "ать", "ять", "ешь", "ишь", "иях", "ью", "ей", "ов",
               "ам", "ям", "ах", "ях", "ой", "ом", "ем", "ые", "ие", "ую", "юю",
               "ая", "яя", "ия", "ий", "а", "я", "ы", "и", "у", "ю", "е")
    for ending in endings:
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[:-len(ending)]
    return word


def grams(word):
    word = "^" + word.replace("ё","е") + "$"
    return {word[i:i+3] for i in range(len(word)-2)}


def content(words):
    return [w for w in words if len(w) > 2 and w not in STOP]


def option(text):
    words = tokens(text)
    if not re.fullmatch(rf"{WORD}(?: {WORD}){{0,4}}", text, re.I):
        return None
    if set(words) & NP_REJECT or any(WORD_END.search(w) for w in words):
        return None
    if not content(words) or words[-1] in STOP:
        return None
    return words


class Stance:
    def __init__(self, data):
        self.counts = data["counts"]
        self.character = data["character"]
        self.objects = data["objects"]
        self.frames = data["frames"]
        self.edges = data["edges"]
        self.hedges = data["hedges"]
        self.frequency = data["frequency"]
        self.spelling = data["spelling"]
        self.role = data["role_type"]
        self.nominals = frozenset(data["nominals"])
        self.discourse = data["discourse"]

    @classmethod
    def load(cls, directory=None):
        directory = Path(directory) if directory is not None else Path(__file__).with_name("assets")
        with gzip.open(directory / "stance.json.gz", "rt", encoding="utf-8") as stream:
            return cls(json.load(stream))

    def _preference(self, words):
        # Character features supply a weak back-off only for explicit alternatives.
        strength = 0
        score = 0.
        for word in content(words):
            positive, negative = self.counts.get(stem(word), (0, 0))
            strength += positive + negative
            score += math.log1p(positive) - math.log1p(negative)
            char_counts = [self.character[g] for g in grams(word) if g in self.character]
            if char_counts:
                score += .16 * sum((p-n)/(p+n+4) for p,n in char_counts)/len(char_counts)
        return score, strength

    def _choose(self, plain):
        if plain.count(" или ") != 1 or re.search(r"\b(?:не|ли|он|она|они|мы|я)\b", plain):
            return None
        left, right = plain.split(" или ")
        # For a free-standing nominal contrast the answer is elliptical.
        # Copying the selected noun phrase preserves its Russian case.
        head = LEFT_FRAME.match(left)
        tail = CHOICE_TAIL.search(right)
        if head and re.search(r"\b(?:выбрал[аи]?|предпоч(?:[её]л|ла|ли))\b", head[0]) and "бы" not in tokens(head[0]):
            return None
        left = left[head.end():] if head else left
        right = right[:tail.start()] if tail else right
        first, second = option(left), option(right)
        if not first or not second or set(first + second) & {"ты", "вы", "тебе", "вам"}:
            return None
        if first == second or (not (head or tail) and (len(first) > 2 or len(second) > 2)):
            return None
        one, a = self._preference(first)
        two, b = self._preference(second)
        only_spelling = False
        if abs(one-two) < .001 and a+b == 0:
            # Two unobserved options may also tie on the small attitude
            # sample. Prefer the more familiar letter sequences in this
            # case, and expose this weaker basis to the calling engine.
            def familiarity(words):
                features = [g for w in content(words) for g in grams(w)]
                return sum(math.log1p(self.spelling.get(g,0)) for g in features)/max(1,len(features))
            one, two = familiarity(first), familiarity(second)
            only_spelling = True
        if abs(one - two) < .001:
            return None
        selected = left if one > two else right
        # Both options may be previously unseen; then this is a weak
        # morphological preference, not knowledge of either object.
        base = .50 if (head or tail) else .43
        confidence = min(.91, base + .09*min(a+b,3) + .12*min(abs(one-two),1))
        if only_spelling:
            confidence = base - .025
        selected_words = first if one > two else second
        if head and max(a,b) >= 2:
            extended = self._opinion(content(selected_words))
            if extended and " не " not in extended[0].casefold():
                return extended[0], confidence
        if head and re.search(r"\b(?:любишь|любите|предпочитаешь|предпочитаете|предпочла|предпоч[её]л|выберешь|выбираешь|выбрала?)\b", head[0]):
            # The options already carry the object case of the question verb.
            frame = "я предпочитаю" if "я предпочитаю" in self.frames else None
            if frame:
                return frame.capitalize() + " " + selected + ".", confidence
        hedge = max(self.hedges, key=lambda h: (self.hedges[h], h))
        separator = ", " if hedge == "пожалуй" else " "
        return hedge.capitalize() + separator + selected + ".", confidence

    def _phrase(self, first, polarity):
        # Walk observed word transitions; end markers learn where a Russian
        # complement can finish. No question/answer strings are stored.
        words = [first]
        for _ in range(4):
            links = self.edges.get(polarity + ":" + " ".join(words[-2:]), {})
            if not links:
                break
            valid = [(n, word) for word,n in links.items() if word == "$" or word not in words]
            if not valid:
                break
            count, word = max(valid, key=lambda item: (item[0], item[1] != "$", item[1]))
            if word == "$":
                break
            words.append(word)
        if words[-1] in STOP or ADJ.search(words[-1]):
            return None
        return words

    def _opinion(self, queried, motion=False):
        target = {stem(w) for w in queried}
        candidates = []
        for frame, firsts in self.objects.items():
            sign = "-" if " не " in " " + frame + " " else "+"
            # Generic "what do you like?" asks for a positive inclination.
            if not target and sign == "-":
                continue
            for first, n in firsts.items():
                if motion and not INF.search(first):
                    continue
                phrase = self._phrase(first, sign)
                if not phrase:
                    continue
                filled = {stem(w) for w in content(phrase)}
                if motion:
                    if queried[0][:-3].replace("ё","е") != first[:-2].replace("ё","е"):
                        continue
                elif target and not target <= filled:
                    continue
                if not target and (len(filled) < 2 or not INF.search(first)):
                    continue
                score, support = self._preference(phrase)
                signed = score if sign == "+" else -score
                if signed < .55 or support < 2:
                    continue
                # With a supplied object add content where the corpus permits.
                more = len(filled - target)
                if target and not more:
                    continue
                rank = signed / math.sqrt(max(1,len(filled))) + math.log1p(n)
                rank += .48 * math.log1p(self.frequency.get(first,0)) + .12 * min(more, 2)
                text = frame.capitalize() + " " + " ".join(phrase) + "."
                candidates.append((rank, text, support))
        if not candidates:
            return None
        _, text, support = max(candidates)
        return text, min(.91, .70 + .035 * min(support, 6))

    def _nominal(self, text):
        # Narrow nominative predication only. Verbal, adverbial, spatial,
        # possessive and copular past-tense descriptions go to other models.
        parts = text.split()
        if len(parts) == 2 and not ADJ.search(parts[0]):
            return None
        if not 1 <= len(parts) <= 2 or not all(SINGLE.fullmatch(w) for w in parts):
            return None
        noun = parts[-1]
        if (noun in STOP | STATE or len(noun) < 3 or WORD_END.search(noun) or INF.search(noun)
                or re.search(r"(?:лся|лась|лись|али|или|ела|ила|ала|ён|ена|ено)$", noun)):
            return None
        if noun == self.role or noun in self.nominals or (len(noun)>=4 and NOUN_END.search(noun)):
            return text
        return None

    def _role_answer(self, plain):
        match = NOMINAL_QUERY.fullmatch(plain)
        if not match:
            return None
        phrases = match[1].split(" или ")
        if len(phrases) > 2 or not all(self._nominal(p) for p in phrases):
            return None
        for phrase in phrases:
            supplied_type = phrase.split()[-1]
            if supplied_type == self.role:
                # Elliptical noun agrees with the supplied alternative.
                # Repeat only the role type; modifiers may add claims.
                if len(phrases) == 2 or " не " in plain:
                    return supplied_type.capitalize() + ".", .95
                return self.discourse["affirm"].capitalize() + " " + supplied_type + ".", .95
        if len(phrases) == 2:
            return self.discourse["contrast"].capitalize() + " " + " или ".join(phrases) + "?", .73
        return self.discourse["echo"].capitalize() + " " + phrases[0] + "?", .77

    def propose(self, question: str) -> tuple[str, float] | None:
        normalized = unicodedata.normalize("NFC", question).strip()
        if normalized.endswith("?"):
            # Earlier complete sentences may supply a preamble. The final
            # clause still has to satisfy a supported question construction.
            normalized = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁа-яё])", normalized)[-1]
        plain = " ".join(normalized.casefold().split())
        if not plain or len(plain) > 200 or re.search(r"[.!?\n]", plain.rstrip("?")):
            return None
        plain = plain.rstrip("?").strip()
        plain = re.sub(r"^(любишь|любите|предпочитаешь|предпочитаете) (?:ли )?(ты|вы) ",
                       r"\2 \1 ", plain)
        role_answer = self._role_answer(plain)
        if role_answer:
            return role_answer
        if " или " in plain:
            return self._choose(plain)
        if OPEN.fullmatch(plain):
            return self._opinion([])
        found = YES.fullmatch(plain)
        if found:
            words = option(found.group(3))
            if words:
                return self._opinion(content(words))
        found = FEELING.fullmatch(plain)
        if found:
            return self._opinion([found[1]], motion=True)
        return None
