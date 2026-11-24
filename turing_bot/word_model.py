"""Conditional word model. Only local transition and association counts are saved."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re

from .neural import detokenize, tokenize

BOS = "<s>"
EOS = "</s>"
RU = re.compile("[а-яё]")
COPY = "@"


def source_tokens(text):
    return [w for w in tokenize(text) if w.isalpha() or w.isdigit()][-12:]


def alignment(token, source):
    """Learn copy/inflection operations from surface matches, including unseen names."""
    if len(token) < 4 and not token.isdigit():
        return token
    best = None
    for offset, word in enumerate(reversed(source)):
        common = 0
        for a, b in zip(word, token):
            if a != b:
                break
            common += 1
        if token != word and (common < 4 or len(word)-common > 2 or len(token)-common > 2):
            continue
        tail = word[common:]
        if best is None or common > best[0]:
            best = (common, f"@{offset}:{tail}:{token[common:]}")
    return best[1] if best else token


def realize(symbol, source):
    if not symbol.startswith(COPY):
        return symbol
    offset, old, new = symbol[1:].split(":")
    if int(offset) >= len(source):
        return None
    word = source[-1 - int(offset)]
    if not word.endswith(old) or (len(word) < 4 and not word.isdigit()):
        return None
    return (word[:-len(old)] if old else word) + new


def features(text):
    words = [w.replace("ё", "е") for w in tokenize(text) if w.isalpha()]
    stems = [w[:5] for w in words]
    return set(stems) | {a + "/" + b for a, b in zip(stems, stems[1:])}


def fit(records, destination):
    transitions = defaultdict(Counter)
    document_frequency, target_frequency = Counter(), Counter()
    links = defaultdict(Counter)
    opening_links = defaultdict(Counter)
    count = 0
    for sample in records:
        target = tokenize(sample["response"])
        if not target or not any(RU.search(t) for t in target):
            continue
        source = features(sample["context"].splitlines()[-1])
        count += 1
        unique = {t for t in target if t.isalpha()}
        target_frequency.update(unique)
        document_frequency.update(source)
        for f in source:
            links[f].update(unique)
            opening_links[f][target[0]] += 1
        source_words = source_tokens(sample["context"].splitlines()[-1])
        aligned = [alignment(t, source_words) for t in target]
        for sequence in ([target, aligned] if aligned != target else [target]):
            prefix = [BOS, BOS]
            for token in [*sequence, EOS]:
                transitions[" ".join(prefix[-2:])][token] += 1
                transitions[prefix[-1]][token] += 1
                prefix.append(token)
    associations = {}
    for f, counts in links.items():
        n = document_frequency[f]
        if n >= max(20, count * .35):
            continue
        weights = {}
        for t, k in counts.items():
            # Positive pointwise mutual information, shrunk toward zero for rare events.
            ratio = (k / n) / (target_frequency[t] / count)
            value = math.log(max(1., ratio)) * (k / (k + 2.5)) * min(1., math.sqrt(n / 3))
            if "/" not in f:
                value *= min(1., (k / 3)**1.5)
            if value > .35:
                weights[t] = round(value, 3)
        if weights:
            associations[f] = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:100]
    opening = {f: counts.most_common(40) for f, counts in opening_links.items() if len(counts) > 1}
    asset = {"format": 1, "transitions": {p: c.most_common(300 if p == BOS+" "+BOS else 100) for p,c in transitions.items()},
             "opening": opening,
             "links": associations, "frequency": dict(document_frequency), "total": count}
    payload = json.dumps(asset, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with open(destination, "wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", compresslevel=6, mtime=0) as compressed:
            compressed.write(payload)


class WordModel:
    def __init__(self, data):
        if data["format"] != 1:
            raise ValueError("Несовместимая версия словесной модели")
        self.transitions = data["transitions"]
        self.links = data["links"]
        self.frequency = data["frequency"]
        self.opening = data.get("opening", {})
        self.total = data["total"]

    @classmethod
    def load(cls, directory=None):
        path = Path(directory) if directory else Path(__file__).with_name("assets")
        with gzip.open(path / "words.json.gz", "rt", encoding="utf-8") as inp:
            return cls(json.load(inp))

    def cues(self, context):
        result = Counter()
        for age, line in enumerate(reversed(context.splitlines()[-3:])):
            strength = (1., .15, .12)[age]
            for f in sorted(features(line)):
                if f not in self.links:
                    continue
                reliability = min(1.5, math.log1p(self.total / self.frequency[f]) / 5)
                weight = strength * reliability * (1.4 if "/" in f else 1.)
                for token, score in self.links[f]:
                    result[token] += weight * score
        peak = max(result.values(), default=1.)
        return {t: min(6., v, 6 * v / max(1., peak)) for t, v in result.items()}

    def fluency(self, tokens):
        prefix = [BOS, BOS]
        value = 0.
        for token in [*tokens, EOS]:
            edges = self.transitions.get(" ".join(prefix[-2:]), self.transitions.get(prefix[-1], []))
            n = sum(k for _, k in edges)
            count = next((k for w,k in edges if w == token), 0)
            value += math.log((count + .01) / max(n + 1, 1))
            prefix.append(token)
        return value / (len(tokens) + 1)

    def candidates(self, context, width=24, limit=25):
        cues = self.cues(context)
        source = source_tokens(context.splitlines()[-1])
        beams = [(0., (BOS, BOS))]
        finished = {}
        initial = Counter()
        for f in sorted(features(context.splitlines()[-1])):
            if f not in self.opening:
                continue
            n = self.frequency[f]
            weight = max(0., math.log(self.total / n)) * (1.3 if "/" in f else 1.)
            for t, k in self.opening[f]:
                initial[t] += weight * k / (n + 6)
        peak = max(initial.values(), default=1.)
        for position in range(limit):
            next_beams = []
            for score, history in beams:
                choices = self.transitions.get(" ".join(history[-2:]))
                if not choices:
                    choices = self.transitions.get(history[-1], [])
                total = sum(n for _, n in choices)
                for token, n in choices:
                    surface = realize(token, source)
                    if surface is None:
                        continue
                    if token == EOS:
                        if len(history) >= 6:
                            words = history[2:]
                            phrase = detokenize([realize(t, source) for t in words])
                            if RU.search(phrase.lower()) and len(phrase) <= 300:
                                final = (score + math.log(n / total)) / (len(words) + 1)**.78
                                finished[phrase] = max(finished.get(phrase, -math.inf), final)
                        continue
                    if history[-1] in ".!?" and history[-1] != BOS:
                        continue
                    if token in history[-2:] or token in ("<unk>", "«", "»", "(", ")"):
                        continue
                    if any(history[i:i+3] == (*history[-2:], token) for i in range(1, len(history)-2)):
                        continue
                    value = score + math.log(n / total) - .4
                    if token not in history:
                        value += .95 * cues.get(surface, 0.)
                    if position == 0:
                        value += 2.5 * initial[surface] / peak
                    if position >= 14:
                        value -= .5
                    next_beams.append((value, (*history, token)))
            if not next_beams:
                break
            next_beams.sort(reverse=True)
            beams = next_beams[:width]
        return sorted(finished.items(), key=lambda pair: -pair[1])[:14]
