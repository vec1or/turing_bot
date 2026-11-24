"""Learn Russian phrase completions from plain sentences, without dialogue labels."""
from collections import Counter, defaultdict
import bz2
import gzip
import json
import math
from pathlib import Path
import re

from .neural import detokenize, tokenize

END = "#"
RUSSIAN = re.compile(r"[а-яё-]+$")
PREPS = {"в", "во", "на", "за", "под", "над", "между", "у", "около", "возле", "дома", "там"}
INSTR = re.compile(r"(?:ом|ем|ём|ой|ей|ью|ами|ями|ым|им|ою|ею)$")
LIGHT = set("чему чего каким какой какая какие какое какому какого какую можно может такой такое если когда почему кажется тебе меня тебя очень только всем этот этой этого быть было были чтобы который вообще тогда всегда просто спросить можешь считаешь думаешь будет была могло здесь также хотел хотела назвать придумай представь".split())
NOISE = re.compile(r"(?:том(?:а|у|ом)?|мэри|бостон|джон|татое[бп]а|\w*убий\w*|убить|\w*пистолет\w*|\w*насил\w*|суицид\w*|\w*наркот\w*|дурак\w*|идиот\w*|\w*трах\w*)", re.I)


def train(archive, path, exclude_ids=()):
    heads, steps = defaultdict(Counter), defaultdict(Counter)
    word_count = Counter()
    count = 0
    with bz2.open(archive, "rt", encoding="utf-8") as rows:
        for line in rows:
            fields = line.rstrip().split("\t")
            if len(fields) < 3 or fields[1] != "rus" or fields[0] in exclude_ids:
                continue
            text = fields[2]
            if len(text) > 180 or "?" in text or any(c in text for c in "«»():;[]0123456789/"):
                continue
            words = tokenize(text)
            if words and words[-1] in ".!":
                words.pop()
            if not 3 <= len(words) <= 18 or not all(RUSSIAN.fullmatch(w) and not NOISE.fullmatch(w) for w in words):
                continue
            count += 1
            word_count.update(words)
            for i, word in enumerate(words[:-1]):
                if len(word) >= 4 and word not in LIGHT:
                    heads[word][words[i+1]] += 1
                    heads["~"+word[:4]][words[i+1]] += 1
            for i in range(1,len(words)):
                following = words[i+1] if i+1 < len(words) else END
                steps[words[i-1] + " " + words[i]][following] += 1
                steps[words[i]][following] += 1
    payload = {"sentences": count, "frequency": dict(word_count),
               "heads": {k: v.most_common(90) for k,v in heads.items()},
               "steps": {k: v.most_common(55) for k,v in steps.items()}}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    with open(path, "wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as dest:
            dest.write(data)
    return count


def query_shape(words):
    """Russian interrogative case restricts the possible form of the missing constituent."""
    for i, w in enumerate(words):
        if w in {"что","чём","чем","чего","кем","ком","кого"} and i and words[i-1] in PREPS | {"к","о","об","от","из","с","без","про"}:
            return {words[i-1]}, False
        if w in {"чем","кем"}:
            return set(), True
        if w in {"где","куда","откуда"}:
            return (PREPS if w != "откуда" else {"из", "от", "с"}), False
    return None


class Cloze:
    @classmethod
    def load(cls, directory=None):
        path = Path(directory) if directory else Path(__file__).with_name("assets")
        with gzip.open(path/"cloze.json.gz","rt",encoding="utf-8") as data:
            return cls(json.load(data))

    def __init__(self, data):
        self.heads, self.steps, self.frequency = data["heads"], data["steps"], data["frequency"]

    def complete(self, question):
        words = tokenize(question)
        shape = query_shape(words)
        if shape is None:
            return []
        prepositions, instrumental = shape
        options = []
        seen_pivots=set()
        for i, word in enumerate(words[-64:]):
            if len(word) < 4 or word in LIGHT or not RUSSIAN.fullmatch(word):
                continue
            if word in seen_pivots:
                continue
            seen_pivots.add(word)
            if len(seen_pivots)>14:
                break
            exact = self.heads.get(word, [])
            counts = exact or self.heads.get("~"+word[:4], [])
            total = sum(n for _,n in counts)
            starts = [(w,n) for w,n in counts if (w in prepositions or instrumental and INSTR.search(w))]
            if not starts:
                continue
            support = sum(n for _,n in starts) / (total + 2)
            support *= 1 if exact else .55
            support *= 1.45 if i and words[-64:][i-1] in {"чем","кем","где","куда"} else 1.
            for w,n in starts:
                beams = [(math.log(n / (1+sum(v for _,v in starts))), [w])]
                for length in range(8):
                    new = []
                    for s, seq in beams:
                        prev = seq[-2] if len(seq)>1 else word
                        edge = self.steps.get(prev+" "+seq[-1], self.steps.get(seq[-1], []))
                        den = sum(freq for _,freq in edge)
                        for nxt,freq in edge:
                            score = s + math.log(freq / den)
                            if nxt == END:
                                if seq[-1] not in PREPS and (not instrumental or INSTR.search(seq[-1])):
                                    rank = math.log(support) + score/(len(seq)**.65) - .13*len(seq)
                                    options.append((detokenize(seq),rank))
                            elif nxt not in seq and nxt not in {"и","но","что","потому","который","чтобы"}:
                                new.append((score, seq+[nxt]))
                    beams = sorted(new,reverse=True)[:4]
                    if not beams:
                        break
        seen = set()
        result = []
        for phrase, score in sorted(options,key=lambda x:-x[1]):
            if phrase not in seen:
                seen.add(phrase)
                result.append((phrase, score))
        return result[:4]
