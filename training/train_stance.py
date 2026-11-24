"""Fit first-person lexical and polarity statistics from Russian CC0 sentences."""
import argparse
import bz2
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turing_bot.stance import ADJ, SINGLE, STOP, content, grams, stem, tokens

CC0_SHA256 = "fb9e878c3d673c7ec5d7ec64d3ec0ae80a7df89f846f41e150144ba4bee4971f"
FRAME = re.compile(r"\b(я (?:не )?(?:люблю|предпочитаю|обожаю)|мне (?:не )?нрав(?:ится|ятся))\s+([^,.;:!?«»\"—]+)", re.I)
BREAK = re.compile(r"\s+(?:и|а|но|хотя|когда|если|потому|поскольку|чтобы)\s+", re.I)
HEDGE = re.compile(r"^\s*(пожалуй|скорее)\b",re.I)
DIRECT = set("тебе тебя вами вам её его их вас я мое моё сам сама такие какой столь такой этот твое ваш ваши ваши наш мне пока же слишком буквально по-настоящему".split())


def train(archive, valid, output, role_type):
    if not SINGLE.fullmatch(role_type):
        raise ValueError("The stage role type must be a single Russian word.")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != CC0_SHA256:
        raise ValueError("The Russian CC0 snapshot SHA-256 differs.")
    excluded = {str(i["sentence_id"]) for line in valid.read_text(encoding="utf-8").splitlines()
                if "sentence_id" in (i := json.loads(line))}
    evidence = Counter()
    heads, edges, stems, chars = (defaultdict(Counter) for _ in range(4))
    hedges, frames, frequency, seen = Counter(), Counter(), Counter(), set()
    discourse = {
        "echo": (re.compile(r"(?:^|[«.!?]\s*)\s*(я что,)\s",re.I),Counter()),
        "contrast": (re.compile(r"\b(почему именно)\s",re.I),Counter()),
        "affirm": (re.compile(r"(?:^|[«.!?]\s*)\s*(да,)\s",re.I),Counter()),
    }
    with bz2.open(archive, "rt", encoding="utf-8") as stream:
        for line in stream:
            sid, lang, sentence, _ = line.rstrip("\n").split("\t")
            if lang != "rus" or sid in excluded:
                continue
            evidence["source_rows"] += 1
            frequency.update(tokens(sentence))
            for expression, counts in discourse.values():
                observed = expression.search(sentence)
                if observed:
                    counts[observed[1].casefold()] += 1
            hedge = HEDGE.match(sentence)
            if hedge:
                hedges[hedge[1].casefold()] += 1
            if len(sentence)>200 or "?" in sentence or "\n" in sentence:
                continue
            for found in FRAME.finditer(sentence):
                frame, tail = found.groups()
                frame = frame.casefold()
                words = BREAK.split(tail, 1)[0].strip().split()
                if not 1 <= len(words) <= 5 or any(not SINGLE.fullmatch(w) or w[0].isupper() for w in words):
                    continue
                if words[0] in STOP or words[-1] in STOP or not content(words) or any(w in DIRECT for w in words):
                    continue
                if ADJ.search(words[-1]):
                    continue
                key = (frame,tuple(words))
                if key in seen:
                    continue
                seen.add(key)
                negative = " не " in frame
                polarity = "-" if negative else "+"
                frames[frame] += 1
                heads[frame][words[0]] += 1
                evidence["negative_observations" if negative else "positive_observations"] += 1
                for word in set(content(words)):
                    stems[stem(word)][polarity] += 1
                    for gram in grams(word):
                        chars[gram][polarity] += 1
                for i in range(1,len(words)+1):
                    history = words[max(0,i-2):i]
                    next_word = words[i] if i<len(words) else "$"
                    edges[polarity+":"+" ".join(history)][next_word] += 1
    if not hedges:
        raise ValueError("No discourse markers were observed")
    spelling = Counter()
    for word, frequency_count in frequency.items():
        for gram in grams(word):
            spelling[gram] += frequency_count
    # A nominative and two observed case forms provide weak evidence of a
    # noun. No occupation, topic, or identity lexicon is supplied.
    nominals = []
    for word in sorted(frequency):
        if len(word) < 3:
            continue
        if re.search("[бвгджзклмнпрстфхцчшщ]$", word):
            root, endings = word, ("а", "у", "ом", "ы", "и", "ов", "ам")
        elif word.endswith(("а", "я")):
            root, endings = word[:-1], ("у", "ю", "е", "ы", "и", "ой", "ей")
        else:
            continue
        if sum(root + ending in frequency for ending in endings) >= 2:
            nominals.append(word)
    expressions = {name:counter.most_common(1)[0][0] for name,(_,counter) in discourse.items() if counter}
    if set(expressions) != set(discourse):
        raise ValueError("Corpus did not attest the supported discourse constructions.")
    data = {
        "counts": {word:[count["+"], count["-"]] for word,count in stems.items()},
        "character": {g:[c["+"], c["-"]] for g,c in chars.items() if sum(c.values())>=2},
        "objects": dict(heads), "edges": dict(edges), "frames": dict(frames), "hedges": dict(hedges),
        "frequency": {word:frequency[word] for counts in heads.values() for word in counts},
        "spelling": {g:n for g,n in spelling.items() if n>=2},
        "role_type":role_type, "nominals":nominals, "discourse":expressions,
        "source": "Tatoeba Russian CC0", "excluded_validation_ids":len(excluded), "observations":dict(evidence)
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    payload = json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    staged = output.with_suffix(".tmp")
    with staged.open("wb") as raw, gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as file:
        file.write(payload)
    staged.replace(output)
    print(json.dumps({"observations":dict(evidence),"distinct_stems":len(stems),"frames":dict(frames),
                      "hedges":dict(hedges),"excluded_validation_ids":len(excluded),
                      "sha256":hashlib.sha256(output.read_bytes()).hexdigest()},ensure_ascii=False))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cc0", required=True, type=Path)
    cli.add_argument("--valid", required=True, type=Path)
    cli.add_argument("--role-type", default="человек", help="One fictional role type, kept explicit in the model card.")
    cli.add_argument("--out", default=Path(__file__).resolve().parents[1]/"turing_bot/assets/stance.json.gz",type=Path)
    args=cli.parse_args()
    train(args.cc0,args.valid,args.out,args.role_type)
