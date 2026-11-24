"""Learn social-act response transitions from published Russian texts."""
import argparse
import bz2
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from turing_bot.social import WORD, direct

CC0_SHA = "fb9e878c3d673c7ec5d7ec64d3ec0ae80a7df89f846f41e150144ba4bee4971f"
MODERN_SHA = "429bcd376ffaa5390fc686dbe029a3c509caadee0de4f3cd2846118884a19875"
BASE_SHA = "10ee9952f7520d7ca344752fbe2936f31ef9f199787cdc16125dc43957cab4e0"
VALID_SHA = "27157b3f21aa2b8a197f1efbef4d3f1cd31661f7e4407051a2ca14761ab81a63"
SEPARATOR = re.compile(r"(?<=[.!?])\s+")
INFLECT = re.compile(r"\b(?:ваше благородие|государ\w*|милостив\w*|милорд|батюшка|сударын\w*)\b",re.I)


def lines(file, sha):
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"Unexpected training snapshot: {file.name}")
    return data.decode().splitlines()


def first_sentence(text):
    return SEPARATOR.split(text.strip(" «»„“”\"—–"),maxsplit=1)[0]


def suitable(text):
    text = first_sentence(text)
    match = re.fullmatch(r"([А-ЯЁа-яё]+(?:[ -][А-ЯЁа-яё]+){0,5})\s*([.!]?)",text.strip())
    if not match or any(word[0].isupper() for word in WORD.findall(match[1])[1:]) or INFLECT.search(match[1]):
        return None
    result = match[1].casefold()
    words = result.split()
    simple = direct(result)
    # These weak semantic labels filter observed replies; the recognizer
    # never supplies a sequence of response words to the decoder.
    label = None
    if simple and simple.startswith(("greet","bye")) and len(words) <= 3:
        label = simple
    elif simple == "thanks" and len(words) <= 3 and not set(words) & {"за", "нет"}:
        label = "gratitude"
    elif (len(words) == 1 and words[0].startswith("пожалуйст") or
          words[-1].startswith("помоч") and any(w.startswith("рад") for w in words) or
          len(words) <= 3 and "что" in words and words[:1] == ["не"]):
        label = "ack"
    elif (not any(w in {"никогда","нельзя","будет","ты","тебе","мне","он","она","у","но"} for w in words)
          and (any(w.startswith("прощаю") for w in words) and "не" not in words or
               words[-1].startswith("порядк") and words[0] in {"всё","все"} or
               words[0] == "ничего" and any(w.startswith("страшн") for w in words))):
        label = "reassure"
    elif len(words)<=3 and any(w.startswith("слушаю") for w in words) and not set(words) & {"не","его","её","их","музыку"}:
        label = "attend"
    if not label:
        return None
    # The model has no speaker-gender setting. A short acknowledgement can
    # omit a gendered past-tense or short-adjective form altogether.
    if label == "ack" and set(words) & {"рад","рада","рады","был","была","были"}:
        return None
    return label, tuple([*words, match[2] or "."])


def compatible(incoming, outgoing):
    if incoming.startswith(("greet","bye")):
        return incoming == outgoing
    return {"thanks":"ack", "sorry":"reassure","celebrate":"gratitude","call":"attend"}.get(incoming) == outgoing


def train(args):
    valid = {str(i["sentence_id"]) for raw in lines(args.valid,VALID_SHA) if "sentence_id" in (i:=json.loads(raw))}
    if hashlib.sha256(args.cc0.read_bytes()).hexdigest() != CC0_SHA:
        raise ValueError("Russian Tatoeba CC0 snapshot mismatch")
    # Saturate duplicate evidence, give an actual turn pair more influence
    # than the occurrence of a formula in an isolated complete utterance.
    paired, isolated = defaultdict(set), defaultdict(set)
    group_counts = Counter()

    def observe(text, source, context=None):
        observation = suitable(text)
        if not observation:
            return
        outgoing, sequence = observation
        if context:
            incoming = direct(first_sentence(context))
            if incoming and compatible(incoming,outgoing):
                paired[incoming].add((sequence,source))
        incoming = {"ack":"thanks","reassure":"sorry","gratitude":"celebrate","attend":"call"}.get(outgoing,outgoing)
        isolated[incoming].add((sequence,source))

    with bz2.open(args.cc0,"rt",encoding="utf-8") as stream:
        for raw in stream:
            sid,lang,text,_ = raw.rstrip("\n").split("\t")
            if lang != "rus" or sid in valid:
                continue
            source = "cc0:" + sid
            observe(text,source)
            quoted = list(re.finditer("«([^«»]+)»",text))
            for match in quoted:
                observe(match[1],source)
            for before,after in zip(quoted,quoted[1:]):
                if re.fullmatch(r"[.!?]?\s*[—–-]\s*",text[before.end():after.start()]):
                    observe(after[1],source,before[1])
            group_counts["cc0_rows"] += 1
    seen = set()
    for path,sha in ((args.train,MODERN_SHA),(args.base_train,BASE_SHA)):
        for raw in lines(path,sha):
            row = json.loads(raw)
            if row["source"] not in {"tatoeba_dialogue","everyday","rusdracor","oasst1","tatoeba_continuation"}:
                continue
            question, answer = row["context"].splitlines()[-1],row["response"]
            key = question.casefold(), answer.casefold()
            if key in seen:
                continue
            seen.add(key)
            source = row["group"]
            observe(question,source)
            # Adjacent same-speaker sentences may supply an expression,
            # never an observed prompt/reply transition.
            observe(answer,source,question if row["source"] != "tatoeba_continuation" else None)
            group_counts[row["source"]] += 1
    models = {}
    for incoming, observations in sorted(isolated.items()):
        edges = defaultdict(Counter)
        for seen_set,weight in ((observations,1),(paired[incoming],12)):
            for sequence,source in sorted(seen_set):
                state = "^"
                for end,word in enumerate((*sequence,"$"),1):
                    edges[state][word] += weight
                    state = " ".join(sequence[max(0,end-2):end])
        models[incoming] = {"edges":edges,"observations":len(observations),"paired":len(paired[incoming])}
    data = {"models":models,"rows":group_counts,"validation_ids":len(valid)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    packed=json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    temp=args.out.with_suffix(".tmp")
    with temp.open("wb") as raw, gzip.GzipFile(fileobj=raw,filename="",mtime=0,mode="wb") as output:
        output.write(packed)
    temp.replace(args.out)
    print(json.dumps({k:(m["observations"],m["paired"]) for k,m in models.items()},ensure_ascii=False))
    print(json.dumps(group_counts), hashlib.sha256(args.out.read_bytes()).hexdigest(), args.out.stat().st_size)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cc0",type=Path,default=ROOT/"training/data/rus_sentences_CC0.tsv.bz2")
    cli.add_argument("--train",required=True,type=Path)
    cli.add_argument("--base-train",required=True,type=Path)
    cli.add_argument("--valid",required=True,type=Path)
    cli.add_argument("--out",type=Path,default=ROOT/"turing_bot/assets/social.json.gz")
    train(cli.parse_args())
