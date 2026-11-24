"""Fit the pair discriminator, mine discourse constituents and audit its operating point."""
import argparse
import bz2
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np

from turing_bot.grounding import Semantic, encode, features, normal
from turing_bot.neural import Generator


def rows(path):
    with path.open(encoding="utf8") as stream:
        return [json.loads(s) for s in stream if s.strip()]


def association(examples):
    dfq, dfa, overlap = Counter(), Counter(), Counter()
    for item in examples:
        q, a = encode(item["context"]).content, encode(item["response"]).content
        dfq.update(q)
        dfa.update(a)
        overlap.update((x, y) for x in q for y in a)
    links = defaultdict(dict)
    for (q,a), n in sorted(overlap.items()):
        if n < 3:
            continue
        lift = math.log1p(n*len(examples)/max(1,dfq[q]*dfa[a])) * n/(n+5)
        if lift > .6:
            links[q][a] = round(lift, 4)
    idf = {w: round(math.log((1+len(examples))/(1+v)), 4) for w,v in dfq.items() if v >= 2}
    return dict(links), idf


def design(examples, links, idf, semantic, seed):
    rng = random.Random(seed)
    answers = [encode(x["response"]) for x in examples]
    by_source, by_question_word, by_reply_word = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, (row, a) in enumerate(zip(examples, answers)):
        by_source[row["source"]].append(i)
        for w in sorted(encode(row["context"]).content):
            by_question_word[w].append(i)
        for w in sorted(a.content):
            by_reply_word[w].append(i)
    matrix, label = [], []
    for i, row in enumerate(examples):
        q, a = encode(row["context"]), answers[i]
        # Match genre for the easy negative. A hard distractor answers a
        # different question about a similar subject. Force literal overlap
        # only if the observed response itself has some, to avoid teaching
        # that repeating the user's topic is evidence of an incorrect reply.
        pool = by_source[row["source"]]
        index = by_reply_word if q.content & a.content and rng.random() < .4 else by_question_word
        topical = [j for w in sorted(q.content) if len(index[w]) < 180 for j in index[w]
                   if examples[j]["group"] != row["group"]]
        used = {a.plain}
        qtext = row["context"].splitlines()[-1]
        matrix.append((*features(q, a, links, idf), semantic.fit(qtext,row["response"])))
        label.append(1)
        for candidates in (pool, topical or pool):
            for _ in range(20):
                j = rng.choice(candidates)
                if examples[j]["group"] != row["group"] and answers[j].plain not in used:
                    used.add(answers[j].plain)
                    matrix.append((*features(q, answers[j], links, idf), semantic.fit(qtext,examples[j]["response"])))
                    label.append(0)
                    break
    return np.array(matrix, dtype=np.float64), np.array(label)


def discourse(path, rejected):
    # Capture reusable constituents of real utterances. Inflected copy slots
    # and their punctuation are produced by the transducer, never indexed Q/A.
    patterns = {
        "unknown": r"^(я не (?:знаю|уверен)), (?:что|как|когда|почему|где|зачем)",
        "verify": r"^(правда ли), что", "origin":r"^(откуда) ",
        "known":r"\b(известно)\b", "what":r"^(что) ",
        "information":r"\b(информация)\b",
        "meaning":r"\b(что ты имеешь в виду)\b",
        "goal":r"\b(какая у тебя цель)\b",
        "more":r"^(ты можешь)(?: еще что-нибудь)? (рассказать)\b",
        "example":r"\b(привести) (пример)(?:ы|а|ов)?\b",
        "request":r"^(ты можешь) ",
        "what_have":r"^(что у тебя) ",
    }
    counts = {k: Counter() for k in patterns}
    expression = {k:re.compile(v) for k,v in patterns.items()}
    with bz2.open(path, "rt", encoding="utf8") as stream:
        for line in stream:
            fields = line.rstrip().split("\t")
            if len(fields)<3 or fields[1]!="rus" or fields[0] in rejected:
                continue
            text = normal(fields[2])
            for key, pattern in expression.items():
                match = pattern.search(text)
                if match:
                    counts[key][" ".join(match.groups())] += 1
    missing = [k for k,v in counts.items() if not v]
    if missing:
        raise ValueError(f"Missing discourse constituents in corpus: {missing}")
    return {k:v.most_common(1)[0][0] for k,v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("turing_bot/assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    train, valid = rows(args.train), rows(args.valid)
    group_sets = {key for key in (r["group"] for r in train)
                  if int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little") % 5 < 3}
    association_rows = [r for r in train if r["group"] in group_sets]
    learning_rows = [r for r in train if r["group"] not in group_sets]
    links, idf = association(association_rows)
    semantic = Semantic(Generator.load(args.model))
    x, y = design(learning_rows, links, idf, semantic, 22461)
    mean, deviation = x.mean(axis=0), x.std(axis=0).clip(.05)
    standardized = (x-mean) / deviation
    beta, intercept = np.zeros(x.shape[1]), 0.
    for _ in range(300):
        pred = 1/(1+np.exp(-np.clip(intercept + standardized@beta,-30,30)))
        residual = pred-y
        beta -= .11*(standardized.T@residual / len(y) + .01*beta)
        intercept -= .11*residual.mean()
    coeff = beta/deviation
    intercept -= mean@coeff
    # Partition by source group before sampling. No book/dialogue/tree spans both.
    selection, measurement = [], []
    for r in valid:
        bit = hashlib.sha256(r["group"].encode()).digest()[0] % 2
        (selection if bit == 0 else measurement).append(r)
    sx, sy = design(selection, links, idf, semantic, 35549)
    mx, my = design(measurement, links, idf, semantic, 58261)
    vx, vy = np.concatenate((sx,mx)), np.concatenate((sy,my))
    scores = 1/(1+np.exp(-np.clip(vx@coeff+intercept,-30,30)))
    # First half of validation selects a precision-prioritising threshold.
    # Report the second half as a separate, unoptimised measurement.
    cut = len(sy)
    options = []
    for thr in np.arange(.35, .995, .005):
        fire = scores[:cut]>=thr
        if fire.sum() >= 8 and vy[:cut][fire].mean() >= .8:
            options.append(float(thr))
    threshold = min(options, default=.95)
    rankings = []
    for name, s, truth in (("threshold_selection",scores[:cut],vy[:cut]), ("measurement",scores[cut:],vy[cut:])):
        fire = s >= threshold
        tp = int(truth[fire].sum())
        rankings.append({"partition":name, "size":len(truth), "selected":int(fire.sum()), "true_selected":tp,
                         "precision":round(tp/max(1,int(fire.sum())),4),
                         "recall":round(tp/max(1,int(truth.sum())),4),
                         "accuracy":round(float(np.mean(fire==truth)),4),
                         "mean_positive":round(float(np.mean(s[truth==1])),4),
                         "mean_negative":round(float(np.mean(s[truth==0])),4)})
    fragments = discourse(args.archive, {str(r["sentence_id"]) for r in valid if "sentence_id" in r})
    model = {"format":"arena-pair-contrast-1", "idf":idf, "association":links, "coefficients":coeff.tolist(),
             "intercept":float(intercept), "threshold":round(threshold,4), "discourse":fragments}
    out = args.output/"confidence.json.gz"
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("wb") as buf, gzip.GzipFile(filename="",fileobj=buf,mode="wb",mtime=0) as f:
        f.write(json.dumps(model, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode())
    report = {"association_rows":len(association_rows), "classifier_training_rows":len(learning_rows),
              "labeled_classifier_pairs":len(y), "original_validation_rows":len(valid),
              "association_edges":sum(len(v) for v in links.values()),"threshold":model["threshold"], "validation":rankings,
              "asset_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
              "training_sha256":hashlib.sha256(args.train.read_bytes()).hexdigest(),
              "validation_sha256":hashlib.sha256(args.valid.read_bytes()).hexdigest(),
              "neural_sha256":hashlib.sha256((args.model/"model.npz").read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report))


if __name__=="__main__":
    main()
