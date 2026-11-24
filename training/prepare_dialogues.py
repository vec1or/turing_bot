"""Build (context, reply) pairs for the dual encoder and the reply index.

Sources (downloaded beforehand, see DATA_SOURCES.md):
  * Den4ikAI/russian_dialogues — Telegram question/answer pairs, MIT. Only rows
    labelled relevance=1 are used.
  (morzecrew/RefinedPersonaChat was inspected and rejected: its dialogues are
    machine-generated, which the contest rules and the bot's style both exclude.)
  * Den4ikAI/russian_dialogues_2 — comment threads (1.7 M), MIT. Consecutive
    turns become pairs; only pairs whose context is a question enter the index.
  * inkoziev/Conversations (RLDD) — dialogues from jokes, books and forums,
    CC BY-NC 4.0. Used for encoder training only, never as index replies.

Outputs (JSONL, fields c = context, p = previous turn, r = reply, s = source):
  train.jsonl, valid.jsonl  — encoder training/validation (split by hash of context)
  index.jsonl               — candidate replies for the bot, stricter filters

    python -m training.prepare_dialogues --raw RAW_DIR --out OUT_DIR
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import pyarrow.parquet as pq

from turing_bot.retrieval import persona_safe

CYR = re.compile(r"[а-яё]", re.I)
LATIN = re.compile(r"[a-z]{3,}", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
PROFANE = re.compile(r"(?:\b|^)(?:ху[йеёияю]|пизд|п[иі]зд|еб[аул]|ёб|ебл|бля|сук[аи]|мудак|мудил|пид[оа]р|гандон|шлюх|залуп|дроч|хер[ау]?\b|жоп|говн|срал|сран|дерьм|нахер|нахуй|долбо|дебил|идиот|тупой|тупая|урод|чмо|лох|даун)", re.I)
SENSITIVE = re.compile(r"\b(?:путин\w*|навальн\w*|зеленск\w*|войн\w*|украин\w*|хохл\w*|москал\w*|нацист\w*|террор\w*|наркот\w*|секс\w*|порн\w*|голая|голый|убить|убью|самоубий\w*|суицид\w*|бог[ауе]?|аллах\w*|религи\w*|выбор\w* президент\w*)\b", re.I)
JUNK = re.compile(r"https?:|www\.|t\.me|@|#|\d{4,}|[<>{}\[\]|\\_*=~^]|\.{4,}")


def clean(text):
    text = unicodedata.normalize("NFC", str(text or ""))
    text = EMOJI.sub("", text)
    text = re.sub(r"^\s*[-—–]\s*", "", text)
    text = re.sub(r"\){2,}", ")", text)
    text = re.sub(r"\({2,}", "(", text)
    text = re.sub(r"([!.,])\1{2,}", r"\1", text)
    return " ".join(text.split())


def usable(text, low=2, high=260):
    if not low <= len(text) <= high or JUNK.search(text) or PROFANE.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(bool(CYR.match(c)) for c in letters) / len(letters) > .9 and not LATIN.search(text)


def index_reply(reply, context):
    """Stricter rules for a reply the bot may actually say."""
    if not (6 <= len(reply) <= 220) or "?" in reply or SENSITIVE.search(reply) or SENSITIVE.search(context):
        return False
    if not persona_safe(reply):
        return False
    words = re.findall(r"[а-яё]+", reply.casefold())
    if len(words) < 2:
        return False
    # Starts mid-sentence or ends with a dangling conjunction.
    if words[-1] in {"и", "а", "но", "что", "как", "в", "на", "с", "к", "по", "не", "или"}:
        return False
    return True


def split_of(context):
    return "valid" if int(hashlib.md5(context.casefold().encode()).hexdigest(), 16) % 100 == 0 else "train"


def telegram(raw):
    for path in sorted(Path(raw).glob("tg_*.parquet")):
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=50000, columns=["question", "answer", "relevance"]):
            d = batch.to_pydict()
            for q, a, rel in zip(d["question"], d["answer"], d["relevance"]):
                if rel == 1:
                    yield None, clean(q), clean(a), "telegram"


def persona(raw):
    table = pq.ParquetFile(Path(raw) / "persona.parquet")
    for batch in table.iter_batches(batch_size=20000):
        d = batch.to_pydict()
        for name, inp, out in zip(d["name"], d["input"], d["output"]):
            if name not in {"dialog_personal_context", "chitchat"}:
                continue
            turns = [clean(t) for t in re.split(r"\n+", inp or "") if t.strip()]
            turns = [re.sub(r"^(?:Собеседник|Пользователь|Ты|Бот|Человек|Q|A|User|Bot)\s*\d*\s*:\s*", "", t) for t in turns]
            turns = [t for t in turns if t and not re.match(r"^(?:Твоя|Ты|Твой|Факты|Персона|Описание|Продолжи|Диалог)\b", t)]
            if not turns:
                continue
            prev = turns[-2] if len(turns) > 1 else None
            yield prev, turns[-1], clean(out), "persona"


def threads(raw, limit):
    n = 0
    for path in sorted(Path(raw).glob("d2_*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=20000):
            for sample in batch.to_pydict()["sample"]:
                turns = [clean(t) for t in sample or [] if t and t.strip()]
                for i in range(1, len(turns)):
                    yield (turns[i - 2] if i > 1 else None), turns[i - 1], turns[i], "threads"
                    n += 1
                if n >= limit:
                    return


def rldd(raw, limit):
    path = Path(raw) / "rldd.jsonl.gz"
    if not path.exists():
        return
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            try:
                conv = json.loads(line)["conversation"]
            except (ValueError, KeyError):
                continue
            turns = [clean(t) for t in conv.split("\n") if t.strip()]
            for i in range(1, len(turns)):
                yield (turns[i - 2] if i > 1 else None), turns[i - 1], turns[i], "rldd"
                n += 1
            if n >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rldd", type=int, default=1500000)
    ap.add_argument("--threads", type=int, default=4000000)
    ap.add_argument("--max-reply-repeats", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen_pairs, reply_count = set(), Counter()
    stats = Counter()
    files = {k: open(out / f"{k}.jsonl", "w", encoding="utf-8") for k in ("train", "valid", "index")}
    for source in (telegram(args.raw), threads(args.raw, args.threads), rldd(args.raw, args.rldd)):
        for prev, context, reply, name in source:
            stats[name + ":seen"] += 1
            if not usable(context, 2, 300) or not usable(reply):
                continue
            if prev and not usable(prev, 2, 300):
                prev = None
            key = hashlib.md5((context.casefold() + "\t" + reply.casefold()).encode()).digest()[:8]
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            r_key = reply.casefold().strip(" .!)")
            reply_count[r_key] += 1
            if reply_count[r_key] > args.max_reply_repeats:
                continue
            row = dict(c=context, r=reply, s=name)
            if prev:
                row["p"] = prev
            part = split_of(context)
            files[part].write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[name + ":" + part] += 1
            question_like = name != "threads" or ("?" in context and len(context) <= 200)
            if part == "train" and name != "rldd" and question_like and index_reply(reply, context) and reply_count[r_key] == 1:
                files["index"].write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[name + ":index"] += 1
    for f in files.values():
        f.close()
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
