import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.request import urlopen

from turing_bot.knowledge import terms

URL = "https://raw.githubusercontent.com/vladislavneon/RuBQ/fade3eae6a4807c35b19cfce20dfa805fc8c0cdf/RuBQ_2.0/RuBQ_2.0_paragraphs.json"
SHA = "f91c2fd1d688b1d9f0f89f33c61d678c834e4e52ed9e1324bb76477374c5a6ab"
FILTER = re.compile(r"стрельб|расстрел|взрыв|изготовлени[ея] оружия|взрывчат|наркоти|изнасил|суицид|дозировк|дозу|дозы|онколог|эроти|порно|лечени[ея]|передозир|подрыв|уязвимост|отравляющ", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.download and not a.input.exists():
        a.input.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(URL, timeout=120) as src:
            a.input.write_bytes(src.read())
    raw = a.input.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA:
        p.error("Исходный файл RuBQ не соответствует зафиксированному хешу")
    articles = json.loads(raw)
    titles = {}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".build")
    db = sqlite3.connect(tmp)
    try:
        db.execute("DROP TABLE IF EXISTS passages")
        db.execute("CREATE VIRTUAL TABLE passages USING fts5(lex, topic, body UNINDEXED, lead UNINDEXED, source UNINDEXED)")
        for item in articles:
            page, text = item["ru_wiki_pageid"], item["text"].replace("\u0301","")
            if page not in titles:
                lhs = text.partition(" — ")[0].split(" (")[0].strip()
                titles[page] = lhs if len(lhs)<=65 and len(lhs.split())<=6 else ""
            title = titles[page]
            for sentence in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])", text):
                if not 24 <= len(sentence)<=285 or sentence[-1] not in ".!" or FILTER.search(sentence):
                    continue
                if any(c in sentence for c in "[]{}") or sentence.count("(")!=sentence.count(")"):
                    continue
                db.execute("INSERT INTO passages VALUES(?,?,?,?,?)",
                           (" ".join(terms(sentence)), " ".join(terms(title)), sentence, title, str(page)))
        db.execute("INSERT INTO passages(passages) VALUES('optimize')")
        print("Предложений:",db.execute("SELECT count(*) FROM passages").fetchone()[0],"Страниц в исходнике:",len(titles))
        db.commit()
    finally:
        db.close()
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
