"""Fit adjective–noun agreement and local topic counts on Russian sentences."""
from collections import Counter, defaultdict
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
from turing_bot.neural import tokenize
from turing_bot.language import content

ADJ=re.compile(r"[а-яё-]{3,}(?:ого|его|ая|яя|ый|ий|ой|ую|юю|ое|ее|ые|ие|ым|им|ой|ей|ых|их)$")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sentences",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    counts=defaultdict(Counter)
    surroundings=defaultdict(Counter)
    with gzip.open(a.sentences,"rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            line=tokenize(row["text"])
            for adj,noun in zip(line,line[1:]):
                if not ADJ.fullmatch(adj) or not noun.isalpha() or len(noun)<3:
                    continue
                counts[noun][adj]+=1
                surroundings[noun,adj].update(w[:5] for w in content(row["text"]) if w not in (adj,noun))
    asset={n:{w:[k,surroundings[n,w].most_common(24)] for w,k in cs.most_common(40) if k>=2}
           for n,cs in counts.items() if len(cs)>=3 and sum(cs.values())>=7}
    a.out.write_bytes(gzip.compress(json.dumps(asset,ensure_ascii=False,separators=(",",":")).encode(),mtime=0))


if __name__=="__main__":
    main()
