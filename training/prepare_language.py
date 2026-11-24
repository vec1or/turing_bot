"""Sample attributed monolingual sentences for training a small model from scratch."""
import argparse
import bz2
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

from training.prepare_data import ASSISTANT_META, REJECT, normalize

BAD = re.compile(r"\b(?:том(?:а|у|е|ом)?|мэри|мэриэн|джон(?:а|у)?|бостон\w*|тат[оа][эе]б\w*|порно\w*|убий\w*|застрел\w*|суицид\w*|сдох\w*|еба\w*|\w*насил\w*|\w*наркот\w*|пистолет\w*|\w*трах\w*|шлюх\w*)\b", re.I)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--valid",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest()!="338d00cb7badb9f9e6ee64a523db956fdac1cab8da4cacd62cba381ac06e3662":
        p.error("Нужна версия дампа Tatoeba из DATA_SOURCES.md")
    valid=[json.loads(s) for s in args.valid.read_text(encoding="utf-8").splitlines()]
    skip={str(x.get("sentence_id","")) for x in valid}
    heldout={normalize(x["response"]).casefold() for x in valid}
    seen=set()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with bz2.open(args.archive,"rt",encoding="utf-8") as inp, gzip.open(args.out,"wt",encoding="utf-8",newline="") as out:
        writer=csv.writer(out,delimiter="\t",lineterminator="\n")
        writer.writerow(["id","author","text"])
        for row in csv.reader(inp,delimiter="\t",quoting=csv.QUOTE_NONE):
            if len(row)<4 or row[0] in skip or not row[3] or row[3]=="\\N":
                continue
            sample=normalize(row[2])
            if not 12<=len(sample)<=175 or REJECT.search(sample) or BAD.search(sample) or ASSISTANT_META.search(sample):
                continue
            if not re.fullmatch(r"[А-ЯЁа-яё ,.!?—–\-«»„“”]+",sample) or sample.count("!")+sample.count("?")>2:
                continue
            key=sample.casefold()
            if key in seen or key in heldout:
                continue
            seen.add(key)
            if hashlib.sha256(row[0].encode()).digest()[0] < 135:
                writer.writerow([row[0],row[3],sample])
    print("Отобрано подходящих уникальных текстов до подвыборки:",len(seen))


if __name__=="__main__":
    main()
