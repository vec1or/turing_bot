"""Learn compact context-to-word counts from ordinary public sentences."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re


def train(source, target):
    counts = defaultdict(Counter)
    sentences = 0
    with gzip.open(source,"rt",encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            sentences += 1
            tokens = re.findall("[а-яё]+",row["text"].casefold())
            for i in range(1,len(tokens)):
                word = tokens[i]
                if len(word)<2:
                    continue
                counts[tokens[i-1]][word] += 1
                if i > 1:
                    counts[" ".join(tokens[i-2:i])][word] += 1
    entries = {ctx: counter.most_common(8) for ctx,counter in sorted(counts.items())
               if sum(counter.values())>=4 and len(counter)>=2}
    payload = {"sentences":sentences,"entries":entries}
    target.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode(),mtime=0))
    print(f"{sentences} sentences, {len(entries)} contexts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    train(args.sentences,args.out)
