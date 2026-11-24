"""Estimate smoothed word/domain associations from unlabelled Russian text."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
from turing_bot.associations import FRAMES,stems


def build(source, destination):
    anchors=defaultdict(set)
    for domain,frame in FRAMES.items():
        for cue in stems(frame[0]):
            anchors[cue].add(domain)
    frequency=Counter()
    cooccurrence={domain:Counter() for domain in FRAMES}
    domain_size=Counter()
    total=0
    with gzip.open(source,"rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            total+=1
            tokens=set(stems(row["text"]))
            frequency.update(tokens)
            domains=set().union(*(anchors[t] for t in tokens if t in anchors))
            for d in domains:
                domain_size[d]+=1
                cooccurrence[d].update(tokens)
    weights={}
    for d,counts in cooccurrence.items():
        values=[]
        for t,n in counts.items():
            if n<4 or t in anchors and d in anchors[t]:
                continue
            ratio=n*total/(domain_size[d]*frequency[t])
            score=min(2.,math.log(max(1,ratio))*n/(n+12))
            if score>.30:
                values.append((t,round(score,4)))
        weights[d]=dict(sorted(values,key=lambda v:(-v[1],v[0]))[:1600])
    data=dict(sentences=total,domains=dict(domain_size),weights=weights)
    destination.write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode(),mtime=0))
    print(total,len(weights),sum(map(len,weights.values())))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sentences",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    build(args.sentences,args.out)
