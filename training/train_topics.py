"""Estimate lexical co-occurrence on attributed Russian sentences."""
from collections import Counter,defaultdict
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from turing_bot.language import content


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sentences",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    forms=defaultdict(Counter)
    def lines():
        with gzip.open(args.sentences,"rt",encoding="utf-8",newline="") as stream:
            for row in csv.DictReader(stream,delimiter="\t"):
                yield content(row["text"])
    for words in lines():
        for w in words:
            forms[w[:5]][w]+=1
    ordered=sorted(forms,key=lambda s:(-sum(forms[s].values()),s))[:5400]
    ids={s:i for i,s in enumerate(ordered)}
    matrix=np.zeros((len(ids),len(ids)),dtype=np.int32)
    n=0
    for words in lines():
        index=np.asarray(sorted({ids[w[:5]] for w in words if w[:5] in ids}),dtype=np.int32)
        matrix[np.ix_(index,index)]+=1
        n+=1
    freq=matrix.diagonal().copy().astype(float)
    result={}
    for i,stem in enumerate(ordered):
        counts=matrix[i].astype(float)
        strength=np.maximum(0.,np.log((counts+.01)*n/(freq[i]*freq))) * counts/(counts+8.)
        strength[(counts<5)|(freq>n*.035)]=0
        strength[i]=0
        related=[]
        for j in np.argsort(-strength,kind="stable")[:16]:
            if strength[j]<=0:
                break
            word=forms[ordered[j]].most_common(1)[0][0]
            related.append([word,round(float(strength[j]),3)])
        if related:
            result[stem]=related
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False,separators=(",",":")).encode(),mtime=0))
    print(n,"sentences,",len(result),"stems")


if __name__=="__main__":
    main()
