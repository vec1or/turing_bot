"""Learn action selection and action-specific conditional word transitions."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from turing_bot.advice import ENDING, FLAME, HAZARD, PERSONAL, WORD, cues, normal


PREFIX = re.compile(r"^(?:если|когда|чтобы|раз уж|прежде чем|перед тем как) (.{3,110}?),\s*(?:то |лучше )?(.+)$",re.I)
SUFFIX = re.compile(r"^(.{3,90}?),\s*(?:если|когда|чтобы|пока|потому что|прежде чем|перед тем как) (.{3,110})$",re.I)
PREP = re.compile(r"^(.{3,75}?)\s+((?:перед|на случай|во время|во избежание|при наличии|в случае)\s+.{3,85})$",re.I)
FRONT = re.compile(r"^(?:в|на|перед|при|во время|в случае|до|после|во избежание)\s",re.I)
PAIR = re.compile(r"^([^.!?]{5,80})[.!]\s+([^.!?]{5,90})[.!]$")
POLITE = re.compile(r"^(?:пожалуйста,?\s+)?(?:не\s+)?([а-яё-]+)",re.I)
FINITE = re.compile(r"(?:й|ь|и)(?:ся|сь)?$")


def vocabulary(path):
    starts = Counter()
    with gzip.open(path,"rt",encoding="utf8") as f:
        next(f)
        for row in f:
            s = normal(row.rstrip().split("\t")[-1])
            match = POLITE.match(s)
            if match:
                starts[match[1]] += 1
    # Singular/plural imperative alternation is productive. Requiring both
    # to occur sentence-initially screens out most noun and indicative forms.
    verbs = {}
    for w in starts:
        if len(w)<3 or not FINITE.search(w):
            continue
        plural = re.sub(r"(?:сь|ся)$","",w)+"те"+("сь" if w.endswith(("ся","сь")) else "")
        if plural in starts:
            verbs[w],verbs[plural] = w,w
    return verbs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args()
    verbs=vocabulary(args.data)
    action=defaultdict(Counter)
    edges=defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    df=Counter()
    observed=set()
    mode=Counter()
    used=[]
    with gzip.open(args.data,"rt",encoding="utf8") as f:
        next(f)
        for line in f:
            fields=line.rstrip().split("\t")
            if len(fields)!=3:
                continue
            uid,author,s=fields
            if (len(s)>200 or not s.endswith((".","!")) or re.search(r"[?;:\"«»—]",s[:-1])
                    or HAZARD.search(normal(s)) or FLAME.search(normal(s))):
                continue
            plain=s[:-1].strip()
            pair=PAIR.fullmatch(s)
            m=PREFIX.match(plain)
            if pair:
                a,b=pair.groups()
                second=POLITE.match(normal(b))
                if second and second[1] in verbs:
                    condition,instruction=a,b
                elif FRONT.search(b):
                    instruction,condition=a,b
                else:
                    continue
                kind="two_clauses_one_original"
            elif re.search(r"[.!]",plain):
                continue
            elif m:
                condition,instruction=m.groups()
                kind="initial_clause"
            else:
                m=SUFFIX.match(plain) or PREP.match(plain)
                if m:
                    instruction,condition=m.groups()
                    kind="final_clause_or_circumstance"
                elif FRONT.match(plain):
                    ws=plain.split()
                    split=next((i for i in range(2,min(len(ws)-1,9)) if normal(ws[i]) in verbs),None)
                    if split is None:
                        continue
                    if normal(ws[split-1])=="не":
                        split-=1
                    condition,instruction=" ".join(ws[:split])," ".join(ws[split:])
                    kind="initial_circumstance"
                else:
                    continue
            # Keep concrete self-contained clauses. A dependent pronoun or
            # name from a particular conversation cannot be transplanted.
            if re.search(r"(?:^|\s)[А-ЯЁ][а-яё]+",condition.lstrip().lower()[:1]+condition.lstrip()[1:]) or re.search(r"[А-ЯЁ][а-яё]+",instruction[1:]):
                continue
            instruction=normal(instruction).strip()
            condition=normal(condition)
            if PERSONAL.search(instruction) or re.search(r"[,;:]",instruction) or re.search(r"\b(?:не|нет|без)\b",condition):
                continue
            tokens=WORD.findall(instruction)
            negated=tokens[:1]==["не"]
            offset=int(negated)
            if len(tokens)<offset+2 or len(tokens)>10:
                continue
            if tokens[offset] in verbs:
                verb=("не " if negated else "")+verbs[tokens[offset]]
                complement=tokens[offset+1:]
            elif not negated and len(tokens)>2 and tokens[0] in {"стоит","следует","нужно","лучше"} and re.search(r"(?:ть|ти|чь)(?:ся|сь)?$",tokens[1]):
                verb=tokens[0]+" "+tokens[1]
                complement=tokens[2:]
            else:
                continue
            attrs=cues(condition)
            fact=(tuple(sorted(attrs)),verb,tuple(complement))
            if not attrs or fact in observed:
                continue
            observed.add(fact)
            mode[kind]+=1
            used.append((uid,author))
            df.update(attrs)
            for k in sorted(attrs):
                action[k][verb]+=1
            last="<s>"
            for pos, word in enumerate([*complement,ENDING]):
                edges[verb][str(pos)+" "+last][word].update(attrs)
                last=word
    strong={k for k,heads in action.items() if max(heads.values())>=2}
    frequency=Counter()
    nearby=defaultdict(Counter)
    sentences=0
    with gzip.open(args.data,"rt",encoding="utf8") as f:
        next(f)
        for line in f:
            sentence=line.rstrip().split("\t")[-1]
            attrs=cues(sentence)
            if len(attrs)>25:
                continue
            sentences+=1
            frequency.update(attrs)
            for k in attrs&strong:
                nearby[k].update(attrs-{k})
    neighbors={}
    for term,context in nearby.items():
        votes={w:round(math.log(n*sentences/(frequency[w]*frequency[term]))*n/(n+4),3) for w,n in context.items() if n>=4}
        neighbors[term]={w:v for w,v in votes.items() if v>=1.35}
    asset={"format":"arena-conditional-action-1","action":action,"edges":edges,"df":df,"count":len(observed),"neighbors":neighbors}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"advice.json.gz"
    with path.open("wb") as buf,gzip.GzipFile(filename="",fileobj=buf,mode="wb",mtime=0) as out:
        out.write(json.dumps(asset,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())
    report={"input_sha256":hashlib.sha256(args.data.read_bytes()).hexdigest(),"used":len(used),"clauses":dict(mode),
            "action_heads":len(edges),"condition_cues":len(df),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "credits":[{"id":i,"author":a} for i,a in used]}
    args.report.parent.mkdir(exist_ok=True,parents=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(len(used),"examples",len(edges),"actions",len(df),"cues")


if __name__=="__main__":
    main()
