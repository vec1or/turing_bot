"""Compact causal Transformer and a learned subword vocabulary; inference uses NumPy."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import zlib

import numpy as np
from .neural import tokenize, detokenize

PAD, DOC, QUESTION, REPLY, END, UNKNOWN = range(6)
RESERVED = ["<pad>", "<doc>", "<question>", "<reply>", "<end>", "<unknown>"]
MARK = "▁"
FUNCTION_WORDS = set("этот этого этой такая такой какие какой каких просто потому чтобы почему меня тебе тебя есть очень когда если только ведь даже тоже тебя себе свои было были были есть пока этого после почти который которая которое значит сказать скажи слова теперь вообще просто будто будет будут такое этого этого совсем можно нужно думаю каждый первого впервые никогда всегда иногда однажды пусть может какой-то чтобы люблю хочешь вопрос человек людям самый самая самое способ своём своем перед могла было могут делать такое почему сколько хотели вообще вопросом между другие очень каждый большой никто всё-таки можешь среди было ничего себя какой-нибудь никакой обычно всего должна должно должны хотел хотела потому".split())


def content(text):
    return list(dict.fromkeys(t for t in tokenize(text) if len(t)>=4 and t.isalpha() and t not in FUNCTION_WORDS))


class Pieces:
    def __init__(self, tokens):
        self.tokens = tokens
        self.ids = {s:i for i,s in enumerate(tokens)}
        self.longest = max(map(len, tokens))

    @classmethod
    def learn(cls, lines, size=4096):
        words = Counter(w for line in lines for w in tokenize(line))
        counts, alphabet = Counter(), set()
        for word,n in words.items():
            w = MARK+word
            alphabet.update(w)
            for i in range(len(w)):
                for k in range(2,min(6,len(w)-i)+1):
                    counts[w[i:i+k]] += n
        vocab = list(RESERVED) + sorted(alphabet)
        vocab.extend(MARK+w for w,_ in words.most_common(size//2))
        known = set(vocab)
        ranked = sorted(counts, key=lambda s: (-counts[s]*min(5,len(s))**.8, s))
        for s in ranked:
            if s not in known:
                vocab.append(s)
                if len(vocab)>=size:
                    break
        return cls(vocab)

    @lru_cache(maxsize=40000)
    def word(self, word):
        text = MARK + word
        result = []
        while text:
            k = min(len(text), self.longest)
            while k>1 and text[:k] not in self.ids:
                k-=1
            result.append(self.ids.get(text[:k],UNKNOWN))
            text = text[k:]
        return tuple(result)

    def encode(self, text):
        return [i for word in tokenize(text) for i in self.word(word)]

    def decode(self, ids):
        raw = "".join(self.tokens[i] for i in ids if i>=len(RESERVED)).replace(MARK," ")
        return detokenize(raw.split())

    def prompt(self, context, plan=()):
        parts=[]
        lines=context.splitlines()[-3:]
        for i,line in enumerate(lines):
            role=QUESTION if (len(lines)-i)%2 else REPLY
            parts += [role] + self.encode(line)[:65]
        hints = [DOC,*self.encode(" ".join(plan))[:12]] if plan else []
        return (parts[-65:] or [QUESTION])+hints+[REPLY]


def _norm(x, gain, bias):
    centered=x-x.mean(axis=-1,keepdims=True)
    return centered/np.sqrt((centered*centered).mean(axis=-1,keepdims=True)+1e-5)*gain+bias


def _linear(x, w):
    return np.einsum("...i,ji->...j",x,w,optimize=False)


class Language:
    @classmethod
    def load(cls, directory=None):
        root=Path(directory) if directory else Path(__file__).with_name("assets")
        config=json.loads((root/"language.json").read_text(encoding="utf-8"))
        with np.load(root/"language.npz",allow_pickle=False) as cache:
            weights={k:np.asarray(cache[k],dtype=np.float32) for k in cache.files}
        return cls(config,weights)

    def __init__(self,config,weights):
        self.pieces=Pieces(config["vocab"])
        self.config,self.w=config,weights

    def step(self, token, state):
        w=self.w
        pos = state[0][0].shape[0] if state else 0
        x=w["embed.weight"][token] + w["pos.weight"][pos]
        heads=self.config["heads"]
        updated=[]
        for layer in range(self.config["layers"]):
            pre=f"blocks.{layer}."
            z=_norm(x,w[pre+"ln1.weight"],w[pre+"ln1.bias"])
            q,k,v=np.split(_linear(z,w[pre+"qkv.weight"]),3)
            q,k,v=(a.reshape(heads,-1) for a in (q,k,v))
            keys=np.concatenate((state[layer][0],k[None])) if state else k[None]
            values=np.concatenate((state[layer][1],v[None])) if state else v[None]
            a=np.einsum("hd,thd->ht",q,keys)/math.sqrt(q.shape[-1])
            a=np.exp(a-a.max(axis=-1,keepdims=True))
            a/=a.sum(axis=-1,keepdims=True)
            h=np.einsum("ht,thd->hd",a,values).reshape(-1)
            x=x+_linear(h,w[pre+"proj.weight"])
            z=_norm(x,w[pre+"ln2.weight"],w[pre+"ln2.bias"])
            ff=_linear(z,w[pre+"up.weight"])+w[pre+"up.bias"]
            ff=.5*ff*(1+np.tanh(.79788456*(ff+.044715*ff**3)))
            x=x+_linear(ff,w[pre+"down.weight"])+w[pre+"down.bias"]
            updated.append((keys,values))
        y=_norm(x,w["norm.weight"],w["norm.bias"])
        return _linear(y,w["embed.weight"]), updated

    def generate(self,context, *, width=3, limit=38, plan=(), guidance=0.):
        state=None
        prompt=self.pieces.prompt(context,plan)
        for token in prompt:
            last,state=self.step(token,state)
        beams=[(0.,(),last,state)]
        final=[]
        objectives=[tuple(self.pieces.encode(w)) for w in plan if w]
        for t in range(min(limit,self.config["length"]-len(prompt))):
            branches=[]
            for score,ids,logits,cache in beams:
                lp=logits.copy()
                if guidance and objectives:
                    # Reward a word already predicted by the language model
                    # when it also realizes a concept selected by the planner.
                    for target in objectives:
                        if not target or any(ids[j:j+len(target)]==target for j in range(len(ids))):
                            continue
                        for n in reversed(range(min(len(ids),len(target)-1)+1)):
                            if n==0 or ids[-n:]==target[:n]:
                                lp[target[n]]+=guidance*(1.2 if n else 1.)
                                break
                lp-=lp.max()
                lp=lp-np.log(np.exp(lp).sum())
                lp[:END]=-np.inf
                lp[UNKNOWN]=-np.inf
                if t<2:
                    lp[END]=-np.inf
                for k in range(len(ids)-2):
                    if ids[k:k+2]==ids[-2:]:
                        lp[ids[k+2]]=-np.inf
                for j in np.argpartition(lp,-width-1)[-width-1:]:
                    j=int(j)
                    if not np.isfinite(lp[j]):
                        continue
                    value=score+float(lp[j])
                    if j==END:
                        text=self.pieces.decode(ids)
                        if text and re.search("[а-яё]",text.casefold()):
                            final.append((value/(t+1)**.8,text))
                    else:
                        branches.append((value,ids+(j,),cache))
            selected=sorted(branches,key=lambda b:(-b[0],b[1]))[:width]
            if not selected:
                break
            beams=[(s,ids,*self.step(ids[-1],cache)) for s,ids,cache in selected]
        if final:
            return max(final)[1]
        return self.pieces.decode(beams[0][1])
