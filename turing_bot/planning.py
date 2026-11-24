"""Select lexical hints for generation and complete attested adjective–noun phrases."""
import gzip
import json
import math
from pathlib import Path
import re
import zlib

from .language import content

IMAGINE=re.compile(r"\b(?:если бы|если б|представ\w*|вообраз\w*|придум\w*|по-твоему|для тебя|метафор\w*|сказоч\w*|были бы|был бы|была бы|каким бы|какой бы)\b",re.I)
LIGHT=set("сказал сказала сказали разговаривать первым первым больше всего какой каких громче всех обычно будет делать можешь человека новую новые этом была были взяла взял собой кажется сможешь каким что-нибудь перед свою одну такое".split())
DEICTIC=("како","тако","само","котор","целы","целог","наш","ваш","этог","один","одног","друго","разны","разно","любог","подобн")


class Planner:
    def __init__(self,folder=None):
        root=Path(folder) if folder else Path(__file__).with_name("assets")
        with gzip.open(root/"topical.json.gz","rt",encoding="utf-8") as f:
            self.associations=json.load(f)
        with gzip.open(root/"properties.json.gz","rt",encoding="utf-8") as f:
            self.properties=json.load(f)

    def hints(self, text):
        keywords=[w for w in content(text) if w not in LIGHT and w[:5] in self.associations]
        if not keywords:
            return []
        # When two content words occur together in human sentences their
        # common surrounding words are better cues than an isolated association.
        scores={}
        for w in keywords:
            for related,weight in self.associations[w[:5]][:9]:
                if related not in keywords and related not in LIGHT:
                    scores[related]=scores.get(related,0.)+weight
        best=max(scores,key=scores.get) if scores else None
        return [keywords[0],best] if best else [keywords[0]]

    def adjective(self,question):
        match=re.search(r"\b(какого|каким|какой|какая|какую|какие|какое)\s+([а-яё-]+)",question,re.I)
        imagined=bool(IMAGINE.search(question))
        ellipsis=re.search(r"\b(?:был[аои]?|будет)\s+([а-яё-]+)[^.!?]{0,35}?[,—]\s*(?:то )?(каким|какой|какая|какое)\b",question,re.I) if imagined else None
        if ellipsis:
            noun,wh=ellipsis[1].casefold(),ellipsis[2].casefold()
            keys={k[:5] for k in content(question[:ellipsis.start()]) if k not in LIGHT}
        elif match:
            wh, noun=match[1].casefold(),match[2].casefold()
            keys={k[:5] for k in content(question[match.end():]) if k not in LIGHT}
        else:
            return None
        forms=self.properties.get(noun)
        if not forms:
            return None
        endings={"какого":("ого","его"),"каким":("ым","им"),"какой":("ый","ий","ой"),
                 "какая":("ая","яя"),"какую":("ую","юю"),"какие":("ые","ие"),"какое":("ое","ее")}
        choices=[]
        for word,(n,contexts) in forms.items():
            instrumental=bool(ellipsis and wh=="каким")
            if word.startswith(DEICTIC) or not word.endswith(("ый","ий","ой") if instrumental else endings[wh]):
                continue
            evidence=sum(freq for root,freq in contexts if root in keys)
            if evidence<2 and not imagined:
                continue
            variety=zlib.crc32((question+word).encode())/2**32
            score=2.5*math.log1p(evidence)+.25*math.log1p(n)+.9*variety
            choices.append((score,word))
        if choices:
            word=max(choices)[1]
            if ellipsis:
                if wh=="каким":
                    word=word[:-2]+("им" if word.endswith("ий") else "ым")
                return word.capitalize()+"."
            return word.capitalize()+" "+noun+"."
        return None
