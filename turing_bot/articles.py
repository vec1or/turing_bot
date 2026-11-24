"""Select a documented definition or explanation from a local topical article."""
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import re

from .knowledge import terms

CAUSE = re.compile(r"потому что|благодаря|из-за|причин|вследствие|обуслов|в результате|объясняется|вызыва|приводит|возника|образуе|происходит|за сч[её]т", re.I)
SECTION = re.compile(r"природа|механизм|причин|процесс|физик|образовани|формировани|принцип",re.I)
BAD_SECTION = re.compile(r"истори|культур|миф|награды|изображени|упоминани|искусств|архив|происхождение названия",re.I)
ANAPHORA = re.compile(r"^(?:также|поэтому|тогда|в тоже|однако|в то же|таким образом|в этом|на это|эти |эта |этот |он |она |оно |они |её |его |их )",re.I)
PROMPT = re.compile(r"\b(?:почему|отчего|как (?:получается|образу|устроен|работа|возника)|что (?:такое|значит|означает)|объясни|расскажи (?:о|про)\b|зачем|какова причин)",re.I)
FANTASY = re.compile(r"\b(?:если бы|будь ты|представь|твой|тебя|за ширмой)\b",re.I)
FILLER = set("слов термин бабушк прост объясн расскаж появля поем поним точк зрени говор счит бывает явля себя разниц причин".split())
ADVICE_FILLER={"челов","совет","посов","вперв","приех","лучше","стоит","поезд","делат","может","тепер","котор","своег","своем"}
SEASON={"зим","лет","осен","весн","декаб","январ","февра","июл","июн","авгус","мороз","жар"}


def roots(text):
    return {w[:5] for w in terms(text) if w[:5] not in FILLER}


def sentences(paragraph):
    cleaned = re.sub(r"\s+", " ",paragraph.strip())
    cleaned = re.sub(r"\s*\((?:[^()]|\([^()]*\)){0,220}\)", "", cleaned)
    for text in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁ«])", cleaned):
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        if 18<=len(text)<=275 and text[-1] in ".!" and not any(c in text for c in "[]=<>"):
            yield text


class Articles:
    def __init__(self, source):
        self.pages=defaultdict(list)
        self.metadata={}
        self.following={}
        for line in source:
            row=json.loads(line)
            page=row["pageid"]
            if page not in self.metadata:
                self.metadata[page]=(row["title"],roots(row["title"]),row["url"])
            paragraph=list(sentences(row["text"]))
            for i,sentence in enumerate(paragraph):
                if ANAPHORA.search(sentence) or len(sentence)<24:
                    continue
                self.pages[page].append((sentence, roots(sentence), row["section"], i))
                if i+1<len(paragraph):
                    nxt=paragraph[i+1]
                    if len(sentence)+len(nxt)<300 and re.match(r"^(?:Эт(?:и|а|о|от)|При этом|В результате|Таким образом|По этой причине|Они)\b",nxt):
                        self.following[(page,sentence)]=nxt

    @classmethod
    def load(cls,directory=None):
        path=(Path(directory) if directory else Path(__file__).with_name("assets"))/"articles.jsonl.gz"
        with gzip.open(path,"rt",encoding="utf-8") as source:
            return cls(source)

    def propose(self, question):
        if not PROMPT.search(question) or FANTASY.search(question):
            return None
        query=roots(question)
        if not query:
            return None
        causal=bool(re.search(r"почему|отчего|как (?:получается|образу|возника)|зачем|причин",question,re.I))
        define=bool(re.search(r"что (?:такое|значит|означает)",question,re.I))
        options=[]
        for page,(title,theme,url) in self.metadata.items():
            shared=theme&query
            if not shared or len(shared)*2<len(theme):
                continue
            # An exact article title is much stronger evidence than an isolated
            # match somewhere in the body of an unrelated page.
            topic=2.2*len(shared)/math.sqrt(len(theme))
            for position,(s,words,section,index) in enumerate(self.pages[page]):
                if BAD_SECTION.search(section):
                    continue
                overlap=words&query
                if not overlap:
                    continue
                lhs=s.partition("—")[0]
                if define and not ("—" in s and shared & roots(lhs) and len(lhs.split())<=7
                                   and shared & roots(" ".join(lhs.split()[:3]))):
                    continue
                is_cause=bool(CAUSE.search(s))
                mechanism=bool(SECTION.search(section))
                if causal and not (is_cause or mechanism):
                    continue
                if len(query)>3 and not (query-theme)&words:
                    continue
                evidence=topic+1.1*len(overlap)+.6*len(overlap-theme)
                evidence+= (2.4*is_cause+.65*mechanism if causal else 1.1*int(not section))
                evidence+=.30*(position<4)+.13*(index==0)-.0035*len(s)
                if s.startswith(("В ", "С ", "На ")) and re.search(r"\d{3,}",s):
                    evidence-=1.3
                options.append((evidence,s,url,page))
        if not options:
            return None
        best=max(options,key=lambda x:x[0])
        if best[0]<3.1:
            return None
        _,s,url,page=best
        if causal and (page,s) in self.following:
            s+=" "+self.following[page,s]
        return s,url

    def support(self, question):
        """Find descriptive conditions when a viewer actually asks for advice."""
        query=roots(question)-ADVICE_FILLER
        seasonal=bool(query&SEASON)
        matches=[]
        for page,(_,theme,url) in self.metadata.items():
            common=theme&query
            if not common:
                continue
            for position,(sentence,body,section,i) in enumerate(self.pages[page]):
                extra=(query-theme)&body
                if not extra or BAD_SECTION.search(section):
                    continue
                score=1.8*len(extra)+len(common)+.4*len(query&roots(section))-.002*len(sentence)
                if seasonal:
                    score+=2*int("клима" in theme)+.9*int(bool(body&SEASON))
                score+=.45*(position<2)-.6*bool(re.search(r"\b(?:1[0-9]{3}|20[012]\d)\b",sentence))
                matches.append((score,sentence,url))
        return max(matches,key=lambda x:x[0])[1:] if matches else None
