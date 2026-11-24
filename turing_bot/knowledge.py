"""Local evidence from article text. No audience questions are indexed."""
import math
from pathlib import Path
import re
import sqlite3
import unicodedata

TOKEN = re.compile(r"[а-яё]+|\d+")
STOP = set("а в во по у о об на из от с со бы и к за без при под над для это что кто где когда почему как чем чего какой какая какие какое каков является такое такая такой такие ли или укажи расскажи объясни простыми словами кратко мне пожалуйста известно иногда обычно после перед очень изза был была были было составляет назови сколько первый впервые появился появилась появляется образуется называется".split())
ENDING = re.compile(r"(иями|ами|ями|ого|его|ому|ему|ыми|ими|иях|ах|ях|ией|ией|ую|юю|ая|яя|ое|ее|ые|ие|ый|ий|ой|ов|ев|ам|ям|ом|ем|ью|ей|ию|ия|ии|а|и|ы|у|ю|е|о|ь)$")
CAUSE = re.compile(r"потому что|обуслов|причин|из-за|благодаря|вследствие|в результате|за сч[её]т|образуется при|возникает при|приводит к|так как", re.I)
FACT = re.compile(r"^(?:(?:расскажи|объясни|скажи)[ ,]+)?(?:почему|отчего|что такое|что значит|кто так(?:ой|ая|ие)|кто (?:создал|написал|изобр[её]л|открыл)|где находится|когда (?:был|появил|созда|основа)|как (?:возника|образу|устроен|работает))", re.I)
PERSONAL = re.compile(r"\b(?:ты|твой|тебя|тебе|советуешь|думаешь|если бы|представь|выбрал|выбрала)\b",re.I)


def terms(text):
    result = []
    for w in TOKEN.findall(unicodedata.normalize("NFC", text.casefold().replace("\u0301","")).replace("ё","е")):
        if w in STOP or len(w) < 3:
            continue
        stem = ENDING.sub("", w)
        result.append(stem if len(stem)>=3 else w)
    return result


class Knowledge:
    def __init__(self, db):
        self.db = db

    @classmethod
    def load(cls, directory=None):
        path = Path(directory) if directory else Path(__file__).with_name("assets")
        url = (path/"facts.db").resolve().as_uri() + "?mode=ro&immutable=1"
        return cls(sqlite3.connect(url, uri=True))

    def propose(self, question):
        if not FACT.search(question) or PERSONAL.search(question):
            return None
        keys = tuple(dict.fromkeys(terms(question)))[:9]
        if not keys:
            return None
        cause = re.search("почему|отчего|возника|образу",question,re.I)
        definition = re.search("что такое|что значит|кто так",question,re.I)
        # Ranking is learned from document frequencies at indexing time (BM25).
        query = " OR ".join('"'+key+'"' for key in keys)
        rows = self.db.execute("SELECT body,lead,source FROM passages WHERE passages MATCH ? "
                               "ORDER BY bm25(passages,1.,2.5,0.,0.,0.) LIMIT 24", (query,))
        options = []
        for body, lead, source in rows:
            body = " ".join(re.sub(r"([.!?])(?=[А-ЯЁ])", r"\1 ", body).split())
            b, title = set(terms(body)), set(terms(lead))
            overlap = set(keys) & b
            if not overlap or (len(keys) >= 2 and len(overlap | (set(keys)&title)) < max(2,math.ceil(len(keys)*.65))):
                continue
            if len(keys)==1 and keys[0] not in title:
                continue
            if cause and not CAUSE.search(body):
                continue
            if definition and not (set(keys)<=title and "—" in body and set(keys) & set(terms(body.split("—",1)[0]))):
                continue
            if re.search("кто так",question,re.I) and not re.search(
                    r"родил|\d{4}|писатель|уч[её]ный|художни|композитор|акт[её]р|поэт|спортсмен|физик|математик|философ|режисс[её]р",body,re.I):
                continue
            if not cause and not definition and (len(keys)<2 or not set(keys)<=b):
                continue
            evidence = 2*len(overlap)/len(keys) + len(title & set(keys))/len(keys)
            if len(body)>300 or body[:1].islower() or re.match(r"^(?:Он|Она|Оно|Они|Там|Этот|Этот же|В этом|Также|Поэтому|Из-за чего)\b",body):
                continue
            options.append((evidence-.001*len(body), body, source))
        return max(options,key=lambda x:x[0])[1:] if options else None
