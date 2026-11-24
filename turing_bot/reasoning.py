"""Small, proof-based solvers for facts supplied in one Russian utterance."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from .reader import adj, colour, normal, stem


W = r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*"
ASK = re.compile(r"\b(?:на сколько|во сколько|через сколько|сколько|кто|когда|что (?:же )?остал\w*|что (?:теперь )?(?:лежит|находится|хранится|выше|ниже|дороже|дешевле|легче|тяжелее|длиннее|короче|покажут)|как(?:ого|ой|ая|ое|ие|ую|ом|им|ими|их)\b|где|куда|откуда|у кого|чей|кому|достаточно ли|хватает ли|чего не хватает|можно ли определить|можно ли узнать|есть ли противоречие|могут ли оба|совместимы ли|не противоречат ли)\b",re.I)
UNSURE = re.compile(r"\b(?:возможно|может быть|кажется|примерно|около половины|якобы|наверно|наверное|неизвестно|не помню|то ли|не уверен\w*)\b",re.I)
CONFLICT = re.compile(r"противореч|совместим|могут ли оба|одновременно (?:верн|истин)",re.I)
DIGITS = dict(zip("ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split(),range(20)))
DIGITS.update(одна=1, одно=1, одну=1, две=2, двадцать=20, тридцать=30, сорок=40, пятьдесят=50)
DIGITS.update(одном=1, двух=2, трех=3, четырех=4, пяти=5, шести=6, семи=7, восьми=8, девяти=9, десяти=10, шестидесяти=60, шестьдесят=60)
NUMBER = r"(?:\d{1,4}|" + "|".join(sorted(DIGITS,key=len,reverse=True)).replace("е","[её]") + r")"
CLOCK = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)")
DURATION = re.compile(rf"\b({NUMBER})\s+(минут\w*|час(?:а|ов)?)\b",re.I)
PLUS = r"добав\w*|прибав\w*|принес\w*|принёс\w*|верну\w*|долож\w*|получи\w*|пополн\w*|прибыл\w*|купил\w*|подлож\w*|положи\w*|докину\w*|подарил\w*"
MINUS = r"убра\w*|забра\w*|унес\w*|унёс\w*|вытащ\w*|выну\w*|взя[лт]\w*|съе[лс]\w*|отдал\w*|разда\w*|потрат\w*|потеря\w*|исчез\w*|ушл\w*|выброс\w*|выкину\w*|израсход\w*|истрати\w*|продал\w*"
CHANGE = re.compile(rf"\b(?:{PLUS}|{MINUS})\b",re.I)
FILL = {"и","а","но","оттуда","из","в","во","на","с","со","за","изначально","там","еще","ещё","их","он","она","оно","они","его","ее","её","один","одну","потом","снова","затем","теперь","уже","штук","штуки"}


def _number(word):
    return int(word) if word.isdigit() else DIGITS[normal(word)]


def _capitalize(text):
    return text[:1].upper()+text[1:]


def _words(text):
    return re.findall(W,text)


def _key(word):
    return stem(normal(word))


def _plural(value, forms):
    end=value%100
    i=0 if value%10==1 and end!=11 else 1 if 2<=value%10<=4 and not 12<=end<=14 else 2
    return f"{value} {forms[i]}"


def _elapsed(minutes, hours=False):
    if not hours:
        return _plural(minutes,("минута","минуты","минут"))+"."
    h,m=divmod(minutes,60)
    parts=[_plural(h,("час","часа","часов"))] if h or not m else []
    if m:
        parts.append(_plural(m,("минута","минуты","минут")))
    return " ".join(parts)+"."


def _split(text):
    # Do not split decimals/times such as 13.15 or quoted punctuation.
    first_question=text.find("?")
    if 0<first_question<len(text)-5 and ASK.search(text[:first_question]) and not ASK.search(text[first_question+1:]):
        return text[first_question+1:].strip(),text[:first_question+1]
    ends = list(re.finditer(r"[.!?;]\s+(?=\S)",text))
    for mark in reversed(ends):
        suffix=text[mark.end():].strip()
        if ASK.search(suffix):
            return text[:mark.start()],suffix
    found=ASK.search(text)
    if found and found.start()>0 and re.search(r"[,;:—]\s*$",text[:found.start()]):
        return text[:found.start()].rstrip(" ,;:—"),text[found.start():]
    return "",text


def _clauses(text):
    return [p.strip(" ,;:.?!—") for p in re.split(r"[;.!?]\s+|,\s*(?:а |и |но )?|\s+(?:а затем|а потом|затем|потом)\s+",text,flags=re.I) if p.strip(" ,;:.?!—")]


def _clock_words(text):
    # Re-express explicit time units, retaining all event words. Bare counts
    # such as «в двух банках» are never clocks.
    text=re.sub(r"\bполчаса\b","30 минут",text,flags=re.I)
    text=re.sub(r"\bчетверть часа\b","15 минут",text,flags=re.I)
    pattern=rf"\b(в|к|с|до|на)\s+({NUMBER})\s+(?:часов|часа|час)\b(?:\s+({NUMBER})\s+минут\w*)?"
    def replace(m):
        hour,minute=_number(m[2]),_number(m[3]) if m[3] else 0
        return f"{m[1]} {hour:02d}:{minute:02d}" if hour<24 and minute<60 else m[0]
    return re.sub(pattern,replace,text,flags=re.I)


def _calendar(facts,query):
    weekdays=("понедельник","вторник","среда","четверг","пятница","суббота","воскресенье")
    if not re.search(r"день недели|какой (?:же )?(?:это будет |будет )?день|какой день",query,re.I):
        return None
    current=re.search(r"\b(?:сегодня|сейчас)\s*[—–:-]?\s*("+ "|".join(weekdays)+r")\b",facts,re.I)
    if not current:
        return None
    day=weekdays.index(normal(current[1]))
    span=re.search(rf"\b(через|спустя)\s+({NUMBER})\s+(?:день|дня|дней|сут\w*)\b",query,re.I)
    if span:
        day+=_number(span[2])
    else:
        offset=re.search(r"\b(послезавтра|позавчера|завтра|вчера)\b",query,re.I)
        if not offset:
            return None
        day+={"послезавтра":2,"позавчера":-2,"завтра":1,"вчера":-1}[normal(offset[1])]
    return _capitalize(weekdays[day%7])+"."


def _schedule(facts,query):
    """Use a named start, deadline or departure as the clock anchor."""
    clocks=list(CLOCK.finditer(facts))
    if len(clocks)!=1 or not re.search(r"когда|во сколько|какое время|что покажут часы",query,re.I):
        return None
    raw=normal(facts+" "+query)
    durations=list(DURATION.finditer(raw))
    if not durations or len(durations)>2 or re.search(r"\bне\b|\bили\b",raw):
        return None
    total=sum(_number(m[1])*(60 if m[2].startswith("час") else 1) for m in durations)
    if re.search(r"раньше|до (?:отправ\w*|начал\w*)|(?:выйти|выходить|выезжать|стартовать)\b",normal(query)):
        if not re.search(r"раньше|дорог\w*|пут[ьи]|идти|ехать|добра\w*|пешком|занима\w*|добират\w*|до (?:отправ\w*|начал\w*)",raw):
            return None
        total=-total
    elif not re.search(r"через|спустя|позже|оконч\w*|конч\w*|заверш\w*|прибуд\w*|приед\w*|будем на месте|законч\w*|приход\w*|придет",normal(query)):
        return None
    clock=clocks[0]
    h,m=divmod((int(clock[1])*60+int(clock[2])+total)%1440,60)
    return f"В {h:02d}:{m:02d}."


def _time(facts, query):
    clocks=list(CLOCK.finditer(facts))
    q=normal(query)
    if not clocks and not facts and re.search(r"\b(?:между|с|от)\b",q):
        clocks=list(CLOCK.finditer(query))
        facts=query
    values=[int(c[1])*60+int(c[2]) for c in clocks]
    if len(values)==1 and re.search(r"\b(?:во сколько|когда)\b",q):
        durations=list(DURATION.finditer(facts))
        if (not durations or len(durations)>2 or not re.search(r"дли\w*|продолжа\w*|через|заня\w*|идет|идти|едет|еха\w*",normal(facts))):
            return None
        start = re.search(r"начал\w*|выеха\w*|выш\w*|отправ\w*|старт\w*",normal(facts))
        finish = re.search(r"закончи\w*|кончи\w*|прибы\w*|приде\w*|приед\w*|заверш\w*",q)
        if not start or not finish or " не " in " "+normal(facts)+" ":
            return None
        count=sum(_number(d[1])*(60 if d[2].startswith("час") else 1) for d in durations)
        h,m=divmod((values[0]+count)%1440,60)
        return f"В {h:02d}:{m:02d}."
    if len(values)!=2 or not re.search(r"\b(?:сколько|разниц\w*)\b",q):
        return None
    if not re.search(r"минут|час|времени|опозд|раньше|позже|прошло|между|длит|длил|занял|занима|продолжа",q):
        return None
    if re.search(r"\b(?:не (?:приб|приш|уш|уех|опоз)|отмен\w*|перенес\w*)",normal(facts)):
        return None
    if re.search(r"опозд|раньше|ждать|ожидан",q):
        # Bind a person's arrival to a distinct reference event. An early
        # arrival must not be reported as positive lateness.
        labels=[]
        reference=[]
        for c in clocks:
            start=max(facts.rfind(sep,0,c.start()) for sep in (",",";","!","?"))+1
            segment=normal(facts[start:c.start()]).rsplit(". ",1)[-1]
            segment=re.split(r"\b(?:а|и я|но я)\b",segment)[-1]
            labels.append(bool(re.search(r"прибе\w*|прибеж\w*|приш\w*|прибы\w*|подош\w*|добра\w*|явил\w*|оказал\w*",segment)))
            reference.append(bool(re.search(r"нача\w*|уш[её]л|ушла|отош\w*|отправ\w*|уех\w*|назнач\w*|договор\w*|расписани\w*|откры\w*|закры\w*|встреча|сеанс|старт\w*|долж\w*|планир\w*",segment)))
        if len(set(labels))!=2 or not reference[labels.index(False)]:
            return None
        actual=values[labels.index(True)]
        scheduled=values[labels.index(False)]
        minutes=(actual-scheduled) if "опозд" in q else (scheduled-actual)
        if minutes<0:
            if "следующ" in normal(facts) or "полноч" in normal(facts):
                minutes+=1440
            else:
                return None
    else:
        minutes=values[1]-values[0]
        if minutes<0:
            if "между" in q:
                minutes=abs(minutes)
            elif re.search(r"следующ\w* (?:день|утро)|полноч\w*|на следующее",normal(facts)):
                minutes+=1440
            else:
                return None
    return _elapsed(minutes,"час" in q)


ORDER = {
    "очередь": (("перед","впереди","раньше"),("после","позади","позже","за"),r"перв\w*|впереди|раньше|начал\w*",r"последн\w*|позади|позже|конце"),
    "рост": (("выше",),("ниже",),r"выше|высок\w*|наивыс\w*",r"ниже|низк\w*"),
    "возраст": (("старше",),("младше",),r"старше|старш\w*",r"младше|младш\w*"),
    "вес": (("тяжелее",),("легче",),r"тяжел\w*",r"легче|легк\w*"),
    "цена": (("дороже",),("дешевле",),r"дороже|дорог\w*",r"дешевле|дешев\w*"),
    "скорость": (("быстрее",),("медленнее",),r"быстр\w*",r"медлен\w*"),
    "длина": (("длиннее",),("короче",),r"длин\w*",r"короче|корот\w*"),
    "сторона": (("левее",),("правее",),r"левее|лев\w*",r"правее|прав\w*"),
}
ORDER_VERB = re.compile(r"\b("+ "|".join(w for spec in ORDER.values() for words in spec[:2] for w in words) +r")\b",re.I)
IRREGULAR = {"павлом":"Павел","павла":"Павел","петром":"Пётр","петра":"Пётр","львом":"Лев","льва":"Лев"}


def _forms(name):
    w=normal(name)
    if w.endswith("а"):
        return {w,w[:-1]+"ы",w[:-1]+"и",w[:-1]+"у",w[:-1]+"ой",w[:-1]+"ей",w[:-1]+"е"}
    if w.endswith("я"):
        return {w,w[:-1]+"и",w[:-1]+"ю",w[:-1]+"ей",w[:-1]+"е"}
    if w.endswith(("ь","й")):
        return {w,w[:-1]+"я",w[:-1]+"ю",w[:-1]+"ем"}
    return {w,w+"а",w+"у",w+"ом",w+"е"}


def _nominative(raw, rel, known):
    w=normal(raw)
    matches=[name for name in known if w in _forms(name)]
    if len(matches)==1:
        return matches[0]
    if w in IRREGULAR:
        return IRREGULAR[w]
    if rel in {"перед","за"}:
        for end, new in (("ьей","ья"),("ой","а"),("ей","я"),("ом",""),("ем","ь")):
            if w.endswith(end):
                root=raw[:-len(end)]
                if end=="ей" and root[-1:].lower() in "шжчщц":
                    new="а"
                return root+new
        return None
    if w.endswith("ы"):
        return raw[:-1]+"а"
    if w.endswith("и"):
        return raw[:-1]+("а" if w[-2:-1] in "гкхжчшщ" else "я")
    if w.endswith("а"):
        return raw[:-1]
    if w.endswith("я"):
        return raw[:-1]+("й" if w.endswith(("ая","ея")) else "ь")
    return raw


def _ordering(facts,query):
    raw=[]
    for clause in _clauses(facts):
        clause=re.sub(r"\b(слева|справа)\s+от\b",lambda m:"левее" if normal(m[1])=="слева" else "правее",clause,flags=re.I)
        clause=re.sub(r"\b(?:вслед|следом|сразу)\s+за\b","за",clause,flags=re.I)
        relation=list(ORDER_VERB.finditer(clause))
        if not relation:
            continue
        if len(relation)!=1 or re.search(r"\b(?:не|или|либо|возможно|если)\b",clause,re.I):
            return None
        rel=relation[0]
        left=_words(clause[:rel.start()])
        right=re.match(rf"\s+(?:чем\s+)?({W})\b",clause[rel.end():],re.I)
        if not right:
            return None
        after=clause[rel.end()+right.end():]
        inverted=re.search(rf"(?:\b(?:стоит|стоял[аи]?|идет|шел|шла|располож\w*|наход\w*|сидит|оказал\w*)\s+|[—–]\s*)({W})\s*$",after,re.I)
        if inverted and not any(w[0].isupper() and normal(w) not in {"в","на","у","слева","справа"} for w in left):
            raw.append((inverted[1],normal(rel[0]),right[1]))
            continue
        if not left:
            return None
        names=[w for w in left if w[0].isupper() and normal(w) not in {"в","на","а","по","но","потом","затем"}]
        subject=names[0] if names else left[0]
        if normal(subject) in {"в","у","на","я","он","она","был","если"}:
            return None
        raw.append((subject,normal(rel[0]),right[1]))
    if not raw:
        return None
    q=normal(query)
    dimension=next((spec for spec in ORDER.values() if all(r in spec[0]+spec[1] for _,r,_ in raw)),None)
    if not dimension:
        return None
    first=bool(re.search(r"\b(?:"+dimension[2]+r")\b",q))
    last=bool(re.search(r"\b(?:"+dimension[3]+r")\b",q))
    if first==last and not CONFLICT.search(q):
        return None
    known={p for p,_,_ in raw}
    follows=defaultdict(set)
    labels={}
    for a,rel,b0 in raw:
        b=_nominative(b0,rel,known)
        if not b:
            return None
        ka,kb=normal(a),normal(b)
        labels[ka],labels[kb]=a,b
        if rel in dimension[0]:
            follows[ka].add(kb)
        else:
            follows[kb].add(ka)
    for middle in labels:
        for node in labels:
            if middle in follows[node]:
                follows[node].update(follows[middle])
    cycle=any(k in follows[k] for k in labels)
    if CONFLICT.search(q):
        return "Условия противоречат друг другу: получается строгое сравнение в обе стороны." if cycle else None
    if cycle:
        return None
    all_forms={word for name in labels.values() for word in _forms(name)}
    foreign_name=any(w[:1].isupper() and normal(w) not in all_forms for w in _words(query)[1:])
    if foreign_name:
        return None
    named=[k for k,name in labels.items() if _forms(name) & set(_words(q))]
    if " или " in q and len(named)!=2:
        return None
    if len(named)==1 and not re.search(r"\bиз (?:них|всех|этой|этого)",q):
        if "непосредственно" in q or "сразу" in q:
            return None
        anchor=named[0]
        proven=[k for k in labels if k!=anchor and (anchor in follows[k] if first else k in follows[anchor])]
        return _capitalize(labels[proven[0]])+"." if len(proven)==1 else None
    considered=set(named) if len(named)>=2 else set(labels)
    candidates=[k for k in considered if all(other==k or (other in follows[k] if first else k in follows[other]) for other in considered)]
    if len(candidates)==1:
        return _capitalize(labels[candidates[0]])+"."
    return None


def _item_key(word):
    key=_key(word)
    if len(word)>4 and normal(word).endswith(("ень","онь")):
        key=re.sub(r"[ео]н$","н",key)
    return re.sub(r"([бвгджзклмнпрстфхцчшщ])[ео]к$",r"\1к",key)


def _quantities(facts,query):
    if not re.search(r"сколько",query,re.I) or not CHANGE.search(facts) or re.search(r"\b(?:не|если|половин\w*|кажд\w*)\b",facts,re.I):
        return None
    if CHANGE.search(query) or re.search(r"\b(?:до|сначала|изначально|было|раньше)\b",query,re.I):
        return None
    states={}
    amounts=re.compile(rf"\b({NUMBER})\b",re.I)
    for clause in _clauses(facts):
        for piece in re.split(rf"\s+и\s+(?=(?:еще\s+)?(?:{PLUS}|{MINUS}|{NUMBER}\b))",clause,flags=re.I):
            found=list(amounts.finditer(piece))
            if not found:
                if CHANGE.search(piece):
                    return None
                continue
            if len(found)!=1:
                return None
            match=found[0]
            value=_number(match[1])
            following=_words(piece[match.end():])
            noun=next((w for w in following[:2] if not adj(w) and normal(w) not in FILL and not CHANGE.fullmatch(w)),None)
            noun=_item_key(noun) if noun else None
            event=CHANGE.search(piece)
            if not event:
                if not noun or noun in states:
                    return None
                states[noun]=value
                continue
            if noun is None:
                if len(states)!=1:
                    return None
                noun=next(iter(states))
            if noun not in states:
                return None
            sign=-1 if re.fullmatch(MINUS,event[0],re.I) else 1
            states[noun]+=sign*value
            if states[noun]<0:
                return None
    query_keys={_item_key(w) for w in _words(query)}
    matches=set(states) & query_keys
    if not matches and len(states)==1 and re.search(r"сколько (?:(?:их|всего|теперь) )?(?:их|всего|остал\w*|стало|теперь)[?!.\s]*$",query,re.I):
        matches=set(states)
    if len(matches)==1:
        return str(states[matches.pop()])+"."
    return None


def _colour_key(word):
    return re.sub(r"(?:ыми|ими|ого|его|ому|ему|ым|им|ом|ем|ой|ей|ый|ий|ая|яя|ое|ее|ые|ие|ых|их|ую|юю|а|ы|о)$","",normal(word))


def _closed_set(facts,query):
    clauses=[p.strip(" ,.!?") for p in re.split(r"[.!?;]\s+",facts) if p.strip(" ,.!?")]
    if len(clauses)==1:
        clauses=_clauses(facts)
    if len(clauses)<2:
        return None
    source=next((c for c in clauses if re.search(r"\b(?:только|исключительно|либо|ровно два|ровно две|один из двух|одна из двух)\b",c,re.I)),None)
    if not source:
        return None
    rest=". ".join(c for c in clauses if c!=source)
    q=normal(query)
    if re.search(r"цвет|оттен",q):
        # Keep the subject of the closed set separate from any other item.
        # E.g. a yellow *cube* cannot exclude a yellow token.
        all_words=_words(source)
        choices={_colour_key(w):w for w in all_words if colour(w)}
        indeces=[i for i,w in enumerate(all_words) if colour(w)]
        if len(choices)<2 or not indeces:
            return None
        later=all_words[indeces[-1]+1:]
        kind=_item_key(later[0]) if later else None
        rest_words={_item_key(w) for w in _words(rest)}
        if kind and kind not in rest_words and not re.search(r"^(?:она|он|оно|это|вынут\w*|выбран\w*)\b",normal(rest)):
            return None
        negative_clauses=[c for c in _clauses(rest) if re.search(r"\b(?:не|кроме|исключая)\b",c,re.I)]
        if kind and any(kind not in {_item_key(w) for w in _words(c)} and
                        not re.match(r"^(?:(?:и|а|но|точно) )*(?:он|она|оно|это|выбран\w*|вынут\w*|не)\b",normal(c))
                        for c in negative_clauses):
            return None
        asked=_words(re.sub(r"\b(?:какого|цвета|оттенка|он|она|оно|они|эт[аои]|эта|этот|же|теперь|был[аои]?|будет|оказа\w*)\b","",q))
        if kind and asked and kind not in {_item_key(w) for w in asked}:
            return None
        excludes={_colour_key(m[1]) for m in re.finditer(rf"\b(?:не|кроме|исключая)\s+({W})",rest,re.I) if colour(m[1])}
        remaining=set(choices)-excludes
        if len(remaining)==1 and excludes and excludes <= set(choices):
            key=remaining.pop()
            sample=choices[key]
            surface=sample[:len(key)]
            ending="его" if re.search("ий|ие|их|яя|ее|его|им",normal(sample)[len(key):]) else "ого"
            return _capitalize(surface+ending)+"."
        return None
    if not re.search(r"\b(?:куда|какой|какая|какое|кто|что)\b",q):
        return None
    body=re.split(r"\b(?:только|исключительно|либо)\s+|:\s*",source,flags=re.I)[-1]
    choices=[v.strip(" .—") for v in re.split(r"\s+(?:или|либо)\s+",body,flags=re.I)]
    if len(choices)!=2 or any(not re.fullmatch(rf"{W}(?:\s+{W})?",p) for p in choices):
        return None
    eliminated=set()
    for i,choice in enumerate(choices):
        anchor=re.escape(normal(choice))
        if re.search(rf"(?:\bне\s+{anchor}\b|\b{anchor}\s+(?:нельзя|исключен\w*|закрыт\w*|не\s+(?:подход\w*|работ\w*|доступ\w*|выбир\w*|поед\w*|пойдем)))",normal(rest)):
            eliminated.add(i)
    if len(eliminated)==1:
        return _capitalize(choices[1-eliminated.pop()])+"."
    return None


POSITION = re.compile(r"\s+\b(?:слева|справа|посередине|рядом|сверху|снизу)\b.*",re.I)
NP = rf"{W}(?:\s+{W}){{0,2}}"


def _nominal_phrase(text):
    words=_words(text)
    if not words or len(words)>3:
        return False
    return all(adj(w) for w in words[:-1]) and normal(words[-1]) not in FILL


def _inventory(facts,query):
    if not re.search(r"\bчто\b.*\bостал\w*",query,re.I):
        return None
    clauses=_clauses(facts)
    if not clauses:
        return None
    first=clauses[0]
    if ":" in first:
        listing=first.rsplit(":",1)[-1]
        if not re.search(r"вещ|предмет|лежа|полож|наход|список|только|коробк",first.split(":")[0],re.I):
            return None
    else:
        lead=re.search(r"\b(?:лежали|лежит|были|были только|находились|стояли|остались|стоят)\s+",first,re.I)
        if not lead:
            return None
        listing=first[lead.end():]
    # Commas may split a list before its last coordination.
    atomized=[listing]
    used=1
    while not re.search(r"\s+и\s+",atomized[-1],re.I) and used<len(clauses) and not CHANGE.search(clauses[used]):
        atomized.append(clauses[used])
        used+=1
    names=[]
    for piece in atomized:
        for obj in re.split(r"\s+и\s+",piece,flags=re.I):
            obj=POSITION.sub("",obj).strip(" .,")
            if not _nominal_phrase(obj):
                return None
            names.append(obj)
    if len(names)<2:
        return None
    items={_item_key(_words(obj)[-1]):obj for obj in names}
    if len(items)!=len(names):
        return None
    trailing=re.search(r"\b(?:когда|после того как|если)\s+(.+)$",query,re.I)
    changes=clauses[used:]+([trailing[1]] if trailing else [])
    if not changes or any(re.search(r"\b(?:не|или|либо)\b",c,re.I) for c in changes):
        return None
    for clause in changes:
        event=CHANGE.search(clause)
        if not event:
            return None
        names_after=_words(clause[event.end():])
        matched={_item_key(w) for w in names_after} & set(items)
        if re.fullmatch(MINUS,event[0],re.I):
            if len(matched)!=1:
                return None
            items.pop(matched.pop())
        else:
            obj=" ".join(names_after)
            if not _nominal_phrase(obj):
                return None
            items[_item_key(names_after[-1])]=obj
    if items:
        remaining=[]
        for result in items.values():
            terms=_words(result)
            if ":" in first and re.search(r"полож|взял|назвал",first,re.I):
                terms[-1]=re.sub(r"у$","а",re.sub(r"ю$","я",terms[-1]))
                terms=[re.sub(r"ую$","ая",re.sub(r"юю$","яя",w)) for w in terms]
            remaining.append(" ".join(terms))
        if len(remaining)==1:
            return _capitalize(remaining[0])+"."
        return _capitalize(", ".join(remaining[:-1])+" и "+remaining[-1])+"."
    if len(items)==0:
        return "Из перечисленного ничего не осталось."
    return None


PLACE = r"(?:лежа\w*|лежит|стои\w*|стоя\w*|хран\w*|наход\w*|спрятан\w*|есть|был[аои]?)"
LOCATION = re.compile(rf"^(в|во|на|под|у)\s+({NP})\s+(?:{PLACE}\s*|[—–-]\s*)(.+)$",re.I)
REVERSE = re.compile(rf"^({NP})\s+(?:{PLACE}\s+|[—–-]\s*)(в|во|на|под|у)\s+({NP})$",re.I)
REVERSE_SHORT = re.compile(rf"^({NP})\s+(в|во|на|под|у)\s+({NP})$",re.I)
PURPOSES = (
    (r"театр\w*|кино\w*|концерт\w*|спектакл\w*|поезд",r"билет\w*"),
    (r"дожд\w*|ливн\w*|промок\w*",r"зонт\w*|дождевик\w*"),
    (r"отпер\w*|замок|дверь",r"ключ\w*"),
    (r"маршрут\w*|дорог\w*|ориентир\w*",r"карт[ауы]|компас\w*"),
    (r"запис\w*|напис\w*",r"ручк\w*|карандаш\w*"),
)


@dataclass(frozen=True)
class Binding:
    item: str
    place: str


def _bindings(facts,query):
    entries=[]
    applications=[]
    shared=None
    for line in _clauses(facts):
        if re.search(r"\b(?:не|если|кажется|вместо|возможно|перелож\w*|перемес\w*|перестав\w*|переда\w*|поменя\w*)\b",line,re.I) or CHANGE.search(line):
            return None
        app=re.match(rf"^({NP})\s+(?:нуж[еннаоы]+|служит|подходит|предназначен\w*)\s+(?:для|чтобы)\s+(.+)$",line,re.I)
        if app:
            applications.append((_item_key(_words(app[1])[-1]),{_key(w) for w in _words(app[2])}))
            continue
        match=LOCATION.fullmatch(line)
        if match:
            prep,loc,obj=match.groups()
        else:
            match=REVERSE.fullmatch(line) or REVERSE_SHORT.fullmatch(line)
            if not match:
                nominal=re.fullmatch(rf"(у|в|во|на|под)\s+(.+)",line,re.I)
                if not nominal:
                    continue
                prep=nominal[1]
                parts=_words(nominal[2])
                width=2 if len(parts)>2 and adj(parts[0]) else 1
                loc,obj=" ".join(parts[:width])," ".join(parts[width:])
            else:
                obj,prep,loc=match.groups()
        if not _nominal_phrase(obj) or len(_words(loc))>3:
            continue
        if len(_words(loc))>=2:
            shared=_words(loc)[-1]
        elif adj(loc) and shared:
            loc=loc+" "+shared
        entries.append(Binding(obj,normal(prep)+" "+loc))
    if not entries:
        return None
    q=normal(query)
    query_keys={_item_key(w) for w in _words(q)}
    if re.search(r"\b(?:где|куда|у кого|чей|кому)\b",q):
        hits={e.place for e in entries if _item_key(_words(e.item)[-1]) in query_keys}
        if len(hits)==1:
            return _capitalize(hits.pop())+"."
        return None
    if re.search(r"\bчто\b",q) and re.search(PLACE,q):
        hits=[e.item for e in entries if {_key(w) for w in _words(e.place)[1:]} <= query_keys]
        return _capitalize(hits[0])+"." if len(set(hits))==1 else None
    if not re.search(r"\b(?:взять|брать|выбрать|искать|нуж\w*|понадоб\w*|открыть)\b",q):
        return None
    explicit={item for item,purpose in applications if purpose & query_keys}
    hints=[item for goal,item in PURPOSES if re.search(r"\b(?:"+goal+r")\b",q)]
    hits=[e for e in entries if _item_key(_words(e.item)[-1]) in explicit or
          not explicit and any(re.search(r"\b(?:"+h+r")\b",normal(e.item)) for h in hints)]
    if len(hits)==1:
        return _capitalize(hits[0].item)+" "+hits[0].place+"."
    return None


def _contradiction(facts,query):
    if not CONFLICT.search(query):
        return None
    yes,no=set(),set()
    for clause in _clauses(facts):
        terms=_words(normal(clause))
        if terms[:1] in (["что"],["но"],["а"]):
            terms=terms[1:]
        if len(terms)<2 or any(w in {"или","мог","если","вчера","завтра"} for w in terms):
            continue
        # Only plain logical negation of the same complete proposition.
        # Opposite-looking words alone are insufficient to establish this.
        target=no if terms.count("не")==1 else yes
        target.add(tuple(w for w in terms if w!="не"))
    if yes & no:
        return "Эти утверждения противоречат друг другу. Одно прямо отрицает то, что сказано в другом."
    return None


def _packages(facts,query):
    if not re.search(r"сколько",query,re.I) or CHANGE.search(query) or re.search(r"\b(?:не|кроме|половин\w*|но|убрал\w*|взял\w*)\b",facts,re.I):
        return None
    portions=list(re.finditer(rf"\bпо\s+({NUMBER})\s+({W})\b",facts,re.I))
    if len(portions)!=1:
        return None
    per=portions[0]
    other=facts[:per.start()]+" "+facts[per.end():]
    groups=list(re.finditer(rf"\b({NUMBER})\s+({W})\b",other,re.I))
    if len(groups)!=1 or _item_key(per[2]) not in {_item_key(w) for w in _words(query)}:
        return None
    if re.search(r"[.!?;]\s",facts) and not re.search(r"кажд\w*",facts,re.I):
        return None
    if _item_key(groups[0][2])==_item_key(per[2]):
        return None
    return str(_number(per[1])*_number(groups[0][1]))+"."


def _stated_counts(facts,query):
    if (not re.search(r"\bсколько\b",query,re.I) or CHANGE.search(facts+" "+query)
            or re.search(r"\b(?:не|раза|раз|больше|меньше|по)\b",facts,re.I)):
        return None
    entries=[]
    last_item=None
    for clause in _clauses(facts):
        matches=list(re.finditer(rf"\b({NUMBER})\b",clause,re.I))
        if len(matches)!=1:
            return None
        amount=matches[0]
        before,after=clause[:amount.start()],clause[amount.end():]
        loc=re.search(rf"\b(?:в|на|у)\s+({W})(?:\s+({W}))?",clause,re.I)
        place=_item_key(loc[2] if loc[2] and adj(loc[1]) else loc[1]) if loc else None
        item=None
        for word in _words(after):
            if normal(word) in PLACES:
                break
            if not adj(word) and normal(word) not in IGNORED and not OBJECT_STILL.fullmatch(word):
                item=_item_key(word)
                break
        if item is None:
            if not last_item or not place:
                return None
            item=last_item
        last_item=item
        entries.append((place,item,_number(amount[1])))
    terms={_item_key(w) for w in _words(query)}
    relevant=[row for row in entries if row[1] in terms]
    if not relevant:
        return None
    scoped=[row for row in relevant if row[0] and row[0] in terms]
    query_has_location=re.search(r"\b(?:на|в|у)\s+\w+",query,re.I)
    if query_has_location and not scoped and not re.search(r"во всех|в обоих|вместе",query,re.I):
        return None
    if len(scoped)==1:
        return str(scoped[0][2])+"."
    if re.search(r"всего|вместе|сумм",query,re.I):
        if len({place for place,_,_ in relevant})==len(relevant):
            return str(sum(n for _,_,n in relevant))+"."
    return str(relevant[0][2])+"." if len(relevant)==1 else None


TRANSFER = re.compile(r"\b(?:подар\w*|переда\w*|вручи\w*|прин[её]с\w*|отпра\w*|продал\w*|дал[аи]?)\b",re.I)


def _transfer(facts,query):
    event=TRANSFER.search(query)
    if not event or not re.search(r"\b(?:кто|кому)\b",query,re.I):
        return None
    anchors={_item_key(w) for w in _words(query)}
    hits=set()
    for clause in _clauses(facts):
        seen=TRANSFER.search(clause)
        if not seen or normal(seen[0])[:3]!=normal(event[0])[:3] or re.search(r"\b(?:не|если|долж\w*|планир\w*)",clause,re.I):
            continue
        source=clause[:seen.start()]
        tail=clause[seen.end():]
        names=[w for w in _words(source) if w[:1].isupper() and normal(w) not in {"в","на","потом","затем","после","утром","вчера","сегодня"}]
        recipients=[w for w in _words(tail) if w[:1].isupper() and normal(w).endswith(("у","ю","е","ии"))]
        objects={_item_key(w) for w in _words(tail) if not w[:1].isupper() and normal(w) not in IGNORED|PLACES}
        if not objects & anchors:
            continue
        result=recipients if "кому" in normal(query) else names
        if len(result)==1:
            hits.add(result[0])
    return next(iter(hits))+"." if len(hits)==1 else None


def _explicit_list(facts,query):
    if not re.search(r"(?:по порядку|слева направо|справа налево|по очереди|в таком порядке|по списку)\s*[—–:]",facts,re.I):
        return None
    tail=re.split(r"[:—–]\s*",facts,maxsplit=1)[-1]
    raw=[s.strip(" .") for s in re.split(r"\s*,\s*|\s+и\s+",tail)]
    if not 2<=len(raw)<=16 or not all(_nominal_phrase(s) for s in raw):
        return None
    q=normal(query)
    index=None
    for prefix,pos in (("перв",0),("втор",1),("трет",2),("четверт",3),("пят",4),("предпослед",-2),("послед",-1)):
        if re.search(r"\b"+prefix+r"\w*",q):
            index=pos
            break
    if index is None and re.search(r"\b(?:слева|справа|левее всех|правее всех)\b",q):
        if not re.search(r"слева направо|справа налево",facts,re.I):
            return None
        reverse="справа налево" in normal(facts)
        wants_right=bool(re.search(r"справа|правее",q))
        index=0 if reverse==wants_right else -1
    if index is not None and -len(raw)<=index<len(raw):
        return _capitalize(raw[index])+"."
    return None


OBJECT_MOTION = re.compile(r"\b(?:перелож\w*|перенес\w*|перенёс\w*|перенос\w*|перемест\w*|перестав\w*|кладу|полож\w*|спрят\w*|постав\w*|помести\w*|убрал\w*|убрала|убрали)\b",re.I)
OBJECT_STILL = re.compile(r"\b(?:леж\w*|стоит|стоял\w*|сидит|вис\w*|висел\w*|наход\w*|хран\w*|оставлен\w*|спрятан\w*)\b",re.I)
PLACES={"в","во","на","под","над","за","у","перед","внутри","около","из","со","с"}
IGNORED={"и","а","но","я","мы","мне","там","тут","потом","позже","вчера","сегодня","теперь",
         "сначала","затем","сперва","сбоку","справа","слева","рядом","далее","наконец","раньше","по-прежнему","еще","даже","также","его","ее","их","он","она","оно","они",
         "оказалась","оказался","была","был","было","уже","однажды","снова","сначала","сейчас"}


def _spatial(facts,query):
    if not re.search(r"\b(?:где|куда|откуда|(?:в|на|под|у) как\w*|что (?:теперь |сейчас )?(?:лежит|находится|хранится))\b",query,re.I):
        return None
    state=defaultdict(set)
    for clause in _clauses(facts):
        move=OBJECT_MOTION.search(clause)
        verb=move or OBJECT_STILL.search(clause)
        if not verb:
            continue
        words=_words(clause)
        low=[normal(w) for w in words]
        if any(s in low for s in ("не","если","или","либо","может")):
            return None
        locations=[]
        used=set()
        for i,w in enumerate(low[:-1]):
            if w not in PLACES:
                continue
            end=i+1
            while end<len(words)-1 and adj(words[end]) and not colour(words[end+1]):
                end+=1
            if end>=len(words) or OBJECT_STILL.fullmatch(words[end]) or OBJECT_MOTION.fullmatch(words[end]):
                continue
            used.update(range(i,end+1))
            if w not in {"из","со","с"}:
                locations.append(" ".join(words[i:end+1]))
        subjects=[(i,w) for i,w in enumerate(words) if i not in used and low[i] not in IGNORED and
                  not OBJECT_STILL.fullmatch(w) and not OBJECT_MOTION.fullmatch(w) and not adj(w)]
        if len(subjects)>1:
            subjects=[(i,w) for i,w in subjects if i!=0 or not w[0].isupper() or not move]
        if len(subjects)==1:
            item=_item_key(subjects[0][1])
        elif not subjects and len(state)==1 and any(w in low for w in ("его","ее","их","он","она","оно")):
            item=next(iter(state))
        else:
            if move:
                return None
            continue
        if move:
            state[item].clear()
        if len(locations)==1:
            state[item].add((locations[0],bool(move)))
        elif locations:
            return None
    q=normal(query)
    anchors={_item_key(w) for w in _words(q)}
    if re.search(r"\bоткуда\b",q):
        return None
    if re.search(r"\bчто\b",q):
        # The item spelling is intentionally left to the preceding, simpler
        # bindings reader for reverse lookup.
        return None
    candidates=[entry for item,places in state.items() if item in anchors for entry in places]
    if len(candidates)!=1:
        return None
    place,moved=candidates[0]
    if not moved and "куда" in q:
        return None
    if moved and "где" in q:
        return "Последнее указанное перемещение — "+place[:1].lower()+place[1:]+"."
    return _capitalize(place)+"."


def _attributes(facts,query):
    q=normal(query)
    if not re.search(r"\bцвет\w*|\bоттен\w*",q):
        return None
    state=defaultdict(set)
    for clause in _clauses(facts):
        if re.search(r"\b(?:только|не|или|если|будет|должен|может)\b",clause,re.I):
            return None
        words=_words(clause)
        for i,word in enumerate(words):
            if not colour(word):
                continue
            subject=None
            if i+2<len(words) and re.fullmatch(r"был[аои]?|оказал\w*|стал[аои]?",words[i+1],re.I):
                subject=words[i+2]
            elif i+1<len(words) and not adj(words[i+1]) and normal(words[i+1]) not in IGNORED|PLACES and not OBJECT_STILL.fullmatch(words[i+1]):
                subject=words[i+1]
            elif i and re.fullmatch(r"был[аои]?|оказал\w*|стал[аои]?|цвета",words[i-1],re.I):
                if i>=2:
                    subject=words[i-2]
            elif i and i==len(words)-1 and len(words)==2:
                subject=words[0]
            if subject:
                state[_item_key(subject)].add(word)
    match=next((values for key,values in state.items() if key in {_item_key(w) for w in _words(q)}),set())
    if len(match)!=1:
        return None
    surface=next(iter(match))
    root=_colour_key(surface)
    suffix=normal(surface)[len(root):]
    ending="его" if re.search(r"ий|ие|их|яя|ее|его|им|ей",suffix) else "ого"
    return _capitalize(surface[:len(root)]+ending)+"."


def _missing(facts,query):
    if not re.search(r"хватает ли|достаточно ли|можно ли (?:точно )?(?:узнать|определить|вычислить|установить)|чего не хватает",query,re.I):
        return None
    if re.search(r"(?:не (?:сказано|указано|названо|сообщено|известно)|неизвестн\w*)",facts,re.I):
        return "Данных недостаточно: в условии прямо указано, что нужное значение неизвестно."
    if re.search(r"скольк\w*|количеств\w*",query,re.I) and CHANGE.search(facts) and len(re.findall(rf"\b{NUMBER}\b",facts,re.I))==1 and not re.search(r"начал\w*|изначально|было",facts,re.I):
        return "Не хватает начального количества. Известно только, на сколько оно изменилось."
    return None


class Reasoning:
    def answer(self, question: str) -> str | None:
        if not isinstance(question,str) or not 8<=len(question)<=1300:
            return None
        text=_clock_words(" ".join(question.split()))
        facts,q=_split(text)
        if not ASK.search(q):
            return None
        missing=_missing(facts,q)
        if missing:
            return missing
        if UNSURE.search(text):
            return None
        for measure in (_calendar,_time,_schedule):
            value=measure(facts,q)
            if value is not None:
                return value
        if not facts or re.search(r"\b(?:возможно|кажется|допустим бы|планирует|мечтает|если)\b",facts,re.I):
            return None
        for engine in (_contradiction,_explicit_list,_ordering,_closed_set,_packages,_quantities,_stated_counts,
                       _inventory,_transfer,_attributes,_spatial,_bindings):
            answer=engine(facts,q)
            if answer is not None:
                return answer
        return None
