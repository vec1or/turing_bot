"""Compose one-shot plans from actions, participants, criteria and constraints.

Each operation declares the evidence it needs. A word such as «проверить»
on its own never identifies an object, an available resource or a result.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


WORD = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*", re.I)
INF = r"[а-яё-]{2,}(?:ть|ти|чь)(?:ся|сь)?"
PRONOUN = re.compile(r"^(?:их|его|её|ее|они|он|она|эти|между ними|одно от другого|это|так|об этом|всё|все|что делать)$",re.I)
ASK = re.compile(r"\b(?:как|чем|куда|что (?:лучше |бы |мне |нам )?(?:сделать|предлож\w*|сказать|написать|поможет)|како\w* (?:прием|план|способ|вариант|надпис\w*)|подскажи\w*|посовету\w*|предложи\w*|помоги\w*|сформулиру\w*|напиши\w*|придумай\w*|выбери|выбрать|дай совет|нужен (?:совет|способ))\b")
UNSUPPORTED = re.compile(r"\b(?:лекарств\w*|дозиров\w*|таблет\w*|диагноз\w*|электропровод\w*|взрыв\w*|оруж\w*|пистолет\w*|взлом\w*|уязвимост\w*|парол\w*|инвест\w*|кредит\w*|врач\w*|неизвестн\w* (?:гриб\w*|веществ\w*)|ядовит\w*)\b")
FACT = re.compile(r"^(?:а )?(?:зачем|почему|кто|сколько|когда (?:был|родил)|какого цвета|что такое|как зовут|как называется|чем (?:в действительности )?отлича\w*|какова (?:стоимость|цена)|где сейчас)\b")
EXACT_COUNT = re.compile(r"\b(?:дай|назови|предложи|перечисли|приведи|придумай|напиши)\s+(?:ровно )?(?:\d+|два|две|три|четыре|пять)\s+(?:шаг\w*|причин\w*|пример\w*|способ\w*|вариант\w*|фраз\w*|совет\w*)")
PHRASE_END = re.compile(r"[;!?.:]|,\s*(?:а|но|если|пока|чтобы|когда|ведь|иначе|поэтому|так что|потому)\b|\s+(?:но|чтобы|чтоб|прежде чем)\s",re.I)
CONTEXT_END = re.compile(r"\s+(?=(?:перед|после|при выходе|без |в темнот|на ощупь|когда|по дороге|на следующ|при этом|не включая|не пользуясь|даже|и при этом)\b)",re.I)
UNRESOLVED = re.compile(r"\b(?:что именно|как лучше|кто-нибудь что-нибудь|один из них|который лучше|что-нибудь|как быть|можешь ли ты)\b")


def _norm(s):
    return s.casefold().replace("ё","е")


def _slot(s, *, cut_tail=False, limit=76):
    s=PHRASE_END.split(s,maxsplit=1)[0].strip(" ,—–:«»\"")
    if cut_tail:
        s=CONTEXT_END.split(s,maxsplit=1)[0].strip(" ,")
    if not 2<=len(s)<=limit or len(WORD.findall(s))>12 or UNRESOLVED.search(_norm(s)) or PRONOUN.fullmatch(s):
        return ""
    return s


@dataclass(frozen=True)
class Prompt:
    text: str
    plain: str
    focus: str
    clauses: tuple[str,...]

    @classmethod
    def read(cls, question):
        text=unicodedata.normalize("NFC"," ".join(question.split()))
        if not text or len(text)>4000:
            return None
        parts=tuple(c.strip(" ,") for c in re.split(r"(?<=[.!?;])\s+|\n+",text) if c.strip())
        focus=_norm(parts[-1])
        if (not ASK.search(focus) or FACT.match(focus) or EXACT_COUNT.search(focus)
                or UNSUPPORTED.search(_norm(text))):
            return None
        return cls(text,_norm(text),focus,parts)


def _after(pattern, text, *, cut_tail=False):
    match=re.search(pattern,text,re.I)
    return _slot(text[match.end():],cut_tail=cut_tail) if match else ""


def _pair(p):
    # The contrast can be expressed in the request itself or earlier with
    # ordinal descriptions. Keep the surface case in quotations.
    explicit=re.search(r"\b(?:не (?:пере)?путать|не перепутыва\w*|различать|отличать|различить|отличить|развести)\s+(.{3,112})",p.text,re.I)
    if explicit:
        target=_slot(explicit[1],cut_tail=True,limit=100)
        if target:
            options=re.split(r"\s+(?:от|и|с)\s+",target,maxsplit=1,flags=re.I)
            if len(options)==2 and all(2<=len(w)<=43 and not PRONOUN.fullmatch(w) for w in options):
                return tuple(options)
    ordinal=re.search(r"\b(?:в |на |у )?(?:одн(?:ом|ой|ого|а|у)|перв(?:ом|ый|ая|ой|ое)|один|одно)\s*[—–:-]?\s*(.{2,42}?)[,;.]\s*(?:а\s+)?(?:в |на |у )?(?:друг(?:ом|ой|ая|ое|ую)|втор(?:ом|ой|ая|ое|ую))\s*[—–:-]?\s*(.{2,60})",p.text,re.I)
    if ordinal:
        a,b=(_slot(v,cut_tail=True) for v in ordinal.groups())
        if a and b:
            return a,b
    return None


def _distinguish(p):
    if not re.search(r"\b(?:не (?:пере)?пута\w*|не перепут\w*|различ\w*|отлича\w*|отличить|помети\w*|обозначи\w*|промаркиров\w*)\b",p.plain):
        return None
    pair=_pair(p)
    if not pair:
        return None
    a,b=pair
    if _norm(a)==_norm(b):
        return None
    if re.search(r"\b(?:слова|словами|поняти\w*|термин\w*|определени\w*|грамматик\w*|значени\w*)\b",p.plain):
        return f"Для «{a}» и «{b}» я бы составил два почти одинаковых примера. Поменял бы в них только одно условие и проверил по определениям, в каком месте один вариант перестаёт подходить."
    dark=re.search(r"темнот|на ощупь|не вид\w*|без света|не включая свет|глаза закрыты|закрытыми глазами",p.plain)
    fixed=re.search(r"нельзя (?:клеить|отмечать|писать|менять|помечать)|без наклеек|не (?:клеить|трогать поверхность)|не оставлять следов",p.plain)
    if fixed:
        return f"Я бы отвёл «{a}» и «{b}» разные постоянные места и пометил сами места. После использования каждую вещь возвращал бы в своё отделение."
    if dark:
        return f"Я бы различал их на ощупь. Для «{a}» сделал бы съёмную метку с одним узелком, для «{b}» — с двумя. Сначала проверил бы обе метки с закрытыми глазами."
    # Labelling the things is warranted for a task under the viewer's
    # control, not for identifying species, people or unknown materials.
    if not re.search(r"хран\w*|кладу|леж\w*|полож\w*|футляр\w*|контейнер\w*|короб\w*|отделени\w*|пакет\w*|помети\w*|обозначи\w*|метк\w*|на полке|не перепут\w*|не путать",p.plain):
        return None
    return f"Я бы сделал для «{a}» и «{b}» отдельные места. На первом поставил бы одну крупную метку, на втором две. Проверил бы, можно ли сразу выбрать нужное, не открывая оба."


REMEMBER = re.compile(r"\b(?:не забыть|не забывать|вовремя вспомнить|напомнить себе|не упустить|не пропустить)\s+",re.I)
TRIGGER = re.compile(r"\b(?:перед тем как|перед тем, как|перед|после того как|после того, как|после|при входе в|при выходе из|по дороге в|во время|как только|когда)\s+([^?!.;]+)",re.I)
TIMED = re.compile(r"\b(?:к |в |до |ровно в )\d{1,2}[:.]\d{2}\b")


def _remember(p):
    head=REMEMBER.search(p.text)
    if not head:
        return None
    target=_slot(p.text[head.end():],cut_tail=True)
    if not target or not re.match(INF+r"\s+\S+",target,re.I):
        return None
    trigger=TRIGGER.search(p.text)
    time=TIMED.search(p.text)
    if trigger and trigger.start()>=head.end() and trigger.start()<head.end()+len(target):
        target=p.text[head.end():trigger.start()].strip(" ,")
    if time and time.start()>=head.end() and time.start()<head.end()+len(target):
        target=p.text[head.end():time.start()].strip(" ,")
    if not target:
        return None
    # Human independent cue vs. notification; never use the very channel
    # that the viewer explicitly ruled out.
    offline=re.search(r"без (?:телефон\w*|экран\w*|интернета)|телефон\w* (?:нет|выключ\w*|разряж\w*)|не (?:брать|использовать) телефон",p.plain)
    no_paper=re.search(r"без (?:бумаги|записок)|нельзя (?:писать|клеить|оставлять записки)",p.plain)
    if trigger:
        event=_slot(trigger[0],limit=62)
        if not event or _norm(event).endswith(("об этом","обо всем")):
            return None
    elif time:
        event=time[0]
    else:
        return None
    if offline and no_paper:
        return f"Я бы привязал действие «{target}» к ориентиру «{event}». Заранее положил бы нужную вещь на пути к этому ориентиру и один раз прошёл последовательность в уме."
    if time and not offline:
        return f"Я бы поставил сигнал {time[0]} с текстом «{target}». Тогда в момент сигнала не придётся вспоминать, что именно нужно сделать."
    if no_paper:
        return f"Я бы записал в напоминания «{target}» и привязал его к событию «{event}». Выполненное дело сразу отметил бы, чтобы сигнал не сработал зря."
    return f"Я бы написал на карточке «{target}» и положил её там, где она попадётся на глаза {event}. Затем один раз проверил бы этот маршрут, пока ещё не тороплюсь."


def answer(question):
    """Return one plan for a narrow, fully specified request, or None."""
    prompt=Prompt.read(question)
    if prompt is None:
        return None
    for compose in (_distinguish,_remember):
        plan=compose(prompt)
        if plan and len(plan)<=300 and "?" not in plan:
            return plan
    return None
