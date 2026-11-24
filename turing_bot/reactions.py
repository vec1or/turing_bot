"""Direct speech for a reported lateness, composed from the stated circumstances.

The module reads a small explicit set of conditions from the question: who was
late (another person or the speaker), whether it happened again, whether an
apology or a warning was given and how long the wait was. Each combination maps
to a short reply written for the fictional student. Anything outside this set
is left to other components.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s
import re
import unicodedata


LATE = re.compile(r"\b(?:опоздал[аи]?|опоздает|опоздают|задержится|опаздыва\w*|опоздани\w*|задержал(?:ся|ась|ись)|пришл?[аи]? (?:позже|с опозданием)|проспал[аи]?)\b")
SPEECH = re.compile(r"\b(?:ответи(?:л[аи]?|шь|те|ть)|отв[её]тишь|отвеча(?:ть|ешь)|отреагир\w*|реагир\w*|(?:что|как) (?:бы )?(?:(?:ты|вы|я|ему|ей|им|мне) )*(?:бы )?(?:ска(?:жешь|зать|зал[аи]?|жете)|напиш(?:ешь|у)|написал[аи]?)|твоя реакция|что (?:бы )?ты (?:ему |ей |им )?сказал[аи]?)\b")
ADVICE = re.compile(r"\b(?:как (?:не|перестать|отучить|избавиться|приучить)|почему (?:люди|некоторые|он|она|они)|что делать,? чтобы)\b")
APOLOGY = re.compile(r"\b(?:извинил(?:ся|ась|ись)|извиня(?:лся|лась|лись|ется)|попросил[аи]? прощения|сказал[аи]? .{0,12}прости)\b")
NO_APOLOGY = re.compile(r"\b(?:не (?:извинил|извиня|попросил[аи]? прощения)\w*|без извинени\w*|даже не извин\w*|ни слова)\b")
AGAIN = re.compile(r"\b(?:опять|снова|в который раз|постоянно|всегда|вечно|каждый раз|регулярно|(?:уже )?(?:второй|третий|четв[её]ртый|пятый|\d+-?(?:й|ий)) раз|не в первый раз|часто)\b")
WARNED = re.compile(r"\b(?:предупредил[аи]?|предупреждал[аи]?|написал[аи]? заранее|заранее написал[аи]?|позвонил[аи]? заранее|заранее позвонил[аи]?|сообщил[аи]? заранее)\b")
NOT_WARNED = re.compile(r"\b(?:не предупредил\w*|не предупреждал\w*|без предупреждени\w*|не написал\w*|не позвонил\w*)\b")
SELF = re.compile(r"^(?:я|если я)\b|\b(?:я (?:сам |сама )?(?:опоздал|опаздываю|проспал|задержал)|ты (?:сам |сама )?(?:опоздал|проспал|задержал)\w*|если ты опоздал\w*)")
FORMAL = re.compile(r"\b(?:преподавател\w*|учител\w*|начальни\w*|руководител\w*|директор\w*|декан\w*|клиент\w*|заказчик\w*)\b")
FEMALE = re.compile(r"\b(?:опоздала|задержалась|проспала|пришла|извинилась|подруга|подружка|сестра|девушка|одногруппница|однокурсница|коллега,? она)\b")
DURATION = re.compile(r"\bна\s+((?:целых\s+|почти\s+|минут\s+)?(?:полчаса|час|полтора часа|два часа|три часа|пару часов|\d{1,3}\s*(?:минут[уы]?|мин|час(?:а|ов)?)|(?:пять|десять|пятнадцать|двадцать|сорок)\s+минут|несколько минут|пару минут|минутку))\b")
MINUTES = (("полчаса",30),("полтора часа",90),("два часа",120),("три часа",180),("пару часов",120),
           ("пару минут",2),("несколько минут",5),("минутку",1),("пять",5),("десять",10),
           ("пятнадцать",15),("двадцать",20),("сорок",40),("час",60))


def _norm(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().replace("ё", "е").split())


def _minutes(phrase):
    if not phrase:
        return None
    digits = re.search(r"\d+", phrase)
    if digits:
        value = int(digits[0])
        return value * 60 if "час" in phrase else value
    for word, value in MINUTES:
        if word in phrase:
            return value
    return None


@dataclass(frozen=True)
class Lateness:
    self_late: bool
    again: bool
    apology: bool
    no_apology: bool
    warned: bool
    formal: bool
    female: bool
    wait: str | None
    minutes: int | None


def read(question):
    norm = _norm(question)
    if len(norm) > 400 or not LATE.search(norm) or not SPEECH.search(norm) or ADVICE.search(norm):
        return None
    wait = DURATION.search(norm)
    phrase = wait[1].replace("целых ", "").replace("почти ", "").strip() if wait else None
    return Lateness(
        self_late=bool(SELF.search(norm)),
        again=bool(AGAIN.search(norm)),
        apology=bool(APOLOGY.search(norm)) and not NO_APOLOGY.search(norm),
        no_apology=bool(NO_APOLOGY.search(norm)),
        warned=bool(WARNED.search(norm)) and not NOT_WARNED.search(norm),
        formal=bool(FORMAL.search(norm)),
        female=bool(FEMALE.search(norm)),
        wait=phrase,
        minutes=_minutes(phrase),
    )


def _pick(options, question):
    digest = blake2s(_norm(question).encode(), digest_size=2).digest()
    return options[int.from_bytes(digest, "big") % len(options)]


def answer(question):
    case = read(question)
    if case is None:
        return None
    a = "а" if case.female else ""
    wait = case.wait
    if case.self_late:
        late = f"на {wait}" if wait else ""
        text = (f"Извините, что опоздал{' ' + late if late else ''}. Это моя ошибка, дальше постараюсь выходить заранее."
                if case.formal else
                f"Прости, что опоздал{' ' + late if late else ''}. Знаю, ждать неприятно. Давай чай за мной, а в следующий раз выйду пораньше.")
        return f"Сказал бы: «{text}»"
    if case.formal:
        text = ("Ничего, я как раз успел ещё раз пробежаться по своим записям. Можем начинать."
                if case.apology or case.warned else
                "Ничего, я подождал. Можем начинать.")
        return f"Сказал бы: «{text}»"
    if case.warned and not case.again:
        text = f"Спасибо, что предупредил{a}. Я пока посижу, чай возьму, не торопись."
        return f"Сказал бы: «{text}»"
    if case.again:
        if case.apology:
            text = ("Извинения принимаю, но это уже не первый раз. Давай договоримся: если не успеваешь, пиши заранее, а то я в следующий раз начну без тебя.")
        else:
            text = ("Слушай, это уже не первый раз. Мне правда неприятно столько ждать. Если не успеваешь, просто напиши заранее, ладно.")
        return f"Сказал бы: «{text}»"
    if case.no_apology:
        waited = f"Я тебя {wait} ждал" if wait else "Я тебя долго ждал"
        text = f"{waited} и уже начал волноваться. Мне правда неприятно, когда так. В следующий раз хоть напиши, что задерживаешься."
        return f"Сказал бы: «{text}»"
    minutes = case.minutes
    if minutes is not None and minutes <= 15:
        text = _pick((f"Да ладно, {wait} — ерунда. Пошли уже.",
                      f"Ничего, {wait} — не страшно. Я сам только подошёл."), question)
    elif minutes is not None and minutes > 60:
        text = (f"Ладно, извинения приняты. Но {wait} — это много, я уже начал волноваться. "
                f"В следующий раз хотя бы напиши, что задерживаешься.") if case.apology else (
                f"{wait[:1].upper() + wait[1:]} — это много, я уже волновался. Хорошо, что всё в порядке. В следующий раз напиши, пожалуйста.")
    elif case.apology:
        spent = f"За {wait} я успел взять чай" if wait else "Я пока успел взять чай"
        text = _pick((f"Ничего страшного, бывает. {spent}, так что пошли. Только в следующий раз напиши, если задерживаешься.",
                      f"Да ладно, бывает. Хорошо, что извинил{'ась' if case.female else 'ся'}. {spent}, так что всё нормально. В следующий раз просто напиши заранее."),
                     question)
    else:
        text = "Ничего, бывает. Главное, что " + ("пришла" if case.female else "пришёл") + ". Пойдём, а то и так время потеряли."
    return f"Сказал бы: «{text}»"
