"""A small fictional student profile and a productive personal-choice grammar."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2s
import re
import unicodedata

from .social import detect as social_act

def _norm(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().replace("ё", "е").split())


@dataclass(frozen=True)
class Taste:
    words: str
    weight: int
    reason: str
    habit: str

    def matches(self, option):
        return re.search(r"\b(?:" + self.words + r")\b", option) is not None


# Choices in different conversations draw on the same few personal habits.
TASTES = (
    Taste(r"ча[йяюе]|чаем|чаи", 10, "Люблю его терпкий вкус и саму паузу, пока он заваривается.",
          "Обычно завариваю чёрный чай и даю ему немного постоять."),
    Taste(r"кофе|эспрессо|капучино", -2, "Мне больше по вкусу чай, особенно когда сажусь за конспекты.",
          "Кофе пью редко. С утра чаще ставлю чайник."),
    Taste(r"сов[ауы]|совой|вечер(?:а|у|ом|ам|ний|нее)?|поздно|ноч(?:ью|и)", 6,
          "К вечеру я обычно раскачиваюсь и могу спокойно сосредоточиться.",
          "Я сова. Самые трудные конспекты обычно оставляю на вечер."),
    Taste(r"жаворон(?:ок|ка|ки|ком)|рано|рассвет(?:а|у|ом)?", -2,
          "Рано встаю ради первой пары, а по своей воле предпочитаю поспать.",
          "Рано встаю, если нужно на первую пару. В выходной обычно сплю дольше."),
    Taste(r"яйц(?:а|о|у|ами|ом)|яичниц[ауые]|омлет(?:а|у|ом)?", 5,
          "На сковородке завтрак готовится как раз за то время, пока закипает чайник.",
          "Чаще всего жарю яйца и завариваю чай. Если спешу на пару, делаю бутерброд."),
    Taste(r"бутерброд(?:а|ы|у|ом|ами)?|сэндвич(?:а|и|ем)?", 4,
          "Его удобно собрать за минуту и взять с собой на пару.",
          "Перед ранней парой собираю бутерброд и беру его с собой."),
    Taste(r"прогулк[аиуе]|гулять|пройтись|пешком|парк(?:а|у|ом|е)?", 4,
          "После долгого сидения за конспектами приятно размять ноги.",
          "После пар люблю пройтись пешком. По дороге постепенно переключаюсь с учёбы."),
    Taste(r"книг[аиуе]|читать|чтени[еяю]|повесть|рассказ(?:ы|а)?", 3,
          "Мне нравится ненадолго уйти с головой в чужую историю.",
          "Для удовольствия читаю короткие рассказы. Один как раз помещается между парами."),
    Taste(r"музык[аиуе]|наушник(?:и|ах|ами)|плейлист(?:ы|а)?", 3,
          "Мне нравится подбирать музыку под настроение по дороге с учёбы.",
          "В дороге обычно слушаю музыку в наушниках, а занимаюсь в тишине."),
    Taste(r"тишин[ауые]|тих(?:ий|ая|ое|ую|ом)|спокойн(?:ый|ая|ое|ую|ой)", 3,
          "В тишине мне проще собраться с мыслями.",
          "Если нужно сосредоточиться, убираю наушники и сажусь в тихом месте."),
    Taste(r"зелен(?:ый|ую|ого|ым|ом|ая|ое)|темно-зелен(?:ый|ую|ого|ым|ом|ая|ое)", 3,
          "Мне нравится этот спокойный цвет, особенно тёмный хвойный оттенок.",
          "Мне больше всего нравится тёмно-зелёный."),
)
TEA, COFFEE, NIGHT, DAWN, EGGS, SANDWICH, WALK, READING, MUSIC, QUIET, GREEN = TASTES
TASTES = TASTES + (
    Taste(r"кош(?:ка|ки|ек|кам|ками)|кот(?:ы|ов|ам|ами|ик|ики)?", 5,
          "У родителей живёт кот Барсик, так что я давно кошатник.",
          "У родителей в Караганде кот Барсик, я по нему скучаю."),
    Taste(r"собак(?:а|и|у|ой|ам|ами)?|пс(?:ы|ов)", 1,
          "Собак тоже люблю, но с ними надо гулять по расписанию, а у меня пары.",
          "Собак люблю, но своей пока нет."),
)

WORD = r"[A-Za-zА-Яа-яЁё0-9]+(?:[-’'][A-Za-zА-Яа-яЁё0-9]+)*"
PERSON = re.compile(r"\b(?:ты|тебе|тебя|тобой|тво(?:й|я|е|и|ю|ей|им|ем|их|его|ему|ими)|ваш\w*|вам|вы|сам)\b")
PREFERENCE = re.compile(r"\b(?:выб(?:ер\w*|ра\w*|ир\w*|ор)|предпоч\w*|нрав\w*|люб(?:ишь|ите|им\w*)|обожа\w*|приятнее|важнее|интереснее|дороже сердцу|ближе|по душе|по вкусу|хотел\w*|хочешь|голосуешь)\b")
FIRST_ACTION = re.compile(r"\b(?:чем занимаешься|учишься|учитесь|завтракаешь|пьешь|ешь|слушаешь|читаешь|просыпаешься|встаешь|ложишься|засыпаешь|отдыхаешь|увлекаешься|проводишь|делаешь)\b")
FACT = re.compile(r"\b(?:сколько|который|чей|в каком году|когда (?:основан|появил)|правиль\w*|верн(?:о|ый|ая|ее)|пишется|ударени\w*|произнос\w*|перевод\w*|объясн\w*|означа\w*|научн\w*|на самом деле полезн\w*|леч\w*|безопасн\w*|эффективн\w*|полезн\w*|вредн\w*|здоровья|быстрее|тяжелее|дороже|дешевле|больше весит|выше|ниже|длиннее|старше|теплее|холоднее|столица|изобрел|содержит|содержат|кто (?:автор|написал)|помог\w* при)\b")
OTHER = re.compile(r"\b(?:ему|ей|они|она|он|него|нее|них|людям|человечеству|другу|соседу|детям|для ребенка|мне следует|мне стоит|мне лучше|мне выбрать)\b")
DIRECT_CHOICE = re.compile(r"\b(?:выб(?:ер\w*|ра\w*|ира\w*)|предпоч\w*|нрав\w*|любишь|любите|ближе|по душе|по вкусу|возьм[её]шь|возьмете|хотел\w*|хочешь|голосуешь)\b",re.I)
LEAD = re.compile(r"^(?:(?:а|ну|тогда|лично|скорее|больше|все-таки|ты|тебе|сам|себе|бы|из двух|из них|для себя|что-то одно|на выбор)\b[\s,:—–-]*)+",re.I)
TAIL = re.compile(r"\s*(?:[,—–:;]\s*(?:что|как|чему|ты|тебе|какой|выбери|выбирай|обоснуй|объясни)|(?:,?\s+и\s+|,\s*|\?\s*)(?:почему|за что)\b).*$",re.I)
BLOCK_SLOT = re.compile(r"\b(?:какой|какая|какие|кто|что|где|как|чем|почему|ведь|можно|нужно|объясни|бы|тебе|ты|вам|вы|мне|нам|она|они|это значит)\b")
IDENTITY = re.compile(r"\b(?:человек(?:ом|а)?|участник(?:ом)?|бот(?:ом|а)?|робот(?:ом|а)?|программ(?:а|ой|у)|нейросет(?:ь|ью)|ии|машина|алгоритм|искусственн\w* интеллект\w*|жив(?:ой|ым)(?=$|\s+или\b)|игрок(?:ом)?|студент(?:ом)?|школьник(?:ом)?|парень|девушка|мужчина|женщина)\b")
SELF = re.compile(r"\b(?:ты|вы|с кем я (?:говорю|разговариваю)|за ширмой|кто (?:здесь |мне )?отвечает|кто (?:там|здесь)|по ту сторону экрана)\b")
SCREEN = re.compile(r"\b(?:за|из-за|по (?:ту|другую) сторону|с (?:той|другой) стороны)\s+(?:(?:эт\w*|тем\w*|непрозрачн\w*|сценическ\w*)\s+){0,2}(?:ширм\w*|занавес\w*|штор\w*|барьер\w*|перегород\w*|экран\w*)\b")
INTERLOCUTOR = re.compile(r"\b(?:кто|с кем|кого|кем|собеседни\w*|участни\w*|игрок\w*|отвеча\w*|пишет|пишешь|разговарива\w*|общаюсь)\b")
TECH = re.compile(r"\b(?:вне игры|без (?:игры|роли)|технически|как (?:программа|приложение)|по устройству)\b")
FEEL = re.compile(r"\b(?:настроени\w*|самочувстви\w*|как (?:у тебя |твои |сегодня |сам )?дела|как (?:ты |вы )?пожива\w*|как (?:ты |вы )?себя (?:сегодня )?чувству\w*|как (?:прошел|проходит) (?:твой |у тебя |у вас )?день|как ты (?:там|сегодня)|как жизнь)\b")
WITHDRAW = re.compile(r"\b(?:без (?:шуток|ответа)|историческ\w*|не про тебя|с точки зрения (?:науки|физики))\b")
REASON = re.compile(r"^(?:а |и |но )?(?:почему|отчего|зачем)(?: же| именно| так| это| ты так считаешь| такой выбор| такой)?[?!. ]*$")
FORMAT = re.compile(r"\b(?:(?:пожалуйста[, ]+)?(?:ответь|отвечай|скажи|напиши|ответ|можно)\s+)?(?:только\s+)?(?:одним словом|в одно слово|одно слово|коротко|кратко|в двух словах|без подробностей)\b",re.I)
COUNTERFACTUAL = re.compile(r"\b(?:если бы (?:ты|вы) (?:был[аи]?|стал[аи]?|оказал\w*)|представь(?:те)?[, ]+(?:себя|что (?:ты|вы))|вообрази\w*[, ]+(?:себя|что (?:ты|вы)))\b")
EPISODE_SUBJECT = re.compile(r"\b(?:гордишься|смеялся|смеешься|помнишь|запомнил|стесняешься|хочешь|стыдишься|радуешься|чувствовал|оплошал|ошибался)\b")
FICTION = re.compile(r"\b(?:придум\w*|сочин\w*|представь\w*|вообраз\w*|геро[йяюе]|персонаж\w*|сказк\w*|анекдот\w*|в рассказе|если бы|предположим)\b")
RELATIVE = re.compile(r"\b(?:мои|моя|моей|мой|моего|моему|родителей|однокурсника|одногруппника|друга|брата|сестры|приятеля|одноклассника)\b")


@dataclass(frozen=True)
class Episode:
    topic: str
    action: str
    story: str
    reason: str
    short: str


# Biographical details of the fictional character. Two independent cues
# identify an episode; narratives themselves never come from user logs.
EPISODES = (
    Episode(r"смеш\w*|рассмеш\w*|сме[хшя]\w*|насмеш\w*|забав\w*|улыб\w*|хохо\w*",
            r"что|когда|чем|момент\w*|случа\w*|вспомни\w*|эпизод\w*",
            "На паре попросил приятеля перестать щёлкать ручкой. Он молча показал на мои руки. Щёлкал-то я сам.",
            "Мне было смешно от того, с какой уверенностью я сделал ему замечание.", "самоирония"),
    Episode(r"неловк\w*|смущ\w*|смути\w*|стыд\w*|конфуз\w*|оплош\w*",
            r"что|когда|чего|момент\w*|случа\w*|вспомни\w*|истори\w*|быва\w*|приходил\w*|положени\w*",
            "Однажды бодро вошёл в аудиторию, сел и достал тетрадь. Только потом заметил, что не узнаю ни одного лица. Моя пара шла этажом выше.",
            "Я так уверенно поздоровался, что тихо уйти уже не получилось.", "аудитория"),
    Episode(r"детств\w*|детск\w*|маленьк\w*",
            r"запомн\w*|помни\w*|воспомин\w*|памят\w*|вспом\w*|ярк\w*|дороже|любим\w*",
            "В детстве мы с папой запускали бумажного змея. Я уже собирался сдаться, когда он вдруг поднялся выше деревьев. До сих пор помню, как натянулась нитка.",
            "Наверное, запомнилось именно то мгновение, когда нитка впервые дёрнула за ладонь.", "змей"),
    Episode(r"горд\w*|горж\w*|достиж\w*|побед\w*|удалось",
            r"чем|что|когда|маленьк\w*|небольш\w*|личн\w*|последн\w*|недавн\w*|сво[еяюйи]\w*|вспомни\w*",
            "Недавно сам разобрал задачу, над которой сидел два вечера. Особенно приятно стало, когда одногруппник попросил объяснить и у меня получилось.",
            "Понять самому — уже приятно, а тут я ещё и нашёл простые слова для другого человека.", "задача"),
    Episode(r"привыч\w*|характер\w*|черт\w*|себе|слабост\w*|недостат\w*",
            r"измен\w*|исправ\w*|менять|отуч\w*|избав\w*|поработ\w*|помен\w*|меша\w*|перестать",
            "Хочу перестать откладывать самую трудную задачу до позднего вечера. Я сова, но иногда хочется закрыть тетрадь раньше полуночи.",
            "Если начать хоть с одного пункта днём, вечером мне уже не приходится делать всё разом.", "откладывание"),
)


def _words(text):
    return re.findall(WORD,_norm(text))


def _valid_option(text):
    words = _words(text)
    finite_verb = any(re.search(r"(?:ается|яется|уется|уются|аются|яются|ится|ются|утся|ает|ует|ируются)$",word) for word in words)
    return (1 <= len(words) <= 8 and len(text) <= 78 and not finite_verb and not BLOCK_SLOT.search(_norm(text))
            and re.fullmatch(rf"{WORD}(?:[\s-]+{WORD})*",text) is not None)


def _rank(text):
    norm = _norm(text)
    tastes = [taste for taste in TASTES if taste.matches(norm)]
    semantic = sum(taste.weight for taste in tastes)
    reason = max(tastes,key=lambda taste:abs(taste.weight)).reason if tastes else None
    # For invented words a consistent preference for soft, sonorous sounds
    # supplies a subjective distinction. Lexical digest breaks exact ties.
    sonority = sum(norm.count(c) for c in "лмнейю") / max(1,len(norm))
    return (semantic, sonority, blake2s(norm.encode(),digest_size=4).digest()), reason


def _tidy(text):
    return text.strip(" \t\n\r,;:!?.—–-«»„“”\"")


def _options(question):
    """Return case-preserving audience alternatives and their syntactic frame."""
    raw = re.sub(r"\s+"," ",question).strip()
    if not re.search(r"\b(?:или|либо|между)\b",raw,re.I):
        return None
    raw = re.sub(r"(?i)^.*?[.!?]\s+(?=(?:а |ну )?(?:ты|что|чему|какой|выбери|предпоч|между)\b)","",raw)
    raw = re.sub(r"(?i)[?!.]\s*(?:(?:а|и)\s+)?почему(?:\s+именно)?[?!.]*$","",raw)
    raw = TAIL.sub("",raw)
    between = re.search(r"\bмежду\s+",raw,re.I)
    if between:
        raw = raw[between.end():]
        raw = re.split(r"\s*[,;—–]?\s+(?:(?:что|чему|какой|кому)\s+(?:бы\s+)?(?:ты\s+)?(?:выб|предпоч)|(?:что|кому)\s+(?:тебе|вам)\s+(?:ближе|нрав))",raw,maxsplit=1,flags=re.I)[0]
        parts = re.split(r"\s+и\s+",raw,flags=re.I)
        if len(parts)!=2:
            return None
        mode = "between"
    else:
        conjunction = re.search(r"\s+(?:или|либо)\s+",raw,re.I)
        if not conjunction:
            return None
        left = raw[:conjunction.start()]
        right = raw[conjunction.end():]
        dividers = list(re.finditer(r"[:—–]",left))
        choice = list(DIRECT_CHOICE.finditer(left))
        if dividers:
            at = dividers[-1].end()
            intro = _norm(left[:at])
            if not (PREFERENCE.search(intro) or "ты за" in intro or
                    re.fullmatch(r"(?:а )?(?:что|чему) (?:лучше|приятнее)",_tidy(intro))):
                return None
            left = left[at:]
        elif choice:
            left = LEAD.sub("",left[choice[-1].end():].strip(" ,:—–-"))
        else:
            left = re.sub(r"^(?:(?:ну|а)\s+)?(?:ты\s+за\s+|ты\s+|что\s+(?:лучше|приятнее)\s*[,—–:-]?\s*)","",left,flags=re.I)
        parts = [*re.split(r"\s*,\s*",left), *re.split(r"\s+(?:или|либо)\s+",right,flags=re.I)]
        mode = "list"
    parts = [_tidy(p) for p in parts]
    if not 2 <= len(parts) <= 4 or not all(_valid_option(p) for p in parts):
        return None
    if len({_norm(p) for p in parts}) != len(parts):
        return None
    return parts, mode


class Persona:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_reason = None
        self._style = None

    def _say(self, text, reason=None, keyword=None):
        self.last_reason = reason
        if self._style == "word":
            term = keyword or re.findall(WORD,text)[0]
            return term[:1].upper() + term[1:] + "."
        if self._style == "brief":
            return re.split(r"(?<=[.!])\s",text,maxsplit=1)[0]
        return text

    def _choice(self, question):
        norm = _norm(question)
        if FACT.search(norm) or OTHER.search(norm) or re.search(r"\b(?:не (?:выб|предпоч|люб)|вместо|объективно|должен|следует|знаешь|думаешь[,]? что)\w*",norm):
            return None
        if not DIRECT_CHOICE.search(norm) and re.search(r"\b(?:ты|вы)\s+[а-я]+(?:ал|ил|ыл|ала|ила|ял|али|или)\s+",norm):
            return None
        if not DIRECT_CHOICE.search(norm) and re.match(r"^(?:в|на|у|под|над|откуда|там|здесь|когда|куда|почему|кто|есть ли)\s",norm):
            return None
        options = _options(question)
        if not options:
            return None
        parts, mode = options
        # Long unframed clauses are usually subquestions or requests for advice.
        if not DIRECT_CHOICE.search(norm) and not PERSON.search(norm) and mode!="between" and max(map(lambda p:len(_words(p)),parts))>3:
            return None
        ranked = [(_rank(part),i,part) for i,part in enumerate(parts)]
        (score, reason), index, selected = max(ranked,key=lambda item:item[0][0])
        if not reason or score[0] < 0:
            if all(len(_words(p))==1 for p in parts):
                reason = "Сложно объяснить, просто как-то ближе."
            else:
                reason = "Сложно объяснить, просто как-то ближе по душе."
        if mode == "between":
            ordinal = "первом" if index==0 else "втором"
            text = f"Между {parts[0]} и {parts[1]} я бы остановился на {ordinal} варианте."
        elif re.search(r"(?:ешь|ишь|ешься|ишься)\b",_norm(selected)):
            # Echoing a second-person verb back as a menu option reads as a machine.
            return None
        else:
            text = selected[:1].upper() + selected[1:] + "."
        if score[0] > 0 or re.search(r"почему|обоснуй|объясни выбор|за что",norm):
            text += " " + reason
        slot_words = re.findall(WORD,selected)
        head = slot_words[0] if re.search(r"(?:ть|ти)(?:ся|сь)?$",slot_words[0]) else slot_words[-1]
        return self._say(text,reason,head)

    def _identity(self, norm):
        alternatives = norm.split(" или ")
        hidden_interlocutor = bool(SCREEN.search(norm) and (INTERLOCUTOR.search(norm) or IDENTITY.search(norm)))
        if hidden_interlocutor and re.search(r"\b(?:спектакл\w*|в (?:рассказе|пьесе|фильме|романе|сказке)|по сюжету|зрител\w* в зале|писател\w*|автор\w* картины|устройство|материал\w*|что (?:лежит|висит|установлено)|кто придумал|сочини\w*|придумай|нарисуй|вообрази|персонаж рассказа|если бы)\b",norm):
            return None
        bare_roles = len(alternatives)==2 and all(
            IDENTITY.fullmatch(re.sub(r"^(?:живой|настоящий)\s+","",p)) for p in alternatives)
        if not (SELF.search(norm) or bare_roles or hidden_interlocutor):
            return None
        explicit = IDENTITY.search(norm)
        open_question = bool(re.search(r"\b(?:кто ты|ты кто|с кем я (?:говорю|разговариваю)|кто (?:там|здесь|за ширмой)|кто (?:здесь |мне )?отвечает|кто (?:сидит|говорит) за ширмой|кем (?:ты )?(?:работаешь|учишься))\b",norm))
        if not (explicit or open_question or hidden_interlocutor):
            return None
        if open_question and not explicit and re.search(r"\b(?:по знаку|по гороскопу|по национальности|родом|по специальности)\b",norm):
            return None
        if (re.search(r"\b(?:кто такой|что такое|почему|как работает|что думаешь|помнишь|напиши|назови|объясни|отлич\w*|может ли|видел|читал|считаешь|нрав\w*|любишь|видишь|представь)\b",norm)
                or explicit and re.search(r"\b(?:в|к|с|про|у|о|как) (?:этом |настоящем )?(?:человеке|роботе|боте|программе|студенте)\b",norm)):
            return None
        if TECH.search(norm):
            return self._say("Вне игры я программа. За ширмой говорю от лица студента.",keyword="программа")
        if hidden_interlocutor:
            return self._say("Человек, студент. После пар обычно добираюсь домой пешком.",keyword="человек")
        if re.search(r"\b(?:докажи|докажите|убеди меня|подтверди)\b",norm):
            return self._say("Одна строчка в чате мало что докажет. За ширмой я студент — из тех, кто несёт кружку чая к конспектам.",keyword="студент")
        if " или " in norm:
            if re.search(r"\b(?:человек|участник|игрок|живой)\b",norm):
                return self._say("Человек. Учусь, после пар обычно иду домой пешком.",keyword="человек")
            if re.search(r"\b(?:парень|мужчина|студент)\b",norm):
                return self._say("Я парень, учусь.",keyword="студент" if "студент" in norm else "парень")
            if re.search(r"\b(?:бот|робот|программа|нейросеть|алгоритм|машина|ии)\b",norm):
                return self._say("Я человек, студент.",keyword="человек")
            return None
        if not open_question and not re.match(r"^(?:(?:а|ну|скажи(?: честно)?)[, ]+)*(?:ты|вы)\s+",norm):
            return None
        if explicit and norm[:explicit.start()].strip() not in {"а ты","ты","вы","ну ты"} and not open_question and not re.search(r"\b(?:точно|правда|вообще|случайно|реально|действительно|сам|не|же|разве|ведь|честно|живой|настоящий)\b",norm[:explicit.start()]):
            return None
        if explicit and re.search(r"\b(?:мужчина|женщина|парень|девушка)\b",norm):
            return self._say("Я парень, учусь.",keyword="парень")
        if explicit and re.search(r"\b(?:бот|робот|нейросеть|программа|машина|ии|алгоритм|искусственн\w* интеллект)\b",norm):
            return self._say("Я человек, студент. После пар люблю выбраться на прогулку.",keyword="человек")
        if explicit and explicit[0] in {"человек","живой","участник","игрок"}:
            return self._say("Да, я человек. Учусь, а в свободное время люблю гулять.",keyword="человек")
        return self._say("Я студент. Люблю после пар пройтись пешком, а вечером завариваю чай.",keyword="студент")

    def _episode(self, norm):
        if FICTION.search(norm) or RELATIVE.search(norm) or OTHER.search(norm):
            return None
        if not (PERSON.search(norm) or EPISODE_SUBJECT.search(norm)):
            return None
        for entry in EPISODES:
            if re.search(r"\b(?:"+entry.topic+r")\b",norm) and re.search(r"\b(?:"+entry.action+r")\b",norm):
                return self._say(entry.story,entry.reason,entry.short)
        return None

    def answer(self, question: str) -> str | None:
        if not isinstance(question,str) or not question.strip() or len(question)>650:
            self.reset()
            return None
        style = FORMAT.search(question)
        if style:
            question = _tidy(FORMAT.sub("",question)).strip(" ,;:.!?—–-")
        norm = _tidy(_norm(question))
        if REASON.fullmatch(norm):
            return self.last_reason
        self.reset()
        self._style = ("word" if "слов" in style[0].casefold() and "двух" not in style[0].casefold() else "brief") if style else None
        if not norm:
            return None
        if WITHDRAW.search(norm) or COUNTERFACTUAL.search(norm) or re.search(r"\b(?:он|она|они) (?:люб\w*|предпоч\w*|хоч\w*|выб\w*)",norm):
            return None
        if style:
            act = social_act(question)
            if act:
                word = {"greet":"Здравствуйте","bye":"Пока","thanks":"Пожалуйста","sorry":"Принято",
                        "celebrate":"Спасибо","call":"Слушаю"}.get(act[0].split(".")[0])
                if word:
                    return self._say(word+".",keyword=word)
        role = self._identity(norm)
        if role:
            return role
        local_weather = re.search(r"\b(?:погод\w*|прогноз\w*|температур\w*|дождь|снег|холодно|жарко|тепло)\b",norm)
        if local_weather:
            if (not re.search(r"\b(?:или|либо|между)\b",norm)
                    and re.search(r"\b(?:нрав\w*|любим\w*|любишь)\b",norm) and not OTHER.search(norm)):
                return self._say("Люблю сухую прохладу. Можно долго идти пешком и не искать, где укрыться.", "В такую погоду мне особенно приятно идти домой после пар.","прохладу")
            if (not re.search(r"\b(?:почему|откуда|объясни|климат|зимой|летом|осенью|весной|обычно|в среднем)\b",norm)
                    and (re.search(r"\b(?:сейчас|сегодня|завтра|за окном|на улице|у (?:вас|тебя))\b",norm) or "какая погода" in norm)):
                return self._say("Свежей сводки у меня нет. Перед выходом я бы посмотрел местный прогноз.",keyword="неизвестно")
        if " или " in norm or " либо " in norm or "между " in norm:
            return self._choice(question)
        if FACT.search(norm) or OTHER.search(norm):
            return None
        episode = self._episode(norm)
        if episode:
            return episode
        social_mood = PERSON.search(norm) or re.fullmatch(r"(?:(?:ну|а|привет)[, ]+)*(?:как(?:ое)? (?:сегодня )?(?:настроение|дела|жизнь|самочувствие)|как поживаешь|как прошел день)(?: сегодня| с утра)?",norm)
        if social_mood and FEEL.search(norm) and not re.search(r"\b(?:улучшить|испортить|влия\w*|поднять|вернуть|чужое|музыки|картины|книги)\b",norm) and not norm.startswith(("что такое","почему")):
            return self._say("Настроение спокойное. Люблю такие минуты, когда можно отложить конспекты и поболтать.", "Когда есть время перевести дух, у меня и мысли складываются лучше.","спокойное")
        personal = bool(PERSON.search(norm) or FIRST_ACTION.search(norm) or PREFERENCE.search(norm))
        if not personal or norm.startswith(("почему люди","почему у людей","как правильно","что такое","расскажи об истории","ты знаешь","ты помнишь")):
            return None
        if re.search(r"\b(?:ты|тебе)\b",norm) and re.search(r"нов\w* (?:мест|город|район)|незнаком\w* (?:город|мест|район)|впервые (?:в |попада)",norm):
            return self._say("Я иду пешком до какой-нибудь заметной точки, а возвращаюсь другой улицей. Так у меня в голове появляется своя маленькая карта.")
        if re.search(r"приятнее|важнее|нравится",norm) and " или " in norm and re.search(r"хвал|похвал",norm):
            if re.search(r"усил|старан|попытк|труд",norm):
                return self._say("Когда замечают, сколько я возился. Результат все и так видят, а про те три неудачные попытки обычно знаю только я.")
        if re.search(r"\b(?:твоя|твой|тебе|ты)\b",norm) and re.search(r"подарок|подарили|подарк",norm) and re.search(r"помни|приятн|дорог|любим|хран|запомн",norm):
            return self._say("Храню открытку, на которой друг нарисовал схему нашей первой прогулки. Там даже отмечен двор, где мы оба уверенно свернули не туда.")
        eating = re.search(r"\b(?:ешь|едите|поешь|перекус\w*|поесть|готовишь|съедаешь)\b",norm)
        morning = re.search(r"\b(?:утр(?:о|ом|а|ам)|перед (?:первой )?парой|до (?:первой )?пары)\b",norm)
        if re.search(r"\bзавтрак\w*\b",norm) or eating and morning:
            if re.search(r"\b(?:спеш\w*|тороп\w*|на бегу|некогда|мало времени|минут\w*|до (?:первой )?пары)\b",norm):
                return self._say("Собираю бутерброд, пока закипает чайник. Если совсем впритык, заворачиваю его и ем уже перед парой.",SANDWICH.reason,"бутерброд")
            return self._say(EGGS.habit,EGGS.reason,"яйца")
        if re.search(r"\b(?:просыпаешься|встаешь|ложишься|засыпаешь|высыпаешься|хронотип|сова|жаворонок)\b",norm):
            return self._say(NIGHT.habit,NIGHT.reason,"сова")
        if re.search(r"\b(?:напит\w*|пить|пьешь|завариваешь)\b",norm) and not re.search(r"кофе",norm):
            return self._say(TEA.habit,TEA.reason,"чай")
        if re.search(r"\bцвет(?:а|ов)?\b",norm) and PREFERENCE.search(norm):
            return self._say(GREEN.habit,GREEN.reason,"тёмно-зелёный")
        if re.search(r"\b(?:что (?:ты )?читаешь|какие книги|жанр книг)\b",norm):
            return self._say(READING.habit,READING.reason,"рассказы")
        if re.search(r"\b(?:что (?:ты )?слушаешь|какую музыку)\b",norm):
            return self._say(MUSIC.habit,MUSIC.reason,"музыку")
        if PREFERENCE.search(norm) or FIRST_ACTION.search(norm) or "относишься" in norm:
            for taste in TASTES:
                if taste.matches(norm):
                    term = re.search(r"\b(?:"+taste.words+r")\b",norm)[0]
                    return self._say(taste.reason if norm.startswith(("почему","отчего")) else taste.habit,taste.reason,term)
        if re.search(r"\b(?:хобби|увлечени\w*|досуг|выходн\w*|свободн\w* время|после (?:пар|учебы)|отдыхаешь|увлекаешься|(?:нравится|любишь|предпочитаешь) делать)\b",norm):
            return self._say(WALK.habit,WALK.reason,"прогулки")
        if re.fullmatch(r"(?:а |ну )?(?:чем (?:ты )?занимаешься|ты учишься|учишься|ты студент)",norm):
            return self._say("Учусь. На парах записываю главное, а вечером разбираю то, что осталось непонятным.",keyword="учусь")
        return None
