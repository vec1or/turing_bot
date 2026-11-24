"""Short audience writing tasks, including a corpus-trained word completer."""
import gzip
import json
from pathlib import Path
import re
from .imagination import objects_in


SENSES = (
    (r"звук|услыш|звуч", "sound"), (r"запах|аромат|пахн", "scent"),
    (r"оттенок|цвет", "colour"), (r"вкус", "taste"),
)
# In a blank slot following a past-tense verb, a subject must agree with it.
# These words constrain the grammatical candidates for creative continuations.
NOUNS = (
    ("кот пёс голубь щенок воробей сторож музыкант ветер фонарь автобус самолёт туман месяц дождь", "m"),
    ("кошка птица белка собака чайка лиса сова девочка луна лодка машина туча стрекоза", "f"),
    ("море облако солнце такси эхо лето дерево кресло письмо радио время", "n"),
    ("птицы коты огни звёзды люди друзья дети волны деревья фонари соседи тени", "p"),
)
PLACES = (
    ("крыш чердак подоконник окн", "кот кошка голубь воробей птицы коты"),
    ("лес ветк дерев полян", "белка сова птица ветер дерево деревья"),
    ("мор волн порт рек берег", "чайка лодка море туман волны"),
    ("улиц дорож остановк дорог", "автобус машина такси фонарь люди фонари"),
    ("кухн комнат кроват диван", "кот кошка щенок кресло дети"),
    ("неб облак горизонт", "месяц самолёт луна облако солнце звёзды"),
)
ANIMATE_VERB = re.compile(r"дрема|спа[лв]|засну|усну|мурлыка|жда[лв]|сидел|прошепта")
ANIMATE = set("кот пёс голубь щенок воробей сторож музыкант кошка птица белка собака чайка лиса сова девочка птицы коты люди друзья дети соседи".split())


class Writing:
    def __init__(self, directory=None):
        root = Path(directory) if directory else Path(__file__).with_name("assets")
        with gzip.open(root/"completion.json.gz","rt",encoding="utf-8") as stream:
            self.next_word = json.load(stream)["entries"]

    def _complete(self, prefix):
        words = re.findall("[а-яё]+",prefix.casefold())
        if not words:
            return None
        last = words[-1]
        actual = self.next_word.get(" ".join(words[-2:]), []) + self.next_word.get(last,[])
        if re.search(r"(?:[аиыеяоу]л[аои]?|нул[аои]?)$",last):
            gender = "f" if last.endswith("ла") else "n" if last.endswith("ло") else "p" if last.endswith("ли") else "m"
            candidates = next(v.split() for v,g in NOUNS if g == gender)
            place = " ".join(words)
            preferred = {noun for cues,nouns in PLACES if any(cue in place for cue in cues.split()) for noun in nouns.split()}
            if ANIMATE_VERB.search(last):
                candidates = [w for w in candidates if w in ANIMATE]
            if candidates:
                ranks = dict(actual)
                return max(candidates,key=lambda word: (word in preferred, ranks.get(word,0), -candidates.index(word))).capitalize()+"."
        # After a preposition or adjective, common inflected corpus continuations
        # are safer than guessing the required grammatical form.
        skip = set("что как когда это его мне тебя него всех ещё если очень чтобы просто самое уже были быть всегда себя который этом".split())
        for word, count in actual:
            if count>=2 and len(word)>3 and word not in skip and word not in words:
                return word.capitalize()+"."
        return None

    def answer(self, question, imagination):
        q = question.casefold().replace("ё","е")
        invent = re.search(r"придум\w*|изобрет\w*|назови|назва\w*|окрести",q)
        device = re.search(r"устройств\w*|прибор\w*|машинк\w*|робот\w*|изобретени\w*",q)
        if invent and device:
            relation = re.search(r"(?:котор\w*|умеет|для того чтобы)\s+(?:(?:может|будет|умеет)\s+)?(ищет|находит|искать|собирает|собирать|чистит|чистить|сортирует|сортировать)\s+([а-я]{3,20})",q)
            if relation:
                thing=relation[2]
                if thing not in {"только","потому","ровно","любые","когда","каждую","каждый"}:
                    stem=thing.rstrip("аиеыуюь")
                    affix="лов" if relation[1] in {"ищет","находит","искать","собирает","собирать"} else "чист" if relation[1].startswith("чист") else "сорт"
                    return "«"+(stem+"о"+affix).capitalize()+"»."
        if re.search(r"придум\w*|введ\w*|объя\w*|изобрет\w*|сдела\w*",q) and re.search(r"праздник|традици",q):
            scene=imagination.scene(re.findall("[а-я]+",q))
            if scene:
                return f"Я бы назвал это «{scene.name}». На входе все закрывают глаза и пытаются узнать друг друга по одному звуку. Для начала пусть звучит {scene.sound}."
            return "День незаконченных дел. Каждый приносит одну мелочь, которую давно откладывал, и мы за чаем помогаем друг другу наконец её доделать."
        if re.search(r"заголовок|озаглав|как (?:бы )?называ\w* (?:эта|так\w*) истори",q):
            cited=re.findall(r'[«„“"]([^«»"”]{2,24})[»”"]',question)
            if len(cited)==1 and "?" not in cited[0]:
                return f"«Один день из жизни: {cited[0]}»."
            scene=imagination.scene(re.findall("[а-я]+",q))
            if scene:
                return "«"+scene.comparison[0].upper()+scene.comparison[1:]+"»."
        if re.search(r"дополни|закончи|продолжи|пропуск|вставь слово",q):
            prefix = re.search(r'[«„“"]([^«»"”\n]{1,140}?)(?:\s*[…_]+|\s*\.{2,}|[»”"])', question)
            if prefix:
                return self._complete(prefix[1])
        sense = next((field for expression,field in SENSES if re.search(expression,q)),None)
        figurative = re.search(r"напомина|ассоци|вспомина|для тебя|у тебя|похож|по-твоему|мысленн|представ|выдум|в твоем воображении|опиши",q)
        feeling = re.search(r"радост|тревог|детств|осен|весн|лет[оау]|зим[ауы]|ноч[ьию]|понедельник|утр[аоу]|дружб|счасть|настроен|выходн|тишин|ожидан|воспомин",q)
        if sense and (figurative or feeling) and not re.search(r"\b(?:кто|где лежит|сколько|почему|зачем|когда изобрели)\b",q):
            scene = imagination.scene(re.findall("[а-я]+",q))
            if scene:
                result = getattr(scene,sense)
                if re.search(r"одн(?:им|о) слов",q):
                    result=result.split()[-1] if sense=="scent" else result.split()[0]
                return result[0].upper()+result[1:]+"."
        if re.search(r"придум|сочин|как бы ты назв|изобрети",q) and re.search(r"новое слово|несуществующ\w* слово|неологизм|выдуманное слово",q):
            tail = re.split(r"для (?:того,? )?|обознача\w* |означа\w* |про |когда ",q,maxsplit=1)
            if len(tail)>1:
                stop={"когда","того","чтобы","человек","который","состояния","состояние","чувства","чувство","очень","после","перед","такой"}
                terms=[w for w in re.findall("[а-я]+",tail[1]) if len(w)>3 and w not in stop]
                if terms:
                    stem=terms[-1][:5].rstrip("аоуеиыь")
                    coinage=stem+"инка" if stem.endswith(("т","д","в","р","н")) else stem+"унчик"
                    return coinage.capitalize()+"." if re.search(r"одним словом|только слово",q) else f"Я бы назвал это «{coinage}». Коротко, и вслух произносится с первого раза."
        if re.search(r"девиз|надпись|табличк|слоган|коротк\w* послани",q):
            for cues, line in ((r"вход|двер|прихож","Заходи, пока чай горячий."),(r"библиотек|читальн|тишин","Тише, у кого-то тут решающая глава."),
                               (r"учеб|студен|экзам|команд","Сначала один черновик. Потом будем спорить."),
                               (r"парк|лес|троп","Унеси только то, с чем пришёл."),(r"вокзал|остановк|поезд","Всем нашлось бы место, если бы рюкзаки тоже покупали билеты."),
                               (r"кофе|кафе|еда|кухн","Пять минут на чай тоже считаются планом.")):
                if re.search(cues,q):
                    return "«"+line.rstrip(".")+"»."
        if re.search(r"сравн|похож|образн",q) and re.search(r"скрип|грох|шурш|жужж|звен|капает|стуч",q):
            for sound, image in (("скрип","Как будто кто-то очень медленно открывает футляр старой скрипки."),
                                 ("грох","Как ящик с ложками, который уронили этажом выше."),
                                 ("шурш","Как страницы, которые кто-то перелистывает в полной темноте."),
                                 ("жужж","Как сердитый шмель, заблудившийся между стёклами."),
                                 ("звен","Как ложка, задевшая край тонкого стакана."),
                                 ("капает","Как часы, которые вместо секунд отсчитывают капли."),
                                 ("стуч","Как монетка, которую в кармане то и дело задевают ключи.")):
                if sound in q:
                    return image
        if re.search(r"назван|назвать|вывеск|имя для",q) and re.search(r"кафе|кофейн|магазин|оркестр|команд|групп|лавк|книжн|мастерск",q):
            for detail, name in ((r"разн|непохож|сборн|перепут","Сборная солянка"),(r"ноч|лун|поздн","После полуночи"),
                                 (r"молч|тихо|тишин","Вполголоса"),(r"вокзал|поезд|дорог","Пересадка"),
                                 (r"врем|час[ыо]|старин","Без пяти"),(r"книг|книж|чита","Между строк"),
                                 (r"потер|забыт","Бюро находок"),(r"сад|растен|цвет","Первый лист"),
                                 (r"кош|кот[ыа]|лап","Мягкие лапы"),(r"чай|кофе","Пока закипает")):
                if re.search(detail,q):
                    return "«"+name+"»."
        if re.search(r"от (?:имени|лица)|глазами|с точки зрения",q) and re.search(r"объявлен|записк|сообщен|письм|жалоб|монолог",q):
            parts = re.split(r"от (?:имени|лица)|глазами|с точки зрения",q,maxsplit=1)
            speaker = parts[-1].split(",")[0]
            if re.search(r"потер|пропа|ищет|ищу",q) and re.search(r"пар[ауые]|вторую половин",q):
                return "Разыскиваю свою вторую половину. Приметы простые: цвет тот же, а если положить нас рядом, сразу видно, что мы пара."
            if re.search(r"устал|надоел|отдых|жалоб",q):
                obj = objects_in(speaker)
                if len(obj)==1:
                    return obj[0][1]+" Мне бы хоть один выходной."
            scene = imagination.scene(re.findall("[а-я]+",speaker))
            if scene:
                return scene.voice
        return None
