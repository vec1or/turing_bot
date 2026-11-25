"""Text boundaries, short verifiable operations and one-shot abstentions."""
from decimal import Decimal, DecimalException, localcontext
import hashlib
import operator
import re
import unicodedata


QUOTES = {"«": "»", "„": "“", "“": "”", '"': '"', "'": "'", "‘": "’"}


def clean(text):
    text = " ".join(unicodedata.normalize("NFC", text).split())
    while text and text[0] in QUOTES:
        closer = QUOTES[text[0]]
        end = len(text.rstrip(".,!? "))
        if end < 2 or text[end-1] != closer or closer in text[1:end-1]:
            break
        inner = text[1:end-1].strip()
        outer_punctuation = text[end:]
        text = inner if inner.endswith(("?", "!", ".")) else inner + outer_punctuation
    return text


def focal_question(text):
    # Keep the premise in the original. Remove independent delivery instructions
    # only when a module specifically needs the clause naming the requested fact.
    clauses = re.split(r"(?<=[?!.])\s+", text)
    questions = [s for s in clauses if "?" in s]
    return questions[-1] if questions else text


def standalone(text):
    if not text or "?" in text or not re.search(r"[а-яё]", text, re.I):
        return False
    return not re.search(r"\b(потому что|если бы|это|бы)\s+\1\b|расскаж\w* (?:мне )?(?:об этом )?подробнее|(?:можешь |можете )?(?:уточни(?:ть|те)?|поясни(?:ть|те)?)\b.{0,25}(?:вопрос|что ты имеешь в виду)", text, re.I)


def requested_format(text, question):
    if not text:
        return None
    forbidden = re.findall(r"(?:без (?:слова|слов)|не использ\w* (?:слово|слова))\s+[«\"]([а-яё -]+)[»\"]", question.casefold())
    if any(re.search(r"(?<!\w)"+re.escape(word)+r"(?!\w)",text.casefold()) for word in forbidden):
        return None
    if re.search(r"\b(?:в одном предложении|одним предложением|в одну фразу|одной фразой)\b",question,re.I):
        first=re.split(r"(?<=[.!])\s+(?=[А-ЯЁ])",text,maxsplit=1)[0]
        if first.startswith("«") and not first.endswith(("»","».")):
            first=first.rstrip(".")+"»."
        return first
    return text


ARITHMETIC = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "плюс": operator.add, "минус": operator.sub, "умножить на": operator.mul,
    "разделить на": operator.truediv, "поделить на": operator.truediv,
}
EXPR = re.compile(r"(?:сколько (?:же )?(?:будет|получится|составит)?|посчитай|вычисли|чему равно|реши(?: пример)?)?\s*[:,=]?\s*"
                  r"(-?\d{1,10}(?:[.,]\d{1,6})?)\s*(плюс|минус|умножить на|разделить на|поделить на|[+*/-])\s*"
                  r"(-?\d{1,10}(?:[.,]\d{1,6})?)(?:\s*=)?[?!. ]*")
QUOTED_WORD = r'[«„“"\']([а-яё-]{1,50})[»“”"\']'


def rough(value):
    """A person estimates a large product instead of reciting every digit."""
    value = abs(float(value))
    for size, word in ((1e9, "миллиарда"), (1e6, "миллиона"), (1e3, "тысяч")):
        if value >= size:
            scaled = value / size
            shown = str(round(scaled)) if scaled >= 10 else f"{scaled:.1f}".replace(".0", "").replace(".", ",")
            return f"Без калькулятора точно не скажу. Где-то около {shown} {word}, наверное."
    return f"Где-то около {round(value)}."


def short_operation(text):
    q = text.casefold().replace("ё", "е").replace("×", "*").replace("÷", "/").replace("−", "-")
    m = EXPR.fullmatch(q)
    if m:
        try:
            with localcontext() as ctx:
                ctx.prec = 12
                value = ARITHMETIC[m[2]](Decimal(m[1].replace(",", ".")), Decimal(m[3].replace(",", ".")))
            a, b = abs(Decimal(m[1].replace(",", "."))), abs(Decimal(m[3].replace(",", ".")))
            if m[2] in ("*", "умножить на") and a * b >= 10000:
                return rough(value)
            answer = format(value,"f")
            if "." in answer:
                answer = answer.rstrip("0").rstrip(".")
            return "Получается " + answer.replace(".", ",") + "."
        except DecimalException:
            return "На ноль делить нельзя."
    # Keep ё in the actual word, even though intent matching folds it into е.
    cited = re.findall(QUOTED_WORD, text, re.I)
    if len(cited) == 1:
        word = cited[0]
        if re.search(r"задом напер[её]д|в обратном порядке|наоборот", q) and re.search(r"слов|прочи|напиши|произнес", q):
            return word[::-1] + "."
        if re.search(r"сколько (?:всего )?букв\b", q):
            return f"В слове «{word}» их {sum(c.isalpha() for c in word)}."
    return None


# Short, casual dodges. A live student also says «не знаю» sometimes, but in
# varied words and without explaining the uncertainty at length.
FALLBACKS = (
    "Хм, даже не знаю.",
    "Сложный вопрос.",
    "Без понятия.",
    "Тут я пас.",
    "Ой, не знаю, если честно.",
    "Хороший вопрос. Не знаю.",
    "Даже не думал об этом.",
    "Мм, сложно сказать.",
    "Не уверен, не хочу врать.",
    "Понятия не имею.",
    "Хз, если честно.",
    "Слушай, без понятия.",
    "Вот тут я завис.",
    "Ну не знаю я.",
    "Надо подумать, сразу не скажу.",
    "Ох, тут я не силён.",
    "Хм. Пропущу, пожалуй.",
    "Даже не знаю, что сказать.",
    "Сложно, сразу не соображу.",
    "Не, тут не подскажу.",
)
OPENINGS = ("Хм.", "Ой.", "Слушай,", "Честно,", "Ну", "Блин,", "Ммм.")
ENDINGS = ("не знаю.", "сложно сказать.", "без понятия.", "не могу сказать.", "пропущу.", "тут я пас.", "даже не думал.")


# When nothing confident is found, a person still answers in the shape of
# the question: a hedged opinion, a vague plan, a half-remembered reason.
TYPED = (
    (r"(?:спасибо|спс|пасиб\w*).{0,25}(?:выслушал|поддерж|совет|помог)",
     ("Да не за что, обращайся)", "Всегда пожалуйста. Держись там)")),
    (r"одиноко|тоскливо|никто не пишет|никому не нужн|мне (?:\w+ ){0,2}(?:плохо|грустно|хреново)|навалилось",
     ("Блин, сочувствую. Что случилось-то, рассказывай, если хочешь. Иногда просто выговориться уже легче.", "Понимаю, бывает такое. Давай просто поболтаем о чём-нибудь, отвлечёшься.")),
    (r"^(?:а |ну |ахах\w* |хаха\w* )?(?:серьезно|правда|реально|да ладно|честно)\W*$",
     ("Ну да, серьёзно)", "Ага, честно.", "Ну да, а что такого)")),
    (r"^(?:а |и )?(?:что|че) (?:было )?(?:потом|дальше)|^(?:а |и )?дальше\W*$|чем (?:\w+ )?(?:закончилось|кончилось)",
     ("Да ничего особенного, как-то само рассосалось.", "Потом всё нормально стало, без драмы.", "Ну дальше как у всех, ничего интересного.")),
    (r"^(?:а |и )?(?:давно|как давно)(?: так| это)?\W*$",
     ("Да уже давно, сам не помню с какого момента.", "Ну пару лет точно.", "Со школы, наверное.")),
    (r"^(?:ну |ладно,? )?(?:спасибо|благодарю|пока[\s)!.]*$|до свидания|было (?:интересно|приятно|весело))|\bспасибо,? (?:было|за)",
     ("Мне тоже, спасибо!", "Взаимно, было весело.", "И тебе спасибо!")),
    (r"\bчто (?:бы )?(?:ты )?(?:бы )?(?:ему |ей |им )?(?:скажешь|сказал|ответишь|ответил)|\bкак (?:бы )?(?:ты )?(?:ответишь|отреагируешь|поступишь)",
     ("Да по ситуации. Скорее всего, сказал бы спокойно и по делу, без наезда.",
      "Сказал бы прямо, что думаю, но спокойно. Обычно так лучше всего работает.",
      "Сказал бы: «Слушай, давай нормально поговорим». А дальше уже смотрел бы по реакции.")),
    (r"\b(?:стоит ли|нужно ли|можно ли|надо ли|есть ли смысл|имеет ли смысл)\b",
     ("Смотря зачем. Если есть понятная причина, то стоит, а если просто так, я бы не торопился.",
      "Я бы сначала взвесил, что теряешь и что получаешь. Однозначного ответа тут нет.",
      "Зависит от ситуации. Я бы посоветовался с кем-то, кто через это прошёл.")),
    (r"\b(?:посоветуй|подскажи|что посоветуешь|что делать|как быть|как (?:мне |лучше |правильно |быстро )?(?:научиться|перестать|начать|сделать|выбрать|познакомиться|подготовиться|справиться|избавиться|найти|бросить))",
     ("Я бы начал с малого и посмотрел, что получится. Универсального рецепта не знаю.",
      "Сам в этом не эксперт. Я бы спросил у тех, у кого уже получилось.",
      "Наверное, просто пробовать и не бросать после первой неудачи. Больше ничего умного не скажу.")),
    (r"\b(?:если бы|ты бы|представь)\b",
     ("Даже не знаю, никогда себя в такой ситуации не представлял. Наверное, сначала растерялся бы.",
      "Хм, сложно. Наверное, сначала подумал бы, а потом уже действовал.",
      "Наверное, так же, как большинство. Героем себя не вижу.")),
    (r"\b(?:как (?:ты )?думаешь|что (?:ты )?думаешь|как (?:ты )?относишься|тво[её] мнение|по-твоему|(?:ты )?считаешь)\b",
     ("Сложно сказать однозначно. Наверное, у каждого своя правда.",
      "Я об этом особо не задумывался. Скорее нейтрально отношусь.",
      "Смотря с какой стороны посмотреть. Мне не близко, но и против ничего не имею.")),
    (r"^(?:а |и |ну )?(?:почему|зачем|отчего|как так)\b.*\b(?:ты|тебя|тебе|твой|твоя|сам)\b",
     ("Ну просто так чувствую. Сложно объяснить словами.", "Не знаю, так сложилось.", "Наверное, привычка. Даже не задумывался.")),
    (r"^(?:а |и |ну )?(?:почему|зачем|отчего|как так)\b",
     ("Хороший вопрос, никогда не задумывался. Наверное, там несколько причин сразу, но точно не скажу.",
      "Знаю, что так бывает, а почему — не вспомню.",
      "Сложно сказать, я в этом не разбирался. Наверное, просто так удобнее, а потом все привыкли.")),
    (r"\b(?:ты когда-нибудь|у тебя (?:был|была|было|были)|ты (?:был|ездил|пробовал|видел|встречал))\b",
     ("Было что-то такое, но ничего интересного, рассказывать особо нечего.",
      "Не припомню такого. Может, и было, но не запомнилось.",
      "Вроде нет. По крайней мере, не помню.")),
    (r"^(?:а|и|ну|так)\b",
     ("Да по-разному, зависит от настроения.", "Ну как сказать. Всего понемногу.", "Это долгая история.",
      "Хм, дай подумать. Сложно так сразу.", "По-разному бывает.", "Тут в двух словах не расскажешь.")),
)
TYPED = tuple((re.compile(expr), replies) for expr, replies in TYPED)


class Abstentions:
    def __init__(self):
        self.seen = set()

    def answer(self, question):
        key = int.from_bytes(hashlib.blake2s(question.encode(), digest_size=4).digest(), "big")
        normalized = question.casefold().replace("ё","е")
        topical = []
        if re.search(r"\b(?:мысленно|загадал\w*|задумал\w*|у меня в (?:руке|кармане)|о чем я думаю)\b", normalized):
            topical.append("Я ж не экстрасенс) Дай хоть подсказку, тогда попробую.")
        if re.search(r"\b(?:кто|куда|где)\b", normalized) and re.search(r"\b(?:у меня|мой|моя|моей|мою|я вчера|я положил\w*)\b", normalized):
            topical.append("Не знаю, я ж этого не видел) Расскажи подробнее.")
        for expr, replies in TYPED:
            if expr.search(normalized.strip()):
                topical.extend(replies)
                break
        pool = [p for p in topical if p not in self.seen] or [p for p in FALLBACKS if p not in self.seen]
        if not pool:
            pool = [(a+" "+(b.capitalize() if a.endswith(".") else b)) for a in OPENINGS for b in ENDINGS if (a+" "+(b.capitalize() if a.endswith(".") else b)) not in self.seen]
        if pool:
            answer = pool[key % len(pool)]
        else:
            # Long-running operator sessions can exceed a normal stage round.
            answer = f"У меня уже {len(self.seen)+1} вопросов за раунд, где я сомневаюсь. Этот тоже пропущу."
        self.seen.add(answer)
        return answer
