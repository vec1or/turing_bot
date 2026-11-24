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


def short_operation(text):
    q = text.casefold().replace("ё", "е").replace("×", "*").replace("÷", "/").replace("−", "-")
    m = EXPR.fullmatch(q)
    if m:
        try:
            with localcontext() as ctx:
                ctx.prec = 12
                value = ARITHMETIC[m[2]](Decimal(m[1].replace(",", ".")), Decimal(m[3].replace(",", ".")))
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


FALLBACKS = (
    "Не знаю. У меня даже приличной догадки сходу нет.",
    "Вот об этом я ещё ни разу не задумывался.",
    "Я бы сейчас ответил наугад. Пожалуй, честнее сказать, что не знаю.",
    "Хорошего ответа у меня пока нет.",
    "Тут я пас. В голове вертится пара догадок, и обе сомнительные.",
    "Честно, я в этом плаваю.",
    "Пока не берусь судить. Слишком легко уверенно сказать ерунду.",
    "Не уверен. Не хочу говорить ерунду.",
    "С ходу ничего толкового не придумал.",
    "У меня на этот счёт пока пусто.",
    "Хм, без понятия, если честно.",
    "Думал, что отвечу сразу, а зацепиться не за что.",
    "Я тут скорее послушал бы знающего человека.",
    "Хм. Тут я, наверное, промолчу.",
    "Сложно сказать. Никогда об этом всерьёз не думал.",
    "Навскидку сказать трудно. Уверенности у меня тут никакой.",
    "Боюсь соврать. Пропущу этот, пожалуй.",
    "Вот тут мне бы пригодилась шпаргалка.",
    "Даже не знаю, что сказать. Пропущу, наверное.",
    "Пожалуй, это тот случай, когда я пожму плечами.",
)
OPENINGS = ("Я задумался.", "С наскока не вышло.", "Тут я замялся.", "Сижу и перебираю варианты.",
            "Как-то я растерялся.", "Минуту назад я об этом даже не думал.", "Неожиданный поворот.")
ENDINGS = ("Внятной версии у меня пока нет.", "Мне самому было бы интересно разобраться.",
           "Боюсь перепутать детали.", "Догадки слишком уж разные.", "При случае поищу ответ.",
           "Своей первой догадке я тут не доверяю.", "Я бы сначала проверил, как всё устроено.")


class Abstentions:
    def __init__(self):
        self.seen = set()

    def answer(self, question):
        key = int.from_bytes(hashlib.blake2s(question.encode(), digest_size=4).digest(), "big")
        normalized = question.casefold().replace("ё","е")
        topical = []
        if re.search(r"\b(?:мысленно|загадал\w*|задумал\w*|у меня в (?:руке|кармане)|о чем я думаю)\b", normalized):
            topical.append("Чужие мысли мне недоступны. Без подсказок любое совпадение было бы случайностью.")
        if re.search(r"\b(?:кто|куда|где)\b", normalized) and re.search(r"\b(?:у меня|мой|моя|моей|мою|я вчера|я положил\w*)\b", normalized):
            topical.append("Я этого не видел. По твоему описанию уверенно восстановить, что произошло, у меня не выходит.")
        pool = [p for p in topical if p not in self.seen] or [p for p in FALLBACKS if p not in self.seen]
        if not pool:
            pool = [a+" "+b for a in OPENINGS for b in ENDINGS if a+" "+b not in self.seen]
        if pool:
            answer = pool[key % len(pool)]
        else:
            # Long-running operator sessions can exceed a normal stage round.
            answer = f"У меня уже {len(self.seen)+1} вопросов за раунд, где я сомневаюсь. Этот тоже пропущу."
        self.seen.add(answer)
        return answer
