"""Response selection over real conversational replies (ConveRT-style dual encoder).

A bidirectional Transformer trained from random initialisation maps a context
and a reply to unit vectors; a reply that fits the context has a large dot
product. Replies of the index were encoded in advance during training. At run
time only the question is encoded (NumPy), the nearest replies are found by a
matrix product, re-ranked with lexical overlap against the context each reply
originally answered, and filtered for the fictional student's consistency.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

from .language import Pieces, RESERVED

PAD, DOC, QUESTION, REPLY, END, UNKNOWN = range(len(RESERVED))
MAX_LEN = 40
WORD = re.compile(r"[а-яёa-z0-9]+", re.I)
STOP = set("и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это твой твоя тебе".split())


def _norm(x, gain, bias):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5) * gain + bias


def _gelu(x):
    return .5 * x * (1 + np.tanh(.79788456 * (x + .044715 * x ** 3)))


def stems(text):
    words = WORD.findall(unicodedata.normalize("NFC", text).casefold().replace("ё", "е"))
    return {w[:5] for w in words if w not in STOP and len(w) > 2}


def context_ids(pieces, question, previous=None):
    ids = [QUESTION] + pieces.encode(question)[:MAX_LEN - 2]
    if previous:
        prev = pieces.encode(previous)[-(MAX_LEN - len(ids) - 1):] if MAX_LEN - len(ids) > 3 else []
        ids = ([DOC] + prev if prev else []) + ids
    return ids[:MAX_LEN]


def reply_ids(pieces, reply):
    return [REPLY] + pieces.encode(reply)[:MAX_LEN - 1]


class Encoder:
    def __init__(self, config, weights):
        self.config = config
        self.w = weights
        self.pieces = Pieces(config["vocab"])
        # Contiguous transposed copies keep every product on the fast BLAS path.
        self.t = {k: np.ascontiguousarray(v.T) for k, v in weights.items() if v.ndim == 2 and "embed" not in k and "pos" not in k}

    def encode_batch(self, seqs, head):
        """Encode several token sequences at once; padding is masked out."""
        w, t, c = self.w, self.t, self.config
        b, n = len(seqs), max(map(len, seqs))
        ids = np.zeros((b, n), dtype=np.int64)
        keep = np.zeros((b, n), dtype=bool)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            keep[i, :len(s)] = True
        x = (w["embed.weight"][ids] + w["pos.weight"][:n]).reshape(b * n, -1)
        heads = c["heads"]
        d = x.shape[-1]
        bias = np.where(keep, 0., -1e9).astype(np.float32)[:, None, None, :]
        for layer in range(c["layers"]):
            p = f"blocks.{layer}."
            z = _norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            q, k, v = np.split(z @ t[p + "qkv.weight"], 3, axis=-1)
            q, k, v = (np.ascontiguousarray(a.reshape(b, n, heads, -1).transpose(0, 2, 1, 3)) for a in (q, k, v))
            a = np.matmul(q, k.transpose(0, 1, 3, 2)) * (1 / math.sqrt(q.shape[-1])) + bias
            a = np.exp(a - a.max(axis=-1, keepdims=True))
            a /= a.sum(axis=-1, keepdims=True)
            h = np.matmul(a, v).transpose(0, 2, 1, 3).reshape(b * n, d)
            x = x + h @ t[p + "proj.weight"]
            z = _norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            x = x + _gelu(z @ t[p + "up.weight"] + w[p + "up.bias"]) @ t[p + "down.weight"] + w[p + "down.bias"]
        x = _norm(x, w["norm.weight"], w["norm.bias"]).reshape(b, n, d)
        x = (x * keep[..., None]).sum(1) / keep.sum(1, keepdims=True)
        y = _gelu(x @ t[head + ".0.weight"] + w[head + ".0.bias"]) @ t[head + ".2.weight"] + w[head + ".2.bias"]
        return (y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)

    def encode(self, ids, head):
        return self.encode_batch([ids], head)[0]

    def contexts(self, texts):
        return self.encode_batch([context_ids(self.pieces, t) for t in texts], "ctx")

    def replies(self, texts):
        return self.encode_batch([reply_ids(self.pieces, t) for t in texts], "rsp")

    def context(self, question, previous=None):
        return self.encode(context_ids(self.pieces, question, previous), "ctx")

    def reply(self, text):
        return self.encode(reply_ids(self.pieces, text), "rsp")



# The fictional student is a young man; replies that state other biographical
# facts (age, job, family, gender) are left to profile.py.
FEMININE = re.compile(r"\bя(?:\s+[а-яё]+){0,2}\s+[а-яё]+(?:ла|лась)\b|\b(?:устала|рада|сама|довольна|счастлива|согласна|свободна|занята|влюблена|беременна)\b|\b(?:я|сама)\s+(?:рада|готова|должна|уверена|согласна|замужем|беременна|сама|одна|влюблена|занята|больна|голодна|счастлива|устала|довольна)\b|\bсама\b", re.I)
BIOGRAPHY = re.compile(r"\bмне\s+\d+|\bмне\s+(?:уже\s+)?(?:двадцать|тридцать|сорок|пятьдесят|шестнадцать|семнадцать|восемнадцать)\b|\b(?:моя жена|мой муж|моя дочь|мой сын|мои дети|у меня (?:есть )?(?:дети|сын|дочь|жена|муж|внук\w*))\b|\bя (?:женат|замужем|работаю|на пенсии|пенсионер\w*|школьни\w*|в \d+ классе|из \w+|живу в)\b|\bменя зовут\b|\b(?:с|у|моим|мой|моего|моему) (?:парн\w*|муж\w*)|\bмой (?:парень|муж)\b|\bмужу\b|\bв меня влюблен\w*|\bродил\w* в\b|\bживу в\b|\bвырос\w* в\b|\b(?:немец|немка|русский|русская|казах|казашка|татарин|украинец|еврей|армянин|грузин|узбек)\b|\bдевочк\w* в родах|\bв родах\b|частн\w* (?:дом|доме)\b|по диплому|хотим сказать|мы (?:хотим|советуем|рекомендуем)|(?:я|когда) одна\b|хочу (?:еще|ещё) больше|я (?:теперь |уже )?учител|\bобл\b|\bбратск|\bиркутск|интернатур|ординатур|\bвсеравно\b|\b(?:первый|второй|третий|четв[её]ртый|пятый|последний) (?:год|курс)\b|\bна (?:первом|втором|третьем|четв[её]ртом|пятом) курсе", re.I)
SENSITIVE = re.compile(r"(?:мраз|ублюд|дебил|тупая|тупой|эфиоп|негр|негри|чурк|хохл|китаез|китаёз|жид|голы[еймх]|голая|голым|ходют|тожа|кофей\b|ихн|евон|обам|абам|трамп|президент|пина[тюе]|пнул|пнуть|избил|избить|\bбью\b|\bбить\b|ударю|ударить|ху[йеёяию]|пизд|\bеб[аунл]|\bёб|бля|\bсс[уы]\b|хрен|жоп|говн|сра[лтн]|насра|муда|пидор|гандон|шлюх|залуп|дроч|херн|нахер|долбо|дебил|идиот|урод|\bчмо|\bлох|кринж|мастурб|секс|сексу|постел|интим|эрот|порн|трах|возбужд|голая|голый|голые|оргазм|презерв|залет|беремен|минет|член\b|грудь|сиськ|попк|задниц|бухл|бухать|водк|пиво|пива|ганж|травк|наркот|курит|сигарет|убит|убью|суицид|повес|путин|навальн|украин|войн|нацист|религ|бог\b|аллах|церк|мусульм|еврей|чурк|хач|негр)", re.I)
REJECT = re.compile(r"\b(?:бот\w*|нейросет\w*|робот\w*|искусственн\w*|админ\w*|канал\w*|чат\w*|подпис\w*|ссылк\w*|телеграм\w*|ответ(?:ах|ы) (?:мейл|майл)\w*|на ответах|мейл\w*|майл\w*|сайт\w*|форум\w*|модератор\w*|баллы|баллов|лучший ответ|вопрос(?:ов)? (?:выше|ниже)|автор\w* вопроса|аск\w*|аве|аву|аватар\w*|ава|аватарк\w*)\b|https?:|@|#|\d{5,}|[а-яё]{19,}", re.I)


FEM_ADJ = re.compile(r"\b(?:я\s+(?:такая\s+|очень\s+|сама\s+)?[а-яё]{3,}ая|[а-яё]{3,}ая\s+я)\b", re.I)
FEM_PAST = re.compile(r"\b[а-яё]{1,}(?:ала|ила|ыла|ела|яла|ола|ула|шла|зла|сла)(?:сь)?\b", re.I)
THIRD_FEM = re.compile(r"\b(?:она|ее|её|мама|мать|жена|подруга|девушка|сестра|дочь|дочка|бабушка|кошка|собака|соседка|учительница|жизнь|погода|зима|весна|осень|машина|работа|страна|вода|она-то)\b", re.I)


STYLE = re.compile(r"[a-z]|:\)|:\(|:d|\)\)|\(\(|\b(?:вы|вас|вам|ваш\w*|вами)\b|\b(?:мну|шо|чо|ща|щас|нафиг|фиг)\b|^\w+(?:йте|ите)\b|\bбаку\b|\b(?:пейте|пишите|звоните|обращайтесь|смотрите|читайте|попробуйте|делайте|скажите|приходите|заходите)\b", re.I)


def persona_safe(reply):
    if STYLE.search(reply):
        return False
    if re.search(r"\b(?:лосин\w*|юбк\w*|платье\w*|платья|колготк\w*|каблук\w*|туфельк\w*|сумочк\w*|макияж\w*|помад\w*|ресниц\w*|маникюр\w*|бюстгальтер\w*|лифчик\w*)\b", reply, re.I):
        return False
    last = re.findall(r"[а-яё]+", reply.casefold())
    if last and last[-1] in {"почти", "очень", "самый", "самая", "такой", "такая", "этот", "эти", "мой", "моя", "свой", "его", "её", "их", "уже", "еще", "ещё", "тоже", "если", "когда", "чтобы", "потому", "который", "до", "в", "на", "с", "к", "по", "за", "из", "от", "и", "а", "но", "что", "как", "у", "о", "для", "без"}:
        return False
    if re.search(r"€|\$|\bевро\b|\bдоллар\w*|\bгермани\w*|\bевроп\w*|\d+\s*(?:р|руб\w*)\b|\bрубл\w*|\bтрамва\w*|\bметро\b|\bэлектричк\w*", reply, re.I):
        return False
    if "©" in reply or re.search(r"\b(?:профил\w* кбс|скидк\w*|акци\w*|звоните|заказывайте)\b", reply, re.I):
        return False
    if re.search(r"\bне печата\w*|\bне человек\b|\bя (?:же )?(?:программа|машина|алгоритм|компьютер|робот|бот|нейросеть|ии)\b|\bросси\w*|воронеж|меня создал|мой создател|создатель|меня обучал|обучили меня|меня запрограммир|мои разработчик|\bдеды\b|армии|армейск|казарм|перловк|хабаровск|москв|питер|петербург|екатеринбург|новосибирск|казан[ьи]|самар|киев|минск|командировк|каджит|\bбывш\w*|\bруб\b|\bруб\.|\bрубл\w*|\bначальник\w*|слышал,? как храп|тротуар|пешеход|геотехник|в корею|в корее|тридцать четыре года|мне тож не ответил|эспрессо, бразилия|каннибал|я игры не очень|две недели|авиатехник|учусь учиться|мне всегда восемнадцать|на один год меньше|ничего не сделал|загорожена луной|в общаге один живу|так я не в алматы|маск[уа]|какие планеты находятся|младше на 4 года|я сестра|я пока мало что умею|вите иташке|день нормально прошел а вечером|а это не море|аэропорт|илон иванов|на юриста|как назвали так и зовут|не все многое|избранный|сутки через трое|не нравится вообще все|83, 5%|в алмате|в сутках часа|семенник|разбираюсь в людях|меня \w+ зовут|\bсоседки\b|риссую|много имен|не совсем не глуп|меня учат|друзей нет|нет друзей|работал в заливе|мужиков-то|на всех планетах|донер а не|\bроссиян\w*", reply, re.I):
        return False
    if re.search(r"\bидеальн(?:а|ая)\b|\bу меня нет ошибок", reply, re.I):
        return False
    if re.search(r"\bучительниц\w*|\bдет(?:ей|и|ям|ьми)\b|\bвнук\w*|\bпенси\w*", reply, re.I):
        return False
    if re.search(r"через двор|\bродител\w* .{0,20}живут|\bживут (?:рядом|недалеко|через)", reply, re.I):
        return False
    # Places, relatives and age belong to the fixed biography in profile.py.
    if re.search(r"^(?:я )?из [а-яё]+|\bобласт\w*|\b(?:есть|у меня) (?:\w+ )?(?:брат\w*|сестр\w*|дети|ребен\w*|ребён\w*)|\bлетн(?:ий|яя)\b|\bмне (?:уже )?(?:за|под) ", reply, re.I):
        return False
    if reply.count("(") != reply.count(")") and not reply.rstrip().endswith(")"):
        return False
    if re.search(r"[а-яё][.,!][а-яё]|\b(?:класс\w*|колледж\w*|техникум\w*|училищ\w*|учусь на|курс по специальност\w*|в школе учусь)\b", reply, re.I):
        return False
    if FEM_ADJ.search(reply) or re.search(r"\b(?:рост|вес)\s+\d+", reply, re.I):
        return False
    if FEMININE.search(reply) or BIOGRAPHY.search(reply) or REJECT.search(reply) or "?" in reply:
        return False
    # A feminine past-tense verb without a feminine subject usually speaks of the speaker.
    return not (FEM_PAST.search(reply) and not THIRD_FEM.search(reply))


def tidy(reply):
    """One consistent surface: sentence-initial capitals, no runs of dots."""
    reply = re.sub(r"\.{2,}", ".", reply).strip()
    reply = re.sub(r"(^|[.!]\s+)([а-яё])", lambda m: m[1] + m[2].upper(), reply)
    return reply


class Retrieval:
    """Nearest stored questions (PCA-reduced int8 vectors), then re-ranking.

    The index holds the context vectors of the stored question/answer pairs.
    A query is projected with the same PCA, the closest stored questions are
    found by a matrix product, and the top candidates are re-scored with the
    full encoder: question-to-question similarity, question-to-reply fit and
    lexical overlap.
    """

    def __init__(self, encoder, replies, contexts, vectors, scales, mean, components):
        self.encoder = encoder
        self.replies = replies
        self.contexts = contexts
        self.vectors = vectors
        self.scales = scales
        self.mean = mean
        self.components = components

    @classmethod
    def load(cls, directory=None):
        root = Path(directory) if directory else Path(__file__).with_name("assets")
        folder = root / "retrieval"
        config = json.loads((folder / "encoder.json").read_text(encoding="utf-8"))
        with np.load(folder / "encoder.npz", allow_pickle=False) as cache:
            weights = {k: np.asarray(cache[k], dtype=np.float32) for k in cache.files}
        replies, contexts = [], []
        for part in sorted(folder.glob("replies*.jsonl.gz")):
            with gzip.open(part, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    replies.append(row["r"])
                    contexts.append(row["c"])
        vectors = np.concatenate([np.load(p) for p in sorted(folder.glob("contexts-*.npy"))])
        scales = np.load(folder / "scales.npy").astype(np.float32)
        with np.load(folder / "pca.npz") as pca:
            mean, components = pca["mean"].astype(np.float32), pca["components"].astype(np.float32)
        if not len(replies) == len(vectors) == len(scales):
            raise ValueError("Индекс ответов повреждён: разное число строк и векторов.")
        return cls(Encoder(config, weights), replies, contexts, vectors, scales, mean, components)

    def search(self, vector, k=64):
        q = (vector - self.mean) @ self.components.T
        q /= np.linalg.norm(q) + 1e-8
        scores = np.empty(len(self.vectors), dtype=np.float32)
        step = 131072
        for start in range(0, len(self.vectors), step):
            scores[start:start + step] = self.vectors[start:start + step].astype(np.float32) @ q
        scores *= self.scales
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return top[np.argsort(-scores[top])]

    def propose(self, question, previous=None, seen=(), k=64):
        """Return (reply, score, details) or None. Scores are not probabilities."""
        q_vec = self.encoder.context(question, previous)
        found = [int(i) for i in self.search(self.encoder.context(question), k)
                 if self.replies[i].casefold() not in seen and persona_safe(self.replies[i])
                 and not SENSITIVE.search(self.replies[i])
                 and not re.match(r"\s*[.,;:!)\-]", self.replies[i])
                 and not re.search(r"мой лучший друг\s*-?\s*это я|не было возможности съездить", self.replies[i], re.I)]
        if not found:
            return None
        same = self.encoder.contexts([self.contexts[i] for i in found]) @ self.encoder.context(question)
        dense = self.encoder.replies([self.replies[i] for i in found]) @ q_vec
        asked = stems(question)
        best = None
        for j, i in enumerate(found):
            reply, context = self.replies[i], self.contexts[i]
            ctx = stems(context)
            overlap = len(asked & ctx) / math.sqrt(max(1, len(asked)) * max(1, len(ctx))) if asked and ctx else 0.
            # Echoing the question's own words in a reply is a weak, bot-like move.
            echo = len(asked & stems(reply)) / max(1, len(asked))
            length = len(reply)
            shape = -.04 if length < 8 else (-.03 if length > 200 else 0.)
            score = .5 * float(dense[j]) + .5 * float(same[j]) + .1 * overlap - .05 * echo + shape
            if best is None or score > best[1]:
                best = (tidy(reply), score,
                        dict(dense=float(dense[j]), same=float(same[j]), overlap=overlap, context=context, index=i))
        return best
