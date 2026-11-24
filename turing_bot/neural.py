"""CPU inference for a locally trained conditional word-level GRU."""
from __future__ import annotations

import json
import math
import re
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

import numpy as np


PAD, BOS, EOS, UNK = range(4)
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
TOKEN = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*|\d+|[.,!?;:«»()—–-]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFC", text).casefold())


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:»)])", r"\1", text)
    text = re.sub(r"([«(])\s+", r"\1", text).strip(" ,;:")
    if text:
        text = re.sub(r"(^|[.!?]\s+)([а-яёa-z])", lambda m: m[1] + m[2].upper(), text)
        if text[-1] not in ".!?»":
            text += "."
    return unicodedata.normalize("NFC", text)


def context_features(text: str, buckets: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Hash local lexical/morphological cues. The most recent turn gets full weight."""
    ids, weights = [], []
    lines = [s for s in text.splitlines() if s.strip()][-3:]
    for age, line in enumerate(reversed(lines)):
        words = [w.replace("ё", "е") for w in tokenize(line) if w[0].isalnum()][-48:]
        features: Counter[str] = Counter()
        for i, word in enumerate(words):
            features["w:" + word] += 2
            features["s:" + word[:5]] += 1
            if i:
                features["b:" + words[i - 1][:5] + "/" + word[:5]] += 1
            padded = "_" + word + "_"
            for j in range(len(padded) - 2):
                features["c:" + padded[j:j + 3]] += 1
        weight = (1.0, .20, .12)[age]
        for key, count in features.items():
            if len(ids) == width:
                break
            ids.append(1 + zlib.crc32(key.encode("utf-8")) % (buckets - 1))
            weights.append(weight * min(count, 3))
    norm = math.sqrt(sum(w * w for w in weights)) or 1.0
    packed = np.zeros(width, dtype=np.int32)
    values = np.zeros(width, dtype=np.float32)
    packed[:len(ids)] = ids
    values[:len(weights)] = np.asarray(weights, dtype=np.float32) / norm
    return packed, values


def _linear(x, weight):
    # A small matrix-vector multiplication. Avoid creating native BLAS worker
    # pools for each token on laptops with many logical cores.
    return np.einsum("bi,ji->bj", x, weight, optimize=False)


class Generator:
    def __init__(self, config: dict, weights: dict):
        self.config = config
        self.tokens = config["tokens"]
        if self.tokens[:4] != SPECIAL or config.get("format") != "arena-context-gru-v1":
            raise ValueError("Несовместимый формат локальной модели.")
        if any(not isinstance(s, str) or not s for s in self.tokens) or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Повреждён словарь генератора.")
        self.ids = {s: i for i, s in enumerate(self.tokens)}
        self.noninitial = [i for i, s in enumerate(self.tokens[4:], 4) if not s[0].isalnum()]
        h, e, v = config["hidden"], config["embedding"], len(self.tokens)
        shapes = {"src.weight": (config["buckets"], e), "ctx.weight": (h, e), "ctx.bias": (h,),
                  "emb.weight": (v, e), "gru.weight_ih_l0": (3*h, e+h), "gru.weight_hh_l0": (3*h, h),
                  "gru.bias_ih_l0": (3*h,), "gru.bias_hh_l0": (3*h,),
                  "out.weight": (v, h), "out.bias": (v,), "topic.weight": (v, h)}
        for name, shape in shapes.items():
            if name not in weights or weights[name].shape != shape or not np.isfinite(weights[name]).all():
                raise ValueError(f"Повреждены параметры генератора: {name}.")
        self.weights = weights

    @classmethod
    def load(cls, directory=None):
        path = Path(directory) if directory is not None else Path(__file__).with_name("assets")
        cfg = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        with np.load(path / "model.npz", allow_pickle=False) as archive:
            weights = {k: np.asarray(archive[k], dtype=np.float32) for k in archive.files}
        return cls(cfg, weights)

    def encode(self, context):
        w = self.weights
        ids, values = context_features(context, self.config["buckets"], self.config["width"])
        summed = (w["src.weight"][ids] * values[:, None]).sum(axis=0, keepdims=True)
        return np.tanh(_linear(summed, w["ctx.weight"]) + w["ctx.bias"])

    def step(self, prev_tokens, hidden, ctx):
        w = self.weights
        emb = w["emb.weight"][prev_tokens]
        inputs = np.concatenate((emb, np.broadcast_to(ctx, hidden.shape)), axis=1)
        ir, iz, inn = np.split(_linear(inputs, w["gru.weight_ih_l0"]) + w["gru.bias_ih_l0"], 3, axis=1)
        hr, hz, hn = np.split(_linear(hidden, w["gru.weight_hh_l0"]) + w["gru.bias_hh_l0"], 3, axis=1)
        reset = 1 / (1 + np.exp(-np.clip(ir + hr, -50, 50)))
        update = 1 / (1 + np.exp(-np.clip(iz + hz, -50, 50)))
        candidate = np.tanh(inn + reset * hn)
        hidden = candidate + update * (hidden - candidate)
        logits = _linear(hidden, w["out.weight"]) + w["out.bias"] + _linear(ctx, w["topic.weight"])
        return hidden, logits

    def score(self, context: str, texts: list[str]) -> list[float]:
        """Mean conditional log probability including EOS, batched by token position."""
        if not texts:
            return []
        if len(texts) > 64:
            raise ValueError("За один вызов можно оценить до 64 вариантов.")
        limit = self.config.get("response_limit", 40)
        rows = [[self.ids.get(t, UNK) for t in tokenize(s)[:limit]] + [EOS] for s in texts]
        ctx = self.encode(context)
        hidden = np.repeat(ctx, len(rows), axis=0)
        prev = np.full(len(rows), BOS, dtype=np.int64)
        values = np.zeros(len(rows), dtype=np.float64)
        for i in range(max(map(len, rows))):
            hidden, logits = self.step(prev, hidden, ctx)
            logits -= logits.max(axis=1, keepdims=True)
            logz = np.log(np.exp(logits).sum(axis=1))
            for j, row in enumerate(rows):
                if i < len(row):
                    prev[j] = row[i]
                    values[j] += logits[j, row[i]] - logz[j]
        return [float(v / len(row)) for v, row in zip(values, rows)]

    def generate(self, context: str, *, max_tokens: int = 40) -> str:
        cfg = self.config
        beam = cfg.get("beam", 4)
        ctx = self.encode(context)
        alive = [([], 0.0)]
        hidden = ctx.copy()
        scale = cfg.get("guidance", 0.0)
        background = np.zeros_like(ctx)
        generic_hidden = background.copy()
        ended = []
        limit = max(1, min(max_tokens, cfg.get("response_limit", 40)))
        for _ in range(limit):
            last = [ids[-1] if ids else BOS for ids, _ in alive]
            new_hidden, logits = self.step(last, hidden, ctx)
            if scale:
                generic_next, generic_logits = self.step(last, generic_hidden, background)
                logits += scale * (logits - generic_logits)
            logits -= logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            logp[:, [PAD, BOS, UNK]] = -np.inf
            nodes = []
            for row, (seq, score) in enumerate(alive):
                if len(seq) < 2:
                    logp[row, EOS] = -np.inf
                if not seq:
                    logp[row, self.noninitial] = -np.inf
                if len(seq) >= 2:
                    for i in range(len(seq) - 2):
                        if seq[i:i + 2] == seq[-2:]:
                            logp[row, seq[i + 2]] = -np.inf
                size = min(beam + 1, logp.shape[1])
                top = np.argsort(-logp[row], kind="stable")[:size]
                for token in top:
                    token = int(token)
                    value = score + float(logp[row, token])
                    if token == EOS:
                        if seq:
                            ended.append((value / ((len(seq) + 1) ** .8), seq))
                    elif np.isfinite(value):
                        nodes.append((value, seq + [token], row))
            if not nodes:
                break
            nodes.sort(key=lambda node: -node[0])
            best = nodes[:beam]
            alive = [(seq, score) for score, seq, _ in best]
            hidden = new_hidden[[row for _, _, row in best]]
            if scale:
                generic_hidden = generic_next[[row for _, _, row in best]]
            # Future log probabilities cannot be positive; use the most
            # generous remaining length as an upper bound on any live beam.
            if ended and max(x[0] for x in ended) >= best[0][0] / ((limit + 1) ** .8):
                break
        if ended:
            seq = max(ended, key=lambda x: x[0])[1]
        else:
            seq = alive[0][0]
        return detokenize([self.tokens[i] for i in seq])


def main():
    """Inspect the neural decoder on a question without the second language model."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ответ только нейросети, без совместного ранжирования")
    parser.add_argument("question")
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    print(Generator.load(args.assets).generate(args.question))


if __name__ == "__main__":
    main()
