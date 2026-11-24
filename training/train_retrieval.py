"""Train the dual encoder of turing_bot/retrieval.py from random initialisation.

Method: ConveRT-style response selection (Henderson et al., 2020). A shared
Transformer encodes a context and a reply; two small heads map them to unit
vectors. Each batch of B pairs is a B-way classification: the true reply must
score higher than the other B-1 replies of the batch (in-batch negatives).

Input: JSONL from training/prepare_dialogues.py with fields c (context),
p (optional previous turn), r (reply). Output: encoder.json + encoder.npz for
NumPy inference, plus a report with validation recall@1 among 100 candidates.

    python -m training.train_retrieval --pairs pairs.jsonl --valid valid.jsonl \
        --out turing_bot/assets/retrieval --steps 40000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from turing_bot.language import Pieces
from turing_bot.retrieval import MAX_LEN, context_ids, reply_ids


class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, ff)
        self.down = nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (a.view(b, n, self.heads, -1).transpose(1, 2) for a in (q, k, v))
        mask = ~pad[:, None, None, :]
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop.p if self.training else 0.)
        x = x + self.drop(self.proj(h.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.down(F.gelu(self.up(self.ln2(x)), approximate="tanh")))


class DualEncoder(nn.Module):
    def __init__(self, vocab, d=256, layers=3, heads=4, ff=768, out=256, dropout=.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.ctx = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, out))
        self.rsp = nn.Sequential(nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, out))
        self.scale = nn.Parameter(torch.tensor(math.log(20.)))
        nn.init.normal_(self.embed.weight, std=.02)
        nn.init.normal_(self.pos.weight, std=.02)

    def encode(self, ids, head):
        pad = ids == 0
        x = self.embed(ids) + self.pos.weight[:ids.shape[1]]
        for block in self.blocks:
            x = block(x, pad)
        x = self.norm(x)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        return F.normalize(getattr(self, head)(pooled), dim=-1)


def batchify(seqs, device):
    n = max(map(len, seqs))
    out = torch.zeros(len(seqs), n, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(np.asarray(s, dtype=np.int64))
    return out.to(device, non_blocking=True)


class Pairs:
    """Token ids of many pairs in two flat int16 arrays (Python lists would not fit in 7 GB)."""
    def __init__(self, path, pieces, limit=None):
        from array import array
        flat = {"c": array("h"), "r": array("h")}
        ends = {"c": array("q"), "r": array("q")}
        with open(path, encoding="utf-8") as stream:
            for n, line in enumerate(stream, 1):
                row = json.loads(line)
                for key, ids in (("c", context_ids(pieces, row["c"], row.get("p"))), ("r", reply_ids(pieces, row["r"]))):
                    flat[key].extend(ids)
                    ends[key].append(len(flat[key]))
                if limit and n >= limit:
                    break
        self.flat = {k: np.frombuffer(v, dtype=np.int16).copy() for k, v in flat.items()}
        self.ends = {k: np.frombuffer(v, dtype=np.int64).copy() for k, v in ends.items()}

    def __len__(self):
        return len(self.ends["c"])

    def get(self, key, i):
        start = self.ends[key][i - 1] if i else 0
        return self.flat[key][start:self.ends[key][i]].astype(np.int64)

    def __getitem__(self, i):
        return self.get("c", i), self.get("r", i)


def load_pairs(path, pieces, limit=None):
    return Pairs(path, pieces, limit)


@torch.no_grad()
def recall(model, valid, device, pool=100):
    model.eval()
    hits, total = 0, 0
    for start in range(0, len(valid) - pool + 1, pool):
        chunk = [valid[i] for i in range(start, start + pool)]
        c = model.encode(batchify([a for a, _ in chunk], device), "ctx")
        r = model.encode(batchify([b for _, b in chunk], device), "rsp")
        hits += (c @ r.T).argmax(1).eq(torch.arange(len(chunk), device=device)).sum().item()
        total += len(chunk)
    model.train()
    return hits / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--vocab", help="encoder.json of an earlier run or a JSON list of pieces")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--init", help="checkpoint.pt to continue from (use with --vocab of the same run)")
    ap.add_argument("--warmup", type=int, default=1000)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.vocab:
        raw = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
        pieces = Pieces(raw["vocab"] if isinstance(raw, dict) else raw)
    else:
        sample = []
        with open(args.pairs, encoding="utf-8") as stream:
            for i, line in enumerate(stream):
                if i % 4 == 0:
                    row = json.loads(line)
                    sample += [row["c"], row["r"]]
                if len(sample) >= 600000:
                    break
        pieces = Pieces.learn(sample, size=args.vocab_size)
    print("vocab", len(pieces.tokens), flush=True)

    train = load_pairs(args.pairs, pieces, args.limit)
    valid = load_pairs(args.valid, pieces, 5000)
    print("pairs", len(train), "valid", len(valid), flush=True)

    model = DualEncoder(len(pieces.tokens), d=args.dim, layers=args.layers, ff=3 * args.dim, out=args.dim).to(args.device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=args.device, weights_only=True))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n and "pos" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "embed" not in n and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": .01}, {"params": other, "weight_decay": 0.}], lr=args.lr, betas=(.9, .98))
    warm = args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * .5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    scaler = torch.amp.GradScaler(enabled=args.device == "cuda")
    log = []
    order = list(range(len(train)))
    random.shuffle(order)
    cursor, started = 0, time.time()
    for step in range(1, args.steps + 1):
        if cursor + args.batch > len(order):
            random.shuffle(order)
            cursor = 0
        batch = [train[i] for i in order[cursor:cursor + args.batch]]
        cursor += args.batch
        c_ids = batchify([a for a, _ in batch], args.device)
        r_ids = batchify([b for _, b in batch], args.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device == "cuda"):
            c = model.encode(c_ids, "ctx")
            r = model.encode(r_ids, "rsp")
            logits = (c @ r.T).float() * model.scale.exp().clamp(max=50)
            target = torch.arange(len(batch), device=args.device)
            loss = (F.cross_entropy(logits, target, label_smoothing=.05) + F.cross_entropy(logits.T, target, label_smoothing=.05)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            entry = dict(step=step, loss=round(loss.item(), 4), seconds=round(time.time() - started))
            if step % 2500 == 0 or step == args.steps:
                entry["recall@1_of_100"] = round(recall(model, valid, args.device), 4)
            log.append(entry)
            print(entry, flush=True)

    state = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items() if k != "scale"}
    np.savez_compressed(out / "encoder.npz", **state)
    config = dict(vocab=pieces.tokens, layers=args.layers, heads=4, dim=args.dim, max_len=MAX_LEN,
                  method="ConveRT-style dual encoder, random init, in-batch negatives")
    (out / "encoder.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    report = dict(args=vars(args), train_pairs=len(train), valid_pairs=len(valid), log=log)
    (out / "training-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(model.state_dict(), out / "checkpoint.pt")


if __name__ == "__main__":
    main()
