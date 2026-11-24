"""Train every parameter of the conditional GRU from random initialization."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from turing_bot.neural import BOS, EOS, PAD, UNK, SPECIAL, Generator, context_features, tokenize


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, hidden, n = cfg["embedding"], cfg["hidden"], len(cfg["tokens"])
        self.src = nn.Embedding(cfg["buckets"], dim, padding_idx=0)
        self.ctx = nn.Linear(dim, hidden)
        self.emb = nn.Embedding(n, dim, padding_idx=PAD)
        self.gru = nn.GRU(dim + hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n)
        self.topic = nn.Linear(hidden, n, bias=False)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.context_dropout = cfg.get("context_dropout", 0.)

    def encode(self, ids, weights):
        return self.ctx((self.src(ids) * weights.unsqueeze(-1)).sum(1)).tanh()

    def forward(self, ids, weights, previous):
        ctx = self.encode(ids, weights)
        if self.training and self.context_dropout:
            keep = torch.rand((len(ctx), 1), device=ctx.device) >= self.context_dropout
            ctx = ctx * keep
        emb = self.dropout(self.emb(previous))
        inputs = torch.cat((emb, ctx.unsqueeze(1).expand(-1, previous.shape[1], -1)), dim=2)
        hidden, _ = self.gru(inputs, ctx.unsqueeze(0))
        return self.out(self.dropout(hidden)) + self.topic(ctx).unsqueeze(1)


def read(path, limit):
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            target = tokenize(row["response"])
            if row["context"].strip() and 1 <= len(target) < limit:
                result.append((row["context"], row["response"], target, row.get("source", "")))
    if not result:
        raise ValueError(f"Нет подходящих пар в {path}")
    return result


def encode_rows(rows, cfg):
    lookup = {t: i for i, t in enumerate(cfg["tokens"])}
    ids = np.zeros((len(rows), cfg["width"]), np.int32)
    weights = np.zeros(ids.shape, np.float32)
    targets = np.full((len(rows), max(len(row[2]) for row in rows) + 2), PAD, np.int64)
    lengths = []
    for i, (ctx, _, target, _) in enumerate(rows):
        ids[i], weights[i] = context_features(ctx, cfg["buckets"], cfg["width"])
        sequence = [BOS] + [lookup.get(w, UNK) for w in target] + [EOS]
        targets[i, :len(sequence)] = sequence
        lengths.append(len(sequence))
    return torch.from_numpy(ids), torch.from_numpy(weights), torch.from_numpy(targets), np.asarray(lengths)


def minibatch(dataset, idx, device):
    ids, weights, target, lengths = dataset
    width = int(max(lengths[idx]))
    return ids[idx].to(device), weights[idx].to(device), target[idx, :width].to(device)


@torch.no_grad()
def validation_loss(network, data, device, batch):
    network.eval()
    loss, tokens = 0.0, 0
    for start in range(0, len(data[0]), batch):
        idx = np.arange(start, min(start + batch, len(data[0])))
        ids, weights, target = minibatch(data, idx, device)
        logits = network(ids, weights, target[:, :-1])
        loss += float(F.cross_entropy(logits.flatten(0, 1), target[:, 1:].flatten(), ignore_index=PAD, reduction="sum"))
        tokens += int(target[:, 1:].ne(PAD).sum())
    return loss / tokens


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def snapshot(network, cfg, output):
    tensors = {key: value.detach().cpu().numpy() for key, value in network.state_dict().items()}
    output.mkdir(exist_ok=True, parents=True)
    pending = output / "model.tmp.npz"
    np.savez_compressed(pending, **tensors)
    pending.replace(output / "model.npz")
    atomic_json(output / "vocab.json", cfg)


def free_evaluation(model, rows, rng, count):
    # All references are held out. Report overlap and actual outputs separately.
    output = []
    for i in rng.sample(range(len(rows)), min(count, len(rows))):
        context, reference, target, source = rows[i]
        response = model.generate(context)
        predicted = [t for t in tokenize(response) if t[0].isalnum()]
        actual = [t for t in target if t[0].isalnum()]
        overlap = sum((Counter(predicted) & Counter(actual)).values())
        output.append({"context": context, "reference": reference, "generated": response, "source": source,
                       "unigram_f1": 2 * overlap / max(1, len(predicted) + len(actual))})
    return output


def main():
    p = argparse.ArgumentParser()
    for name in ("train", "valid", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--vocab", type=int, default=5500)
    p.add_argument("--buckets", type=int, default=8192)
    p.add_argument("--embedding", type=int, default=96)
    p.add_argument("--hidden", type=int, default=160)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--seed", type=int, default=272)
    p.add_argument("--learning-rate", type=float, default=.003)
    p.add_argument("--dropout", type=float, default=.12)
    p.add_argument("--contrastive", type=float, default=.12,
                   help="Weight for matching each context to its observed reply within a batch.")
    p.add_argument("--context-dropout", type=float, default=0.0)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--focus-source", default="")
    p.add_argument("--focus-weight", type=int, default=1)
    p.add_argument("--evaluation-count", type=int, default=48)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    train, valid = read(args.train, args.max_tokens), read(args.valid, args.max_tokens)
    frequency = Counter(t for _, _, target, _ in train for t in target)
    tokens = SPECIAL + sorted(frequency, key=lambda t: (-frequency[t], t))[:args.vocab - len(SPECIAL)]
    cfg = {"format": "arena-context-gru-v1", "tokens": tokens, "width": args.width,
           "buckets": args.buckets, "embedding": args.embedding, "hidden": args.hidden,
           "dropout": args.dropout, "beam": 4, "response_limit": args.max_tokens,
           "context_dropout": args.context_dropout, "guidance": args.guidance}
    tr, va = encode_rows(train, cfg), encode_rows(valid, cfg)
    net = Network(cfg).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=.005)
    focus = [i for i, row in enumerate(train) if args.focus_source and row[3].startswith(args.focus_source)]
    sampling = np.asarray(list(range(len(train))) + focus * (args.focus_weight - 1))
    best, best_epoch, epochs = math.inf, 0, []
    print(f"Training {len(train)}/{len(valid)} pairs, {len(tokens)} tokens, {len(sampling)} draws, {device}", flush=True)
    report = {"args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "python_torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
              "accelerator": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
              "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
              "valid_sha256": hashlib.sha256(args.valid.read_bytes()).hexdigest(),
              "train_rows_used": len(train), "validation_rows_used": len(valid),
              "sources": dict(Counter(row[3] for row in train)), "focused_rows": len(focus),
              "parameters": sum(x.numel() for x in net.parameters()), "epochs": epochs}
    for epoch in range(1, args.epochs + 1):
        net.train()
        np.random.shuffle(sampling)
        loss_total, total = 0, 0
        for offset in range(0, len(sampling), args.batch):
            idx = sampling[offset:offset + args.batch]
            ids, weights, target = minibatch(tr, idx, device)
            logits = net(ids, weights, target[:, :-1])
            labels = target[:, 1:].reshape(-1)
            nll = F.cross_entropy(logits.flatten(0, 1), labels, ignore_index=PAD)
            loss = nll
            if args.contrastive > 0 and len(idx) > 1:
                actual = target[:, 1:]
                mask = actual.ge(len(SPECIAL)).unsqueeze(-1)
                reply_vector = (F.embedding(actual, net.topic.weight) * mask).sum(1)
                query_vector = net.encode(ids, weights)
                similarities = F.normalize(query_vector) @ F.normalize(reply_vector).T
                contrast = F.cross_entropy(similarities / .15, torch.arange(len(idx), device=device))
                loss = nll + args.contrastive * contrast
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            n = int(labels.ne(PAD).sum())
            loss_total += float(nll.detach()) * n
            total += n
        vnll = validation_loss(net, va, device, args.batch)
        if vnll < best - 1e-5:
            best, best_epoch = vnll, epoch
            snapshot(net, cfg, args.output)
        metrics = {"epoch": epoch, "train_nll": loss_total / total, "valid_nll": vnll,
                   "seconds_from_start": time.perf_counter() - start}
        epochs.append(metrics)
        report.update(best_epoch=best_epoch, best_valid_nll=best)
        atomic_json(args.report, report)
        print(json.dumps(metrics), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model = Generator.load(args.output)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in model.weights.items()})
    permutation = torch.as_tensor(np.random.default_rng(args.seed).permutation(len(valid)))
    swapped = (va[0][permutation], va[1][permutation], va[2], va[3])
    shuffled_nll = validation_loss(net, swapped, device, args.batch)
    report["context_test"] = {"aligned_nll": best, "shuffled_context_nll": shuffled_nll,
                              "gain_nats_per_token": shuffled_nll - best}
    valid_tokens = va[2][:, 1:]
    report["validation_unknown_fraction"] = float(valid_tokens.eq(UNK).sum() / valid_tokens.ne(PAD).sum())
    generated = free_evaluation(model, valid, random.Random(args.seed), args.evaluation_count)
    report["free_decoding"] = {
        "samples": generated, "distinct_responses": len({r["generated"] for r in generated}),
        "mean_unigram_f1": sum(r["unigram_f1"] for r in generated) / max(1, len(generated))}
    report["elapsed_seconds"] = time.perf_counter() - start
    atomic_json(args.report, report)


if __name__ == "__main__":
    main()
