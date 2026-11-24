"""Encode the stored questions of the candidate pairs and write the index.

Each candidate is a (question, reply) pair. The index stores the context
vector of the question, reduced with PCA to --dim components and quantised to
int8 with one scale per row, in shards under 45 MB so every file stays well
below GitHub's 100 MB limit. Replies are encoded at run time for the few
nearest candidates only.

    python -m training.build_reply_index --index prepared/index.jsonl \
        --model model_dir --out turing_bot/assets/retrieval --device cuda
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from turing_bot.language import Pieces
from turing_bot.retrieval import SENSITIVE, context_ids, persona_safe
from training.train_retrieval import DualEncoder, batchify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--limit", type=int, default=1200000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-mb", type=int, default=45)
    args = ap.parse_args()
    folder, out = Path(args.model), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((folder / "encoder.json").read_text(encoding="utf-8"))
    pieces = Pieces(config["vocab"])
    model = DualEncoder(len(pieces.tokens), d=config["dim"], layers=config["layers"], ff=3 * config["dim"], out=config["dim"])
    model.load_state_dict(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True))
    model.to(args.device).eval()

    rows, seen = [], set()
    for line in open(args.index, encoding="utf-8"):
        r = json.loads(line)
        key = (r["c"].casefold(), r["r"].casefold())
        if key in seen or not persona_safe(r["r"]) or SENSITIVE.search(r["r"]) or SENSITIVE.search(r["c"]):
            continue
        seen.add(key)
        rows.append(r)
        if len(rows) >= args.limit:
            break
    vectors = []
    with torch.no_grad():
        for start in range(0, len(rows), 2048):
            chunk = rows[start:start + 2048]
            ids = batchify([context_ids(pieces, r["c"]) for r in chunk], args.device)
            vectors.append(model.encode(ids, "ctx").float().cpu().numpy())
    vectors = np.concatenate(vectors)
    sample = vectors[np.random.default_rng(0).choice(len(vectors), min(200000, len(vectors)), replace=False)]
    mean = sample.mean(axis=0)
    _, singular, vt = np.linalg.svd(sample - mean, full_matrices=False)
    components = vt[:args.dim]
    kept = float((singular[:args.dim] ** 2).sum() / (singular ** 2).sum())
    reduced = (vectors - mean) @ components.T
    reduced /= np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-8
    scales = np.abs(reduced).max(axis=1) / 127.
    quant = np.round(reduced / scales[:, None]).astype(np.int8)

    for old in list(out.glob("contexts-*.npy")) + list(out.glob("vectors-*.npy")):
        old.unlink()
    per = max(1, args.shard_mb * 1024 * 1024 // quant.shape[1])
    for n, start in enumerate(range(0, len(quant), per)):
        np.save(out / f"contexts-{n:02d}.npy", quant[start:start + per])
    np.save(out / "scales.npy", scales.astype(np.float16))
    np.savez(out / "pca.npz", mean=mean.astype(np.float32), components=components.astype(np.float32))
    for old in out.glob("replies*.jsonl.gz"):
        old.unlink()
    per_part = 700000  # keeps each compressed part under 50 MB
    for n, start in enumerate(range(0, len(rows), per_part)):
        with gzip.open(out / f"replies-{n:02d}.jsonl.gz", "wt", encoding="utf-8") as stream:
            for r in rows[start:start + per_part]:
                stream.write(json.dumps(dict(c=r["c"], r=r["r"]), ensure_ascii=False) + "\n")
    for name in ("encoder.json", "encoder.npz", "training-report.json"):
        if (folder / name).resolve() != (out / name).resolve():
            shutil.copyfile(folder / name, out / name)
    report = dict(pairs=len(rows), pca_dim=args.dim, pca_variance_kept=round(kept, 4))
    (out / "index-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
