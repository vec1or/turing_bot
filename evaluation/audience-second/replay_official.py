"""Повторить только публичную тройку в новой сессии, используя зафиксированные вопросы."""

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
import unicodedata
from pathlib import Path

from replay import FILES, data_for, neural


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Выберите новое имя")
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    create, answer = neural(root)
    bot = create()
    questions = data_for(Path(__file__).with_name("official.json"), FILES["official"])
    rows = []
    for turn in questions["dialogues"][0]["turns"]:
        start = time.perf_counter()
        text, bot = answer(turn["question"], bot)
        elapsed = time.perf_counter() - start
        rows.append({"question": turn["question"], "text": text, "engine_seconds": elapsed,
                     "length": len(text), "nfc": unicodedata.is_normalized("NFC", text)})
    package = root / "turing_bot"
    digest = hashlib.sha256()
    for file in sorted(package.rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts:
            digest.update(file.relative_to(package).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(file.read_bytes()).digest())
    result = {"status": "post-disclosure public retest", "runtime_sha256": digest.hexdigest(),
              "python": platform.python_version(), "platform": platform.platform(),
              "interfaces": [name for _, name in socket.if_nameindex()], "rows": rows}
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return any(not (r["nfc"] and r["text"].strip() and r["length"] <= 300 and r["engine_seconds"] <= 45)
               for r in rows)


if __name__ == "__main__":
    sys.exit(run())
