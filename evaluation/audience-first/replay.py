"""Запуск 27 закрытых и 3 публичных вопросов на выбранном дереве исходников."""

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = {
    "audience": "4f1b1e4c9b8a984f3b75eb3d0896091f6bba0ffe4319c61d54e0987589abaf1b",
    "official": "4f03dde1d12f31dd9266a7b6607bfbb9572957205dd40219d41a21f3753ef87d",
}


def sessions(dataset):
    for case in dataset["single"]:
        yield case["id"], [case]
    for dialogue in dataset["dialogues"]:
        yield dialogue["id"], dialogue["turns"]


def data_for(tag):
    raw = (ROOT / f"{tag}.json").read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != FILES[tag]:
        raise ValueError(f"Изменился зафиксированный набор {tag}")
    return json.loads(raw)


def legacy(project):
    from turing_bot import engine, state

    if Path(engine.__file__).resolve().parent != project / "turing_bot":
        raise ValueError("Импортировано другое дерево исходников")
    bot = engine.Engine()

    def reply(question, memory):
        result = bot.answer(question, memory)
        return result.text, result.state

    return state.SessionState, reply


def neural(project):
    import turing_bot

    if Path(turing_bot.__file__).resolve().parent != project / "turing_bot":
        raise ValueError("Импортировано другое дерево исходников")

    def reply(question, session):
        return session.answer(question), session

    return turing_bot.Session, reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--version", required=True, choices=("1.1.0", "2.0"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Укажите новое имя протокола")
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    start = time.perf_counter()
    create_state, generate = (legacy if args.version == "1.1.0" else neural)(project)
    load = time.perf_counter() - start
    rows, session_seconds = [], []
    for tag in FILES:
        dataset = data_for(tag)
        for case, turns in sessions(dataset):
            session_start = time.perf_counter()
            state = create_state()
            session_seconds.append(time.perf_counter() - session_start)
            for n, turn in enumerate(turns, 1):
                begin = time.perf_counter()
                try:
                    text, state = generate(turn["question"], state)
                    elapsed = time.perf_counter() - begin
                    error = None
                except Exception as exc:
                    elapsed = time.perf_counter() - begin
                    text, error = "", f"{type(exc).__name__}: {exc}"
                rows.append({"id": f"{case}.{n}", "suite": tag, "turn": n,
                             "category": turn["category"], "question": turn["question"],
                             "expectation": turn["expectation"], "text": text, "error": error,
                             "length": len(text), "nfc": unicodedata.is_normalized("NFC", text),
                             "engine_seconds": elapsed})
    result = {"version": args.version, "datasets_sha256": FILES, "platform": platform.platform(),
              "python": platform.python_version(), "init_seconds": load, "rows": rows,
              "session_init_seconds": session_seconds,
              "interfaces": [name for _, name in socket.if_nameindex()]}
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    faults = sum(bool(row["error"] or not row["nfc"] or not row["text"].strip()
                      or row["length"] > 300 or row["engine_seconds"] > 45) for row in rows)
    print(f"{args.version}: {len(rows)} / {faults}")
    return bool(faults)


if __name__ == "__main__":
    sys.exit(main())
