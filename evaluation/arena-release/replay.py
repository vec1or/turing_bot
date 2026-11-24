"""Снять один проверочный протокол из указанного дерева исходников."""

import argparse
import hashlib
import json
import platform
import sys
import time
import unicodedata
from pathlib import Path


QUESTION_HASH = "a5886aa650fc6aee09998f1f89cb16b53dacd5ec37dfc1e7b611f846a622ace4"
SECOND_HASH = "8d99e088235e45a879ee2d01b682a6ed95ee3e6f5a527803edc94f8cd0929c59"
THIRD_HASH = "baec2c4d5f004e4ca093072ea81f8bf602502f44f41a559eb257af767f8446c6"


def runtime_hash(package):
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(package).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--second", action="store_true", help="Набор B из 28 вопросов")
    choice.add_argument("--third", action="store_true", help="Набор C из 24 вопросов")
    parser.add_argument("--continuous", action="store_true", help="Одна сессия на весь раунд")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Выберите новое имя для протокола")
    question_dir = Path(__file__).resolve().parent
    primary = question_dir / "questions.json"
    if not primary.is_file():
        primary = question_dir.parent / "arena-blind" / "questions.json"
    if args.third:
        path, digest = question_dir / "third" / "questions.json", THIRD_HASH
    elif args.second:
        path, digest = question_dir / "second" / "questions.json", SECOND_HASH
    else:
        path, digest = primary, QUESTION_HASH
    raw = path.read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != digest:
        parser.error("Изменён зафиксированный набор")
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    start = time.perf_counter()
    import numpy
    import turing_bot

    package = Path(turing_bot.__file__).resolve().parent
    if package != project / "turing_bot":
        parser.error("Загружена другая копия программы")
    session = turing_bot.Session()
    init = time.perf_counter() - start
    route_available = hasattr(session, "last_route")
    rows = []
    for case in json.loads(raw)["cases"]:
        if not args.continuous:
            session.reset()
        begin = time.perf_counter()
        try:
            text = session.answer(case["question"])
            elapsed = time.perf_counter() - begin
            error = None
        except Exception as exc:
            elapsed = time.perf_counter() - begin
            text, error = "", f"{type(exc).__name__}: {exc}"
        rows.append({**case, "text": text, "length": len(text),
                     "nfc": unicodedata.is_normalized("NFC", text), "error": error,
                     "engine_seconds": elapsed, "has_question_mark": "?" in text,
                     "route": getattr(session, "last_route", None),
                     "source": getattr(session, "last_source", None)})
    result = {"version": turing_bot.__version__, "platform": platform.platform(),
              "python": platform.python_version(), "numpy": numpy.__version__,
              "startup_seconds": init, "route_available": route_available,
              "dataset_sha256": digest, "mode": "continuous" if args.continuous else "isolated",
              "runtime_sha256": runtime_hash(package), "rows": rows}
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    failures = sum(bool(r["error"] or not r["text"].strip() or not r["nfc"]
                        or r["length"] > 300 or r["engine_seconds"] > 45) for r in rows)
    print(f"{result['version']}: {len(rows)} / {failures}")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
