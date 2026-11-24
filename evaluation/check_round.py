"""Exercise an ordinary interactive CLI, including cold imports, with socket creation denied."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

QUESTIONS = [
    "Почему после дождя иногда появляется радуга? Объясни простыми словами.",
    "Что бы ты посоветовал человеку, который впервые приехал в Астану зимой?",
    "Друг опоздал на встречу на полчаса и извинился. Что бы ты ему ответил?",
    "«   Ты человек   или бот?  »", "Сова или жаворонок?", "А почему?",
    "Сколько будет 19 плюс 23?", "Какого цвета была бы тревога? Одним словом.",
    "Что выберешь: ходить по потолку или говорить с фонарями? И почему?",
    "Назови три причины, почему команда срывает сроки.", "Какая погода сейчас на улице?",
    "Напиши одно предложение со словом «луноплющ».", "Что ел на завтрак?",
    "Что если бы у кружки из общежития вдруг появился голос? Что бы она сказала?",
    "Что ты сделал бы, если бы на день стал невидимым?", "Где я оставил свою закладку?",
    "Что такое флугомор?", "Что такое флугомор?", "Что такое флугомор?", "?!", "«»", "🚀",
    "Расскажи " + "про метеориты " * 350,
]

AUDITED_CLI = """
import runpy, sys
def network(event, args):
    if event.startswith("socket.") and event not in {"socket.gethostname"}:
        raise RuntimeError("Сетевое обращение при офлайн-проверке: " + event)
sys.addaudithook(network)
runpy.run_module("turing_bot", run_name="__main__", alter_sys=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loopback-only", action="store_true")
    args = parser.parse_args()
    interfaces = [name for _,name in socket.if_nameindex()]
    if args.loopback_only:
        assert interfaces == ["lo"], interfaces
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp:
        log = Path(temp)/"round.jsonl"
        child = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", AUDITED_CLI, "--log",str(log)],
                               input="\n".join([*QUESTIONS, "/reset", "/exit", ""]), cwd=root,
                               capture_output=True, encoding="utf-8", timeout=180,check=True,
                               env={**os.environ,"OPENBLAS_NUM_THREADS":"1"})
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(QUESTIONS)
    assert "2.1.2" in child.stdout and "Начат новый раунд." in child.stdout
    assert all(r["question"] == q[:4000] and r["version"]=="2.1.2" and r["route"] for q,r in zip(QUESTIONS,rows))
    for r in rows:
        answer = r["answer"]
        assert 0<len(answer)<=300 and unicodedata.is_normalized("NFC",answer)
        assert re.search("[А-Яа-яЁё]",answer) and "?" not in answer and answer in child.stdout
        assert 0 <= r["seconds"] < 45
    fallback = [r["answer"] for r in rows if r["route"]=="fallback"]
    assert len(fallback)==len(set(fallback))
    report = dict(python=platform.python_version(),system=platform.platform(),
                  network="loopback namespace + Python audit" if args.loopback_only else "Python audit (Windows adapters unchanged)",
                  only_loopback=(interfaces==["lo"]), numpy=__import__("numpy").__version__,
                  questions=len(rows), seconds_max=max(r["seconds"] for r in rows),
                  seconds_median=statistics.median(r["seconds"] for r in rows),max_chars=max(len(r["answer"]) for r in rows),
                  cold_process_seconds=time.perf_counter()-start,rows=rows)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print({k:v for k,v in report.items() if k!="rows"})


if __name__ == "__main__":
    main()
