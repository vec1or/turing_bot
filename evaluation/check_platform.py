"""Run the real CLI as a fresh subprocess and verify the operator's JSONL journal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

p=argparse.ArgumentParser()
p.add_argument("--out",type=Path, required=True)
p.add_argument("--network",required=True)
a=p.parse_args()
examples=["Добрый вечер!", "Чем пахнет вторник?", "Ты бот или человек?", "А если наоборот, человек или бот?",
          "Я положил фиолетовый зонт в рюкзак.", "Какого цвета был зонт?", "Что выберешь, какао или кефир?",
          "Ты случайно детектив?", "Почему лёд плавает?", "Как поддержать друга перед выступлением?",
          "Что такое атмосфера?", "Что такое фотосинтез?", "Что такое вулкан?", "Что такое квантовый кактус?",
          "Спасибо за разговор!", "?!", "🚀", "Расскажи " + "про метеориты " * 350,
          "Если бы у города был голос, каким бы он был?", "Что ты выберешь, чай или какао?",
          "Что ты только что выбрал?", "Где лежат " + "носки " * 600 + "?"]
start=time.perf_counter()
with tempfile.TemporaryDirectory() as td:
    log=Path(td)/"session.jsonl"
    result=subprocess.run([sys.executable,"-m","turing_bot","--log",str(log)],
                          input="\n".join([*examples,"/reset","/exit",""]),capture_output=True,
                          text=True,encoding="utf-8",timeout=180,check=True)
    rows=[json.loads(s) for s in log.read_text(encoding="utf-8").splitlines()]
assert len(rows)==len(examples)
assert all(0<len(r["answer"])<=300 and unicodedata.is_normalized("NFC",r["answer"]) for r in rows)
assert all(re.search("[А-Яа-яЁё]",r["answer"]) for r in rows)
assert all(0<=r["seconds"]<45 for r in rows)
assert all(r["answer"] in result.stdout for r in rows)
measure=sorted(r["seconds"] for r in rows)
report={"platform":platform.platform(),"python":sys.version.split()[0],"network":a.network,
        "numpy":__import__("numpy").__version__,"questions":len(rows),
        "seconds_median":statistics.median(measure),"seconds_p95":measure[int(.95*len(measure)+.99999)-1],
        "seconds_max":max(measure),"max_chars":max(len(r["answer"]) for r in rows),
        "cold_process_total_seconds":time.perf_counter()-start,
        "model_sha256":hashlib.sha256(Path("turing_bot/assets/model.npz").read_bytes()).hexdigest(),
        "rows":rows}
a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print({k:v for k,v in report.items() if k!="rows"})
