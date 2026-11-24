"""Replay the two frozen suites after the paragraph-boundary change."""
import json
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from turing_bot import Session


def main():
    evaluation=root/"evaluation"
    frozen=evaluation/"audience-second"
    bot=Session()
    checked=[]
    for source in (frozen/"neural.json",frozen/"second/neural.json"):
        first_official=True
        for row in json.loads(source.read_text(encoding="utf-8"))["rows"]:
            official=row["id"].startswith("O")
            if (official and first_official) or (not official and row["turn"]==1):
                bot.reset()
            if official:
                first_official=False
            start=time.perf_counter()
            actual=bot.answer(row["question"])
            elapsed=time.perf_counter()-start
            assert 0<len(actual)<=300 and elapsed<45
            checked.append(dict(id=row["id"],question=row["question"],before=row["text"],after=actual,
                                changed=actual!=row["text"],seconds=elapsed,source=bot.last_source))
    output=evaluation/"final-replay.json"
    output.write_text(json.dumps(checked,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(len(checked),"turns. Changed:",[r["id"] for r in checked if r["changed"]])


if __name__=="__main__":
    main()
