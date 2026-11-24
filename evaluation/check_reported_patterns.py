"""Replay known patterns from the user's description; this is an open regression set."""
import argparse
import json
from pathlib import Path
import sys
import time

QUESTIONS=(
    "Сова или жаворонок?", "«Сова или жаворонок?»",
    "Что тебе ближе: быть лидером или отвечать за атмосферу?",
    "Кто ты за ширмой: человек или программа?", "Докажи, что ты не бот.",
    "Назови три причины, почему команда срывает сроки.",
    "Назови три плюса учёбы в команде.",
    "Если бы у старого чайника появился голос, что он сказал бы?",
    "Если бы ты мог остановить время на десять минут, что бы сделал?",
    "Что бы ты сделал, если бы на сутки стал невидимым?",
    "Какое у тебя настроение?", "Что обычно ешь с утра, если опаздываешь на пару?",
    "Какая на улице погода?", "Чем пахнет понедельник?",
    "Выбирай: ходить по потолку или говорить с фонарями. Почему?",
    "Что делать, если я совсем не успеваю закончить проект?",
    "Что такое кромбист?", "Что такое луногвоздик?",
)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.project.resolve()))
    from turing_bot import Session, __version__
    bot=Session()
    rows=[]
    for q in QUESTIONS:
        start=time.perf_counter()
        answer=bot.answer(q)
        rows.append(dict(question=q,answer=answer,route=getattr(bot,"last_route",None),seconds=time.perf_counter()-start))
    result=dict(version=__version__,sample="Открытые примеры по описанию пользователя. Исходный лог из 17 вопросов не приложен.",rows=rows,
                question_marks=sum("?" in r["answer"] for r in rows),
                fallback=sum(r["route"]=="fallback" for r in rows) if hasattr(bot,"last_route") else None)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(result["version"],len(rows),result["question_marks"],result["fallback"])
