import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from . import __version__
from .session import MAX_INPUT, Session


def main():
    parser = argparse.ArgumentParser(description="Текстовый диалог за ширмой")
    parser.add_argument("--question", help="Один вопрос, затем выход")
    parser.add_argument("--log", type=Path, help="Записать вопросы, ответы и время в новый локальный файл JSONL")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8")
    try:
        bot = Session()
        log = args.log.open("x", encoding="utf-8") if args.log else None
    except (OSError, ValueError) as exc:
        print(f"Ошибка запуска. {exc}", file=sys.stderr)
        return 2
    try:
        if args.question is None:
            print(f"Тьюринг Арена {__version__}. Введите вопрос. /reset — новый диалог, /exit — выход.", flush=True)
        while True:
            if args.question is None:
                try:
                    question = input("Вы> " if sys.stdin.isatty() else "")
                except EOFError:
                    break
                if question.strip() == "/exit":
                    break
                if question.strip() == "/reset":
                    bot.reset()
                    print("Начат новый раунд.", flush=True)
                    continue
            else:
                question = args.question
            start = perf_counter()
            if not question.strip():
                print("Введите непустой вопрос.", file=sys.stderr, flush=True)
                if args.question is not None:
                    return 2
                continue
            reply = bot.answer(question)
            print(reply, flush=True)
            elapsed = perf_counter() - start
            print(f"Время {elapsed:.3f} с · {len(reply)} симв.", file=sys.stderr, flush=True)
            if bot.last_source:
                print(f"Источник: участники Википедии, RuBQ, CC BY-SA 4.0 · {bot.last_source}", file=sys.stderr, flush=True)
            if log:
                log.write(json.dumps(dict(question=question[:MAX_INPUT], answer=reply, seconds=elapsed, source=bot.last_source,
                                          route=bot.last_route, version=__version__),
                                     ensure_ascii=False) + "\n")
                log.flush()
            if args.question is not None:
                break
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Ошибка обработки. {exc}", file=sys.stderr)
        return 2
    finally:
        if log:
            log.close()
    return 0
