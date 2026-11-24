import argparse
import json
from pathlib import Path

from turing_bot.word_model import fit


def main():
    parser = argparse.ArgumentParser(description="Обучение словесной модели на train.jsonl")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.train.open(encoding="utf-8") as inp:
        fit((json.loads(row) for row in inp), args.output / "words.json.gz")


if __name__ == "__main__":
    main()
