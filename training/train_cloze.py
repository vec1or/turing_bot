import argparse
from hashlib import sha256
import json
from pathlib import Path
from turing_bot.cloze import train


def main():
    p = argparse.ArgumentParser(description="Обучение заполнению пропусков по русским предложениям CC0 Tatoeba")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--valid", type=Path, required=True, help="Исключить исходные предложения отложенного корпуса")
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    print(f"SHA-256 {sha256(a.archive.read_bytes()).hexdigest()}")
    with a.valid.open(encoding="utf-8") as inp:
        exclude_ids = {str(sample["sentence_id"]) for row in inp if "sentence_id" in (sample := json.loads(row))}
    print(f"Предложений после фильтра: {train(a.archive,a.output/'cloze.json.gz',exclude_ids)}")


if __name__ == "__main__":
    main()
