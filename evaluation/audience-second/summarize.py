"""Построить таблицу реальных реплик и посчитать предварительные оценки QA."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(folder, filename):
    return json.loads((folder / filename).read_text(encoding="utf-8"))


def rows_for(folder, tag):
    raw = read(folder, f"{tag}.json")["rows"]
    scores = read(folder, f"{tag}-ratings.json")
    assert list(scores) == [row["id"] for row in raw]
    for row in raw:
        score = scores[row["id"]]
        assert score["grammar"] in (0, 1, 2) and score["substance"] in (0, 1, 2)
        if row["turn"] == 1:
            assert score["context"] is None
        else:
            assert score["context"] in (0, 1, 2)
        yield row | score


def accepted(row):
    return (row["grammar"] >= 1 and row["substance"] == 2
            and row["context"] in (None, 1, 2))


def table(rows):
    def line(cells):
        return "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"

    return "\n".join([line(rows[0]), line(["---"] * len(rows[0])), *(line(r) for r in rows[1:])])


def distribution(rows, field):
    numbers = Counter(row[field] for row in rows)
    return " / ".join(str(numbers[score]) for score in (0, 1, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT, help="Папка с протоколами и разметкой")
    args = parser.parse_args()
    folder = args.folder
    versions = {"1.1.0": list(rows_for(folder, "legacy")), "2.0": list(rows_for(folder, "neural"))}
    short = [["Версия", "Набор", "Ходов", "Грамматика 0/1/2", "Содержание 0/1/2",
              "Контекст 0/1/2", "Прошли порог"]]
    categories = {"conversation": "Свободная беседа", "situation": "Бытовая или социальная ситуация",
                  "reason": "Открытый вопрос с объяснением", "identity": "Прямое сомнение",
                  "limit": "Неизвестная или неразрешимая задача"}
    by_category = [["Категория", "Ходов", "Прошли 1.1.0", "Прошли 2.0"]]
    for name, rows in versions.items():
        for suite in dict.fromkeys(row["suite"] for row in rows):
            subset = [row for row in rows if row["suite"] == suite]
            short.append([name, suite, len(subset), distribution(subset, "grammar"),
                          distribution(subset, "substance"), distribution(subset, "context"),
                          sum(accepted(row) for row in subset)])
    for category, title in categories.items():
        pairs = [[row for row in rows if row["category"].replace("scope", "limit") == category]
                 for rows in versions.values()]
        by_category.append([title, len(pairs[0]), *(sum(accepted(r) for r in rows) for rows in pairs)])
    details = [["Ход", "Вопрос", "Ответ 1.1.0", "Ответ 2.0",
                "2.0, грамматика / содержание / контекст", "Пояснение QA о 2.0"]]
    form = [["Ход", "Вопрос", "Ответ 2.0", "Ожидание", "Грамматика", "Содержание",
             "Контекст", "Комментарий"]]
    for old, new in zip(*versions.values(), strict=True):
        assert old["id"] == new["id"] and old["question"] == new["question"]
        context = "н/п" if new["context"] is None else new["context"]
        mark = f"{new['grammar']} / {new['substance']} / {context}"
        details.append([new["id"], new["question"], old["text"], new["text"], mark, new["note"]])
        form.append([new["id"], new["question"], new["text"], new["expectation"],
                     "", "", "н/п" if new["turn"] == 1 else "", ""])
    page = ("# Контрольные ответы и предварительная оценка QA\n\n"
            "В каждой тройке приведено число оценок 0, 1 и 2. Контекст оценивался "
            "только на продолжениях беседы. Порог для отдельной реплики требует "
            "содержательности 2, грамматики от 1 и контекста от 1 при применимости.\n\n"
            + table(short) + "\n\n" + table(by_category) + "\n\n" + table(details) + "\n")
    (folder / "comparison.md").write_text(page, encoding="utf-8")
    human = folder / "human-review.md"
    if not human.exists():
        human.write_text("# Лист для оценщика\n\nОценщик ________________ Дата __________\n\n"
                         "Человеческая оценка не выполнена. Критерии от 0 до 2 зафиксированы "
                         "в audience.json. Читайте реплики каждого диалога в указанном порядке.\n\n"
                         + table(form) + "\n",
                         encoding="utf-8")


if __name__ == "__main__":
    main()
