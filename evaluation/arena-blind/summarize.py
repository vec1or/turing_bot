"""Пересчитать сводку из фактических протоколов и независимой разметки."""

import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(filename):
    return json.loads((ROOT / filename).read_text(encoding="utf-8"))


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def technical(row):
    return (row["error"] is None and bool(row["text"].strip()) and row["nfc"]
            and row["length"] <= 300 and row["engine_seconds"] <= 45)


def score_rows(tag):
    protocol, scores = read(f"actual-{tag}.json"), read(f"qa-{tag}.json")
    assert list(scores) == [row["id"] for row in protocol["rows"]]
    rows = []
    for row in protocol["rows"]:
        score = scores[row["id"]]
        assert score["language"] in {0, 1, 2} and score["meaning"] in {0, 1, 2}
        assert isinstance(score["generic"], bool)
        passed = technical(row) and score["meaning"] == 2 and score["language"] >= 1
        rows.append(row | score | {"passed": passed})
    return rows


def table(rows):
    def line(cells):
        values = [str(x).replace("|", "\\|").replace("\n", " ") for x in cells]
        return "| " + " | ".join(values) + " |"
    return "\n".join([line(rows[0]), line(["---"] * len(rows[0])), *(line(row) for row in rows[1:])])


def metrics(rows):
    times = sorted(r["engine_seconds"] for r in rows)
    return [
        sum(r["passed"] for r in rows), sum(r["generic"] for r in rows),
        sum(r["has_question_mark"] for r in rows),
        len(rows) - len({normalized(r["text"]) for r in rows}), sum(technical(r) for r in rows),
        f"{statistics.median(times):.3f}", f"{times[math.ceil(.95 * len(times)) - 1]:.3f}",
        f"{max(times):.3f}", max(r["length"] for r in rows),
    ]


def main():
    spec = read("questions.json")
    old = score_rows("2.0")
    new = score_rows("2.1")
    assert [r["id"] for r in new] == [c["id"] for c in spec["cases"]]
    assert [r["question"] for r in old] == [r["question"] for r in new]
    topline = [["Версия", "Уместный ответ", "Уход", "Со знаком ?", "Повторные строки",
                "Прошли технику", "Медиана, с", "p95, с", "Макс., с", "Макс. знаков"],
               ["2.0", *metrics(old)], ["2.1", *metrics(new)]]
    topics = [["Категория", "Ходов", "Уместно 2.0", "Уместно 2.1", "Уход 2.0", "Уход 2.1"]]
    for category, label in spec["categories"].items():
        previous = [r for r in old if r["category"] == category]
        current = [r for r in new if r["category"] == category]
        topics.append([label, len(current), sum(r["passed"] for r in previous),
                       sum(r["passed"] for r in current), sum(r["generic"] for r in previous),
                       sum(r["generic"] for r in current)])
    domains = [["Ось", "0 / 1 / 2 в 2.0", "0 / 1 / 2 в 2.1"]]
    for field, title in (("meaning", "Содержательность"), ("language", "Язык")):
        domains.append([title, *[" / ".join(str(sum(r[field] == v for r in rows)) for v in range(3))
                                for rows in (old, new)]])
    routes = Counter(str(r["route"]) for r in new)
    route_rows = [["Значение last_route в 2.1", "Ходов"],
                  *[[route, n] for route, n in routes.most_common()]]
    answers = [["№", "Вопрос", "Ответ 2.0", "Ответ 2.1", "Маршрут 2.1", "Содержание / язык 2.1",
                "Уход 2.1", "Пояснение QA к 2.1"]]
    sheet = [["№", "Вопрос", "Предварительное ожидание", "Ответ 2.1",
              "Содержание 0–2", "Язык 0–2", "Уход да/нет", "Пояснение"]]
    for a, b in zip(old, new, strict=True):
        answers.append([a["id"], a["question"], a["text"], b["text"], b["route"],
                        f"{b['meaning']} / {b['language']}", "да" if b["generic"] else "нет", b["note"]])
        sheet.append([b["id"], b["question"], b["expectation"], b["text"], "", "", "", ""])
    report = ("# Все ответы и оценки\n\nОценивал QA-агент Codex. Маршрут из программы записан "
              "отдельно от смысловой оценки ответа.\n\n" + table(topline) + "\n\n"
              + table(topics) + "\n\n" + table(domains) + "\n\n" + table(route_rows)
              + "\n\nУ 2.0 поле last_route отсутствует. Время измерено вокруг Session.answer. "
              "Повторные строки — число ответов после первого в группе, совпавшей после NFC, "
              "приведения к нижнему регистру и нормализации пробелов.\n\n" + table(answers) + "\n")
    (ROOT / "comparison.md").write_text(report, encoding="utf-8")
    sheet_path = ROOT / "human-review.md"
    if not sheet_path.exists():
        sheet_path.write_text("# Лист для человеческой оценки\n\nЧеловеческая оценка не выполнена. "
                              "Оценщик __________ Дата __________. Критерии указаны в questions.json.\n\n"
                              + table(sheet) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
