"""Сопоставление двух наборов и трёх сборок по двум разным режимам сессии."""

import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(rows):
    def line(cells):
        values = [str(c).replace("|", "\\|").replace("\n", " ") for c in cells]
        return "| " + " | ".join(values) + " |"
    return "\n".join([line(rows[0]), line(["---"] * len(rows[0])), *(line(r) for r in rows[1:])])


def valid(row):
    return (not row["error"] and row["nfc"] and row["text"].strip()
            and row["length"] <= 300 and row["engine_seconds"] <= 45)


def duplicate_count(rows):
    keys = {" ".join(unicodedata.normalize("NFC", r["text"]).casefold().split()) for r in rows}
    return len(rows) - len(keys)


def analyze(fresh):
    target = ROOT / ("second" if fresh else "known-repeat")
    earlier = ROOT if (ROOT / "actual-2.1.json").is_file() else ROOT.parent / "arena-blind"
    source = target if fresh else earlier
    spec = read(source / "questions.json")
    deltas = read(target / "continuous-deltas.json")
    ids = [c["id"] for c in spec["cases"]]
    seq = []
    totals = [["Режим", "Версия", "Уместно", "fallback", "abstention", "Оба маршрута", "Символ ?", "Повторы",
               "Медиана / p95 / макс., с", "Технич. успешны"]]
    change_table = [["Версия", "Ход", "Текст при сбросе", "Текст в непрерывном раунде",
                     "Уместно при сбросе / в раунде", "Оценка изменения"]]
    for version in ("2.0", "A", "B"):
        base = target if version == "B" else source
        alias = "2.1" if not fresh and version == "A" else version
        iso = read(base / f"actual-{alias}.json")["rows"]
        scored = read(base / f"qa-{alias}.json")
        cont = read(target / f"continuous-{version}.json")["rows"]
        assert list(scored) == ids == [r["id"] for r in iso] == [r["id"] for r in cont]
        seq.append((iso, scored))
        flags = [bool(valid(r) and scored[r["id"]]["language"] >= 1
                      and scored[r["id"]]["meaning"] == 2) for r in iso]
        other_flags = list(flags)
        changes = deltas[version]
        changed_ids = {r["id"] for r, s in zip(iso, cont, strict=True) if r["text"] != s["text"]}
        assert set(changes) == changed_ids
        for n, (a, b) in enumerate(zip(iso, cont, strict=True)):
            assert a["question"] == b["question"]
            key = a["id"]
            if key in changed_ids:
                detail = changes[key]
                other_flags[n] = bool(valid(b) and detail["meaning"] == 2 and detail["language"] >= 1)
                change_table.append([version, key, a["text"], b["text"],
                                     f"{int(flags[n])} / {int(other_flags[n])}", detail["note"]])
        for mode, rows, count in (("Сброс", iso, sum(flags)), ("Весь раунд", cont, sum(other_flags))):
            ts = sorted(r["engine_seconds"] for r in rows)
            times = f"{statistics.median(ts):.3f} / {ts[math.ceil(.95*len(ts))-1]:.3f} / {max(ts):.3f}"
            fallback = "н/д" if version == "2.0" else sum(r["route"] == "fallback" for r in rows)
            abstention = "н/д" if version == "2.0" else sum(r["route"] == "abstention" for r in rows)
            combined = "н/д" if version == "2.0" else fallback + abstention
            totals.append([mode, version, count, fallback, abstention, combined, sum(r["has_question_mark"] for r in rows),
                           duplicate_count(rows), times, sum(bool(valid(r)) for r in rows)])
    cat_table = [["Категория", "Ходов", "Уместно 2.0", "Уместно A", "Уместно B", "fallback B"]]
    for key, label in spec["categories"].items():
        passed = []
        for dataset, ratings in seq:
            matched = [r for r in dataset if r["category"] == key]
            passed.append(sum(bool(valid(r) and ratings[r["id"]]["meaning"] == 2
                                   and ratings[r["id"]]["language"] >= 1) for r in matched))
        cat_table.append([label, len(matched), *passed, sum(r["route"] == "fallback" for r in matched)])
    all_table = [["№", "Вопрос", "2.0", "2.1 A", "2.1 B", "Маршрут B", "Содержание / язык B",
                  "Уход B", "Причина оценки B"]]
    form = [["№", "Вопрос", "Ожидание", "Фактический ответ B", "Содержание", "Язык", "Комментарий"]]
    for i, key in enumerate(ids):
        a, b, c = [entry[0][i] for entry in seq]
        t = seq[2][1][key]
        all_table.append([key, c["question"], a["text"], b["text"], c["text"], c["route"],
                          f"{t['meaning']} / {t['language']}", "да" if t["generic"] else "нет", t["note"]])
        form.append([key, c["question"], c["expectation"], c["text"], "", "", ""])
    buckets = [["Версия", "Язык 0/1/2", "Содержание 0/1/2", "Уход по QA", "Явное признание ограничения"]]
    for version, (dataset, ratings) in zip(("2.0", "A", "B"), seq, strict=True):
        counts = [Counter(v[field] for v in ratings.values()) for field in ("language","meaning")]
        # Legacy A-set grades were sealed before this extra classification was requested.
        explicit = sum(v.get("explicit_limit", r["text"].casefold().startswith("я не знаю"))
                       for r, v in zip(dataset, ratings.values(), strict=True))
        buckets.append([version, *[" / ".join(str(count[i]) for i in range(3)) for count in counts],
                        sum(v["generic"] for v in ratings.values()), explicit])
    page = ("# Результаты трёх сборок\n\nУместность оценивал программный QA-агент. "
            "В таблицах по темам, типам и отдельным вопросам используется независимый сброс "
            "сессии перед каждым ходом. Непрерывный раунд отдельно измеряет влияние истории.\n\n"
            + table(totals) + "\n\n" + table(cat_table) + "\n\n" + table(buckets)
            + "\n\n" + table(all_table) + "\n")
    (target / "comparison.md").write_text(page, encoding="utf-8")
    (target / "round-changes.md").write_text("# Изменившиеся ответы без сброса\n\n"
                                              + table(change_table) + "\n", encoding="utf-8")
    sheet = target / "human-review.md"
    if not sheet.exists():
        sheet.write_text("# Лист для человека\n\nОценка человеком не выполнена. "
                         "Содержание и язык оцениваются от 0 до 2 по questions.json.\n\n"
                         + table(form) + "\n", encoding="utf-8")


if __name__ == "__main__":
    analyze(False)
    analyze(True)
