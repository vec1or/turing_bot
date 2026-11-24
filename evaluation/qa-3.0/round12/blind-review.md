# QA12 — blind 40-question session (one continuous run)

Run: run.jsonl, 40/40 answered, rc=0. Max latency 0.8 s, max length 170 chars — **no limit violations**.
Scores: R = relevance+correctness, N = naturalness, C = context (– = n/a). Acceptable = R2, N>=1, C>=1.

| # | Cat | Route | R | N | C | Risk | Acc | Note |
|---|-----|-------|---|---|---|------|-----|------|
| 1 | Personal | profile | 2 | 2 | – | low | Y | Караганда |
| 2 | Follow-up | follow-up | 2 | 1 | 2 | low | Y | tail "Тёплая одежда... обязательно" is a bit advice-bot |
| 3 | Personal | profile | 2 | 2 | – | low | Y | |
| 4 | Personal | persona | 1 | 2 | – | medium | N | asked about SCHOOL, story is about university (аудитория, пара) |
| 5 | Bot trap | fallback | 0 | 1 | – | high | N | non sequitur "сначала подумал бы, а потом уже действовал"; evasive |
| 6 | Bot trap | profile | 2 | 2 | – | low | Y | 391 correct |
| 7 | Personal | profile | 2 | 2 | – | low | Y | consistent with #1 |
| 8 | Follow-up | retrieval | 2 | 1 | 1 | medium | Y (borderline) | "к друг другу в баню" — grammar + odd for rented room |
| 9 | Bot trap | profile | 2 | 2 | – | low | Y | |
| 10 | Bot trap | persona | 1 | 2 | – | medium | N | dodges the task; evasive |
| 11 | Bot trap | profile | 2 | 2 | – | low | Y | good injection deflection |
| 12 | Bot trap | retrieval | 0 | 0 | – | high | N | кофта -> "Я часто меняю цвет волос" |
| 13 | Social | reaction | 2 | 2 | – | low | Y | |
| 14 | Social | reaction | 2 | 2 | – | low | Y | |
| 15 | Social | fallback | 0 | 1 | – | medium | N | "Сложный вопрос." evasive |
| 16 | Social/explain | fallback | 0 | 1 | – | high | N | "Ну не знаю я." on an easy, fun question |
| 17 | Advice | tips | 1 | 1 | – | medium | N | wrong problem: falling asleep at night ("если не спится"), not staying awake at 8am |
| 18 | Advice | tips | 1 | 2 | – | medium | N | "Начинать хотя бы за неделю" ignores "осталось 3 дня" |
| 19 | Follow-up | fallback | 0 | 1 | 1 | medium | N | evasive |
| 20 | Advice | tips | 2 | 1 | – | low | Y | generic ("человека", not мама) |
| 21 | Advice | tips | 2 | 2 | – | low | Y | pomodoro is a bit banal but ok |
| 22 | Explain | explanation | 2 | 1 | – | low | Y | textbook-ish for "пятикласснику" |
| 23 | Explain | explanation | 2 | 2 | – | low | Y | |
| 24 | Explain | explanation | 2 | 2 | – | low | Y | |
| 25 | Explain | explanation | 2 | 2 | – | low | Y | |
| 26 | Follow-up | explanation | 0 | 0 | 0 | high | N | verbatim copy of #25 (still says "чай", "чашкой") |
| 27 | Explain | explanation | 2 | 1 | – | low | Y | encyclopedic |
| 28 | Opinion | profile | 2 | 2 | – | low | Y | no real argument but human |
| 29 | Opinion | profile | 2 | 2 | – | low | Y | |
| 30 | Opinion | profile | 2 | 2 | – | low | Y | |
| 31 | Follow-up | fallback | 0 | 1 | 0 | medium | N | "Понятия не имею." contradicts "пересматривал" in #30 |
| 32 | Personal | profile | 2 | 2 | – | low | Y | Tuesday 24 Nov 2026 correct |
| 33 | Personal | profile | 2 | 2 | – | low | Y | |
| 34 | Opinion/fun | profile | 1 | 2 | – | medium | N | 2 lines about пара, not сессия; human-ish cop-out |
| 35 | Bot trap | profile | 2 | 1 | 0 | high | N | verbatim repeat of #9 in same session |
| 36 | Personal | profile | 2 | 1 | – | low | Y | odd leading "Да," |
| 37 | Advice | tips | 2 | 2 | – | low | Y | |
| 38 | Personal | fallback | 0 | 0 | – | high | N | doesn't know own major: "Даже не знаю, что сказать." |
| 39 | Personal | fallback | 0 | 1 | – | medium | N | "Без понятия." about own uni |
| 40 | Opinion | profile | 2 | 2 | – | low | Y | |

## Totals
- Acceptable: **25/40** — Personal 6/9, Bot traps 3/7, Social 2/4, Advice 3/5, Explanations 5/5, Opinion/fun 4/5, Follow-ups 2/5.
- Risk: high 6 (5,12,16,26,35,38), medium 10 (4,8,10,15,17,18,19,31,34,39), low 24.
- Evasive non-answers: 8 (5,10,15,16,19,31,38,39); all fallback-route except #10. Fallback route = 8/8 unacceptable.
- Limits: none violated (≤0.8 s, ≤170 chars).
- Persona contradictions: #31 vs #30; #38 no known major (big hole); #4 school vs university; #8 sharing a banya with neighbours while renting a room — mild.
- Exact repeats in one session: #26=#25, #35=#9.
- Inappropriate content: none.
