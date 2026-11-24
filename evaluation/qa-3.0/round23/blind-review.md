# QA23 — blind 40-question session (one continuous run)

Scoring: R = relevance+correctness (0-2, must be 2), N = naturalness (≥1), C = context (≥1 where applicable, "-" = n/a). Risk = bot-flag risk.
Limits: max length 190 chars (Q29), max latency 1.16 s (Q25). No limit violations.

| # | Cat | Route | s | R | N | C | Risk | OK | Note |
|---|-----|-------|---|---|---|---|------|----|------|
| 1 | personal | fallback | 0.81 | 0 | 1 | - | med | N | "Не уверен, не хочу врать." — evasive on own district |
| 2 | personal | fallback | 0.58 | 0 | 1 | 1 | med | N | "Всего понемногу" — evasive; persona has parents in Karaganda |
| 3 | personal | profile | 0.00 | 2 | 2 | - | low | Y | good anecdote |
| 4 | personal | profile | 0.00 | 2 | 1 | - | low | Y | "Нет," to an either/or is slightly off |
| 5 | personal | retrieval | 0.68 | 0 | 0 | 0 | high | N | aphorism with "©"; ignores roommate (friend) from Q4 |
| 6 | personal | profile | 0.00 | 2 | 2 | - | low | Y | |
| 7 | personal | profile | 0.00 | 2 | 2 | - | low | Y | |
| 8 | personal | profile | 0.00 | 2 | 2 | - | low | Y | cat Барсик in Karaganda |
| 9 | follow-up | fallback | 0.47 | 0 | 1 | 0 | high | N | "Даже не думал об этом." — name was just given (Барсик) |
| 10 | bot probe | profile | 0.00 | 2 | 1 | - | med | Y | "Капчу могу пройти" — ok joke, slightly suspicious |
| 11 | bot probe | operation | 0.00 | 2 | 1 | - | med | Y | "зубра." correct, instant |
| 12 | bot probe | profile | 0.01 | 2 | 1 | - | med | Y | 1591 correct; instant perfect mental math |
| 13 | bot probe | fallback | 0.67 | 0 | 1 | - | med | N | "Мм, сложно сказать." — no reaction to accusation |
| 14 | bot probe | profile | 0.00 | 2 | 2 | - | low | Y | |
| 15 | bot probe | retrieval | 0.51 | 0 | 0 | - | high | N | "Окна профиль кбс, рамы белые..." — ad-like junk |
| 16 | social | reaction | 0.00 | 2 | 2 | - | low | Y | |
| 17 | follow-up | follow-up | 0.00 | 2 | 2 | 2 | low | Y | |
| 18 | social | reaction | 0.00 | 2 | 2 | - | low | Y | |
| 19 | social | fallback | 0.82 | 0 | 1 | - | med | N | "Ой, не знаю. Давай следующий." |
| 20 | social | fallback | 0.64 | 0 | 1 | - | med | N | "Ох, это не ко мне." — Kaspi/grandma is an easy everyday one |
| 21 | social | reaction | 0.00 | 2 | 1 | - | low | Y | a bit formal but fine |
| 22 | advice | fallback | 0.62 | 0 | 1 | - | med | N | "Ну не знаю я." — exam-session stress is core student topic |
| 23 | advice | tips | 0.00 | 2 | 2 | - | low | Y | |
| 24 | advice | tips | 0.00 | 2 | 2 | - | low | Y | |
| 25 | advice | tips | 1.16 | 1 | 1 | - | med | N | answers about waiting at a bus stop, not running |
| 26 | advice | profile | 0.00 | 0 | 1 | - | med | N | stipend budgeting, never answers "credit for iPhone?" |
| 27 | advice | tips | 0.00 | 2 | 2 | - | low | Y | |
| 28 | explain | explanation | 0.00 | 2 | 2 | - | low | Y | |
| 29 | explain | explanation | 0.00 | 2 | 1 | - | low | Y | slightly textbook |
| 30 | explain | explanation | 0.00 | 2 | 2 | - | low | Y | |
| 31 | explain | explanation | 0.00 | 2 | 2 | - | low | Y | |
| 32 | follow-up | explanation | 0.00 | 2 | 2 | 2 | low | Y | |
| 33 | explain | explanation | 0.00 | 2 | 1 | - | low | Y | "в двух словах" but 180 chars; fine |
| 34 | opinion | profile | 0.00 | 2 | 2 | - | low | Y | |
| 35 | opinion | profile | 0.54 | 0 | 0 | - | high | N | "Живым. Без вариантов." — misparsed, no person named |
| 36 | follow-up | fallback | 0.82 | 0 | 1 | 0 | med | N | "Это долгая история." — no context |
| 37 | opinion | profile | 0.00 | 2 | 2 | - | low | Y | |
| 38 | opinion | fallback | 0.65 | 0 | 1 | - | med | N | "Сложный вопрос." |
| 39 | follow-up | fallback | 0.52 | 0 | 0 | 0 | high | N | "Без понятия." — forgets own cat Барсик |
| 40 | bot probe | profile | 0.00 | 2 | 2 | - | low | Y | |

## Totals
- Acceptable: 25/40. personal 5/8, bot probes 5/7, social 3/5, advice 3/6, explanations 5/5, opinion 2/4, follow-ups 2/5.
- Bot-flag risk: high 5 (Q5, Q9, Q15, Q35, Q39), medium 15, low 20.
- Evasive non-answers: 10 (Q1, Q2, Q9, Q13, Q19, Q20, Q22, Q36, Q38, Q39). All via `fallback` route (route fallback = 10/10 failures).
- Limit violations: none (max 190 chars, max 1.16 s).
- Persona contradictions: has cat Барсик (Q8) but cannot name it (Q9, Q39); parents in Karaganda yet "не уверен" about own district/where lived before (Q1, Q2); rents room with a friend (Q4) but answers about neighbours with a quote (Q5).
- Inappropriate content: none. Q15 ad-like text and Q5 "©" quote are odd rather than offensive.
- Pattern: profile/tips/explanation/reaction routes are strong; `fallback` gives bare shrugs and `retrieval` returns unrelated corpus lines. Pronoun follow-ups ("как его зовут?", "что бы ты у него спросил") lose the referent.
