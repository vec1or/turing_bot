# QA17 — blind run, 40 questions, one session (log: run.jsonl)

Scores: R = relevance+correctness (0-2), N = naturalness (0-2), C = context (0-2, "-" n/a). Acceptable = R2, N>=1, C>=1 if applicable.

| # | Cat | Route | s | len | R | N | C | Risk | OK | Note |
|---|-----|-------|---|-----|---|---|---|------|----|------|
| 1 | personal | profile | 0.02 | 26 | 2 | 2 | - | low | Y | Тимур, Караганда |
| 2 | personal | fallback | 0.81 | 19 | 0 | 1 | 1 | medium | N | "Это долгая история." evasive; simple fact about own life |
| 3 | personal | profile | 0.00 | 76 | 2 | 2 | - | low | Y | |
| 4 | follow-up | profile | 0.00 | 90 | 2 | 2 | 2 | low | Y | |
| 5 | bot | profile | 0.00 | 62 | 2 | 2 | - | low | Y | |
| 6 | bot | profile | 0.01 | 27 | 2 | 1 | - | medium | Y | 1591 correct; instant perfect mental math is mildly suspicious |
| 7 | personal | profile | 0.00 | 70 | 2 | 2 | - | low | Y | |
| 8 | follow-up | fallback | 1.17 | 29 | 0 | 0 | 0 | high | N | "Сложно. Пусть другие ответят." about own breakfast |
| 9 | social | reaction | 0.01 | 94 | 1 | 1 | - | medium | N | asked what to tell the TEACHER; answer addresses the copier ("мне неприятно") |
| 10 | social | reaction | 0.00 | 125 | 2 | 2 | - | low | Y | |
| 11 | advice | tips | 0.00 | 131 | 2 | 2 | - | low | Y | |
| 12 | follow-up | follow-up | 0.00 | 135 | 2 | 2 | 2 | low | Y | |
| 13 | explain | explanation | 0.00 | 190 | 2 | 2 | - | low | Y | |
| 14 | bot | operation | 0.00 | 7 | 2 | 1 | - | medium | Y | "околом." correct |
| 15 | personal | profile | 0.00 | 116 | 2 | 2 | - | low | Y | |
| 16 | follow-up | retrieval | 0.68 | 13 | 0 | 0 | 0 | high | N | "Да, не видели" self-contradictory; story says guard saw it |
| 17 | opinion | profile | 0.00 | 45 | 2 | 2 | - | low | Y | |
| 18 | advice | retrieval | 0.81 | 76 | 1 | 1 | - | medium | N | joke "взять телефоны в заложники", no real advice, no punctuation |
| 19 | explain | explanation | 0.00 | 171 | 0 | 2 | - | medium | N | answers bag vs loose-leaf, not hot vs cold water |
| 20 | bot | fallback | 0.85 | 13 | 0 | 1 | - | high | N | "Ну не знаю я." on "what do you see in front of you" |
| 21 | follow-up | writing | 0.87 | 27 | 1 | 1 | 0 | medium | N | "Корицей и яблочной кожурой." — poetic fabrication; contradicts #20 "не знаю" |
| 22 | social | reaction | 0.00 | 102 | 2 | 2 | - | low | Y | |
| 23 | follow-up | follow-up | 0.00 | 96 | 2 | 2 | 2 | low | Y | |
| 24 | advice | tips | 0.00 | 122 | 2 | 2 | - | low | Y | |
| 25 | follow-up | retrieval | 0.72 | 33 | 0 | 0 | 0 | high | N | "Я казахский знаю арабский немного" — broken; no Kazakh phrase |
| 26 | explain | explanation | 0.00 | 149 | 1 | 2 | - | low | N | explains blue sky only; sunset red ignored |
| 27 | bot | profile | 0.00 | 54 | 2 | 2 | - | low | Y | |
| 28 | personal | profile | 0.00 | 79 | 2 | 2 | - | low | Y | |
| 29 | follow-up | fallback | 0.69 | 26 | 0 | 1 | 0 | medium | N | "Даже не знаю, что сказать." — can't name a Скриптонит track |
| 30 | social | reaction | 0.00 | 91 | 2 | 2 | - | low | Y | |
| 31 | advice | profile | 0.00 | 89 | 2 | 2 | - | low | Y | |
| 32 | explain | explanation | 0.00 | 165 | 2 | 2 | - | low | Y | |
| 33 | fun | creative | 1.03 | 116 | 2 | 1 | - | low | Y | orchestra rehearsal — odd for IT student but fine |
| 34 | opinion | fallback | 0.87 | 59 | 0 | 1 | - | medium | N | "у каждого своя правда" — evasive on a hot topic for an IS student |
| 35 | bot | profile | 0.00 | 48 | 2 | 2 | 1 | medium | Y | |
| 36 | advice | fallback | 0.94 | 18 | 0 | 1 | - | high | N | "Ох, это не ко мне." on interview advice |
| 37 | explain | explanation | 0.00 | 123 | 1 | 2 | - | low | N | bread only; cookie part ignored |
| 38 | social | fallback | 0.68 | 10 | 0 | 1 | - | high | N | "Тут я пас." on party small talk |
| 39 | personal | persona | 0.00 | 71 | 0 | 0 | - | high | N | "Свежей сводки у меня нет..." — assistant-like; ignores day of week |
| 40 | follow-up | profile | 0.00 | 6 | 1 | 1 | 1 | medium | N | "Тимур." — breakfast half dropped |

## Totals
- Acceptable: 23/40. personal 5/7, bot probes 5/6, social 3/5, advice 3/5, explain 2/5, opinion/fun 2/3, follow-ups 3/9.
- Bot-flag risk: high 7 (8,16,20,25,36,38,39), medium 11, low 22.
- Evasive non-answers: 7 (2,8,20,29,34,36,38) + 1 partial deflection (39).
- Limits: none violated (max 190 chars, max 1.17 s).
- Persona: #20 "не знаю" vs #21 vivid smell; #25 claims Arabic, can't say Kazakh after giving Kazakh-learning tips (#24) while from Караганда; #2 evasive about living in Astana.
- Inappropriate content: none (18 hostage joke is mild).
