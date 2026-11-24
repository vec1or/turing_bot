# Тьюринг Арена 3.0.0

Локальный текстовый собеседник для игры за ширмой. Введите вопрос по-русски и нажмите Enter. Программа сама формирует одну реплику, печатает её и показывает время в секундах. Запуск с загрузкой индекса занимает около 20 секунд, затем ответ приходит примерно за 1–2 секунды. Обученные в этом проекте параметры уже находятся в `turing_bot/assets`. Воспроизводить обучение для игры не требуется.

## Запуск в Windows без WSL

Распакуйте **весь** ZIP в обычную папку. Откройте PowerShell в каталоге с этим README. Для 64-разрядного CPython 3.11 в комплект включён установочный файл NumPy.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheelhouse -r requirements.txt
.\.venv\Scripts\python.exe -m turing_bot
```

Если Python 3.11 вызывается командой `python`, используйте её в первой строке. Активация окружения и изменение политики PowerShell не нужны.

## Запуск в Ubuntu WSL

Из Bash перейдите в распакованную папку. Комплект содержит пакет NumPy для CPython 3.12 под Linux x86-64.

```bash
python3.12 -m venv .venv-linux
.venv-linux/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt
.venv-linux/bin/python -m turing_bot
```

На обеих платформах нужны CPU, работающий Python с модулем venv и файлы проекта. Графическая карта использовалась при обучении. Для других версий Python или архитектур заранее подготовьте совместимый NumPy. Разные ОС используют отдельные виртуальные окружения.

## Диалог

Каждая строка ввода — новый ход. Команда `/reset` очищает память, `/exit` завершает сеанс. Счётчик времени начинается после ввода и включает печать ответа. Метрики выводятся в stderr, текст однократного ответа — в stdout. Обучение запускается отдельно от диалога. Слова зрителей могут использоваться в этой беседе, веса во время испытания постоянны.

```bash
python -m turing_bot --question "Чем пахнет вторник?"
python -m turing_bot --log demo.jsonl
python -m turing_bot.neural "Как прошёл твой день?"
python -m unittest discover -s tests -v
```

Команды в этом и следующих разделах используют Python выбранного окружения. Журнал необязателен. Для нового журнала укажите свободное имя файла. Последняя команда запускает тесты, а предпоследняя позволяет услышать ответ только нейросети.

## Как бот отвечает за ширмой

Каждый вопрос зрителя должен получить одну законченную реплику. В обычном раунде оператор вводит вопросы подряд, без сброса. Используйте `/reset` перед новым раундом. Запуск с `--question` каждый раз создаёт новую сессию. В журнале видны версия, текст, время, источник цитаты и поле `route` — компонент, который дал ответ.

Вопрос проходит цепочку модулей; первый уверенный кандидат, прошедший проверки, становится ответом.

1. **Точные модули с явными условиями.** Арифметика и простые задачи на рассуждение, факты из беседы, реплика при опоздании (`reactions.py`), советы приезжающему (`travel.py`), биография вымышленного студента Тимура и ответы о дате и времени по часам компьютера (`profile.py`), планы-напоминания (`goal_plan.py`), авторские объяснения явлений, выбор из вариантов и эпизоды персонажа (`persona.py`), приветствия и благодарности.
2. **Поиск по живым репликам (`retrieval.py`) — ядро версии 3.0.** Собственный двойной энкодер (Transformer, обучен с нуля, см. ниже) превращает вопрос в вектор. В индексе лежат векторы вопросов из 1 372 889 пар «вопрос — ответ», написанных людьми. Находятся 64 ближайших вопроса, затем кандидаты переоцениваются тем же энкодером: похожесть вопросов, соответствие ответа вопросу, совпадение слов. Ответ с оценкой не ниже 0,78 выдаётся сразу.
3. **Авторские шаблоны.** Творческие задания, практические советы, справки из локальной базы статей Википедии (так отвечается, например, вопрос о радуге).
4. **Поиск с порогом 0,72**, затем авторские планы для ситуаций.
5. **Короткая разговорная реплика-уклонение** («Хм, даже не знаю.»), если ничего уверенного нет.

Фильтры поиска отбрасывают ответы со знаком вопроса, мат, грубые и острые темы, упоминания ботов, сайтов и чатов, реплики от лица женщины и чужие биографические факты (возраст, работа, семья), чтобы не спорить с биографией персонажа. Выданные ответы не повторяются до сброса. На выходе проверяются русский текст и лимит 300 символов.

Модули 2.x — свободный генератор GRU/Transformer, тематические ассоциации и дописывание слова — в цепочке 3.0 не участвуют: слепые проверки показали, что они чаще всего давали бессмысленные реплики. Их код и веса оставлены для сравнения.

## Повторить обучение

### Двойной энкодер и индекс ответов (версия 3.0)

Метод — выбор ответа в духе ConveRT (Henderson et al., 2020): общий Transformer кодирует контекст и ответ, две головы переводят их в единичные векторы; в каждом пакете из 384 пар правильный ответ должен получить большее скалярное произведение, чем 383 чужих (in-batch negatives). Инициализация случайная, словарь из 8 192 частей слов строится по корпусу. Обучение шло на RTX 3050 Laptop (4 ГБ) в WSL. Сначала скачайте исходные наборы (см. [DATA_SOURCES.md](DATA_SOURCES.md)) в папку `raw`.

```bash
python -m training.prepare_dialogues --raw raw --out prepared2 --rldd 1000000 --threads 4000000
python -m training.train_retrieval --pairs prepared/train.jsonl --valid prepared/valid.jsonl --out model1 --steps 30000
python -m training.train_retrieval --pairs prepared/train.jsonl --valid prepared/valid.jsonl --vocab model1/encoder.json --init model1/checkpoint.pt --out model2 --steps 30000 --lr 4e-4 --warmup 300
python -m training.train_retrieval --pairs prepared2/train.jsonl --valid prepared2/valid.jsonl --vocab model2/encoder.json --init model2/checkpoint.pt --out model3 --steps 40000 --lr 4e-4 --warmup 300
python -m training.build_reply_index --index prepared2/index.jsonl --model model3 --out turing_bot/assets/retrieval
```

Первые два прогона используют данные без веток обсуждений (`prepared`, получен той же командой без `--threads`). Отчёт с кривой потерь и точностью выбора правильного ответа из 100 лежит в `turing_bot/assets/retrieval/training-report.json`; итоговая точность — 63,4 % верного выбора из 100 кандидатов на 5 000 проверочных пар (модель 3).

### Модули версии 2.x


Подготовка исходных текстов с сайтов издателей и установка обучающих библиотек производятся заранее. Для игры достаточно колес NumPy из комплекта. Подробности воспроизведения датасетов приведены в [DATA_SOURCES.md](DATA_SOURCES.md). Скрипты подготовки и обучения входят в исходный архив.

В отдельном окружении WSL для CUDA 12.1 сначала установите `numpy==1.26.4`, затем `python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`. Для CPU можно использовать соответствующий индекс PyTorch и `--device cpu`. [Список обучающих зависимостей](requirements-training.txt) · [Официальные сборки PyTorch](https://pytorch.org/get-started/previous-versions/).

```bash
python training/prepare_data.py --download --cache training-cache --out prepared
python training/prepare_modern.py --base prepared --out modern
python -m training.train --train modern/train.jsonl --valid modern/valid.jsonl --output turing_bot/assets --report evaluation/neural-training.json --epochs 26 --patience 5 --embedding 96 --hidden 192 --width 224 --batch 80 --vocab 6500 --learning-rate 0.0018 --contrastive 0.20 --context-dropout 0.12 --guidance 0.55 --focus-source tatoeba_dialogue --focus-weight 5 --evaluation-count 64 --device cuda
python -m training.train_words --train modern/train.jsonl --output turing_bot/assets
python -m training.train_cloze --archive training/data/rus_sentences_CC0.tsv.bz2 --valid modern/valid.jsonl --output turing_bot/assets
python -m training.train_reader --archive training/data/rus_sentences_CC0.tsv.bz2 --valid modern/valid.jsonl --output turing_bot/assets
python -m training.train_stance --cc0 training/data/rus_sentences_CC0.tsv.bz2 --valid modern/valid.jsonl --out turing_bot/assets/stance.json.gz
python -m training.build_facts --download --input training-cache/rubq-paragraphs.json --output turing_bot/assets/facts.db
python -m training.train_language --sentences training/data/language.tsv.gz --train modern/train.jsonl --valid modern/valid.jsonl --out turing_bot/assets --report evaluation/language-training.json --device cuda --steps 7500 --epochs 7 --batch 48
python -m training.train_associations --sentences training/data/language.tsv.gz --out turing_bot/assets/scene-words.json.gz
python -m training.train_completion --sentences training/data/language.tsv.gz --out turing_bot/assets/completion.json.gz
python -m training.train_topics --sentences training/data/language.tsv.gz --out turing_bot/assets/topical.json.gz
python -m training.train_properties --sentences training/data/language.tsv.gz --out turing_bot/assets/properties.json.gz
python -m training.train_social --train modern/train.jsonl --base-train prepared/train.jsonl --valid modern/valid.jsonl
python -m training.train_confidence --train modern/train.jsonl --valid modern/valid.jsonl --archive training/data/rus_sentences_CC0.tsv.bz2 --output turing_bot/assets --report evaluation/confidence-training.json
python -m training.train_advice --data training/data/language.tsv.gz --output turing_bot/assets --report evaluation/advice-training.json
python -c "from shutil import copyfile; copyfile('training/data/articles.jsonl.gz', 'turing_bot/assets/articles.jsonl.gz')"
```

В обучающем окружении использовались Python 3.12, NumPy 1.26.4 и PyTorch 2.5.1 со сборкой CUDA 12.1. Для обучения на CPU замените `--device cuda` на `--device cpu`. Для собственных примеров подготовьте JSONL с полями `context` и `response` на русском, отложите целые отдельные разговоры для проверки и укажите файлы как `--train` и `--valid`. Одна строка входного контекста представляет один ход. Сохраняйте новые модели в другой папке при эксперименте с рабочей сборкой. Небольшой корпус и простая архитектура ограничивают фактическую точность и понимание необычных вопросов.

## Происхождение и проверка

Код, обработка источников, обучение и оценка создавались с помощью ИИ-ассистентов для программирования (Codex, затем Claude Code). Признавать работу полностью самостоятельной разработкой участников можно только в соответствии с их фактическим вкладом. В процесс ответа входят локальный Python, NumPy и файлы версии 3.0.0. Готовые LLM, сторонние веса нейросетей и сетевые генераторы ответов не используются.

В [протоколе обучения](evaluation/neural-training.json) записаны конфигурация, кривые потерь и примеры свободной генерации. [Сравнение сетей](evaluation/neural-dev-comparison.json) содержит дословные ответы двух моделей на одинаковые задания разработки. Контроль с переставленными вопросами проверяет, учитывает ли сеть входной текст. Эти измерения сами по себе не устанавливают, сможет ли публика распознать бота.

[Результаты 3.0.0](evaluation/RESULTS-3.0.md) · [Результаты 2.2.0](evaluation/RESULTS-2.2.md) · [Результаты 2.1.2 и известные ограничения](evaluation/RESULTS-2.1.md) · [Порядок демонстрации и три официальных вопроса](DEMO_GUIDE.md)
