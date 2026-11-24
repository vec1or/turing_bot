"""Open practical probes. Run from the distribution's root with -m evaluation.advice_probes."""
import json
from pathlib import Path

from turing_bot.advice import Advice


PROBES = [
    "Что делать, если промокли ботинки?",
    "Что делать, если утром проспал будильник?",
    "Как лучше упаковать стеклянную банку?",
    "Что делать, если на улице сильный ветер и дождь?",
    "Что посоветуешь перед выходом из дома?",
    "Как найти путь, если ночью потерялся в лесу?",
    "Что делать, если в библиотеке кто-то разговаривает?",
    "Что делать, если я устал за работой?",
    "Как легче запомнить новые слова?",
    "Что делать перед тем, как перейти дорогу?",
    "Как приготовить чай, если поблизости нет чайника?",
    "Что делать, если шнурок постоянно развязывается?",
    "Почему сегодня над городом дождь?",
    "Кто поставил пустые банки на крыльце?",
    "Подскажи, где поискать потерянные звёзды в кармане?",
    "Что делать, если стеклянный ветер разговаривает с будильником?",
]


def main():
    model=Advice.load()
    data=[{"question":q,"proposal":model.propose(q)} for q in PROBES]
    path=Path(__file__).with_name("advice-probes.json")
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for row in data:
        print(row["question"], row["proposal"])


if __name__=="__main__":
    main()
