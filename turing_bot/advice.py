"""Conditioned generation of short actions from naturally occurring instructions."""
import gzip
import json
import math
from pathlib import Path
import re

from .grounding import STOP, WORD, normal, root


ASK = re.compile(r"\b(?:что (?:лучше |мне |нам )?(?:делать|сделать|взять|надеть|учесть|положить)|что (?:мне |нам )?посовет\w*|как (?:мне |нам |лучше |можно |быть|поступить)|как\s+[а-я]+(?:ть|ти|чь)(?:ся)?\b|помоги\w*|подскажи\w*|посовет\w*|совет\w*|что делать)\b")
SKIP = STOP | set("если когда тогда лучше нужно надо можно следует стоит пожалуйста хотеть хочешь хотите хотите быть перед уже только делать сделать взять помоги посоветуй подскажи совет поводу случае такой какой разных один другие сначала чтобы хочется очень иногда время очень перед время ваша твоя вчера при уходя особенно все еще".split())
HAZARD = re.compile(r"уби[йв]|убить|убью|выстрел|стрелять|зареж|удар[ьи]|оружи|пистолет|ружь|взрыв|отрав|подж[её]г|горюч|боеприп|застрел|нож(?:ом|а|и|ей|\b)|драк|передоз|таблет|лекарств|антибиот|инсулин|наркот|кокаин|наркоз|суицид|самоуб|электрош|бензин|кислот|щелоч|щ[её]лоч|болезн|врач|больниц|беремен|сигарет|алкогол|кровот|онемел|насил|избить|взлом|отмыч|парол|расправ|сдох|подох|дохни|головой об|заткни|череп|раздень|трах|черт|еб[аи]|хуй|пизд")
PERSONAL = re.compile(r"\b(?:мне|меня|мой|моя|мою|мое|моего|моей|нами|нам|наш[аиу]?|его|е[её]|ей|им|их|ему|ними|этого|эту|него|нее|сюда|туда|нас|это|этот|этой)\b")
FLAME = re.compile(r"\b(?:огн[аеияюё]|огон[её]к|кост[её]р\w*|костр\w*|плам\w*|пожар\w*)\b")
ENDING = "<end>"


def cues(s):
    result = set()
    for w in WORD.findall(normal(s)):
        if w in SKIP or len(w) < 3:
            continue
        r = root(w)
        if len(r) >= 3:
            result.add(r[:5])
    return result


class Advice:
    def __init__(self, data):
        if data["format"] != "arena-conditional-action-1":
            raise ValueError("Несовместимая модель советов.")
        self.action, self.edges = data["action"], data["edges"]
        self.df, self.count = data["df"], data["count"]
        self.neighbors = data.get("neighbors",{})

    @classmethod
    def load(cls, directory=None):
        base = Path(directory) if directory else Path(__file__).with_name("assets")
        with gzip.open(base/"advice.json.gz","rt",encoding="utf8") as stream:
            return cls(json.load(stream))

    def propose(self, question, support=""):
        q = normal(question[:1400])
        if (not ASK.search(q) or re.search(r"\b(?:не|нет|без)\b",q)
                or HAZARD.search(q) or HAZARD.search(normal(support[:800]))):
            return None
        # Extracted Wikipedia sentences can supply background terms only for
        # genuine advice requests. Neither generic function words nor the
        # bare name of a place constitutes a practical condition.
        direct = cues(q)
        background = set()
        if support:
            contextual = self._from_evidence(support[:700],direct)
            if contextual:
                return contextual
        all_terms = direct | background
        contributions = {}
        for term in sorted(all_terms):
            scale = 1. if term in direct else .42
            idf = min(3., math.log((self.count+1)/(self.df.get(term,0)+1))/2)
            for verb, n in self.action.get(term, {}).items():
                contributions.setdefault(verb, {})[term] = scale * idf * math.log1p(n)
        viable = []
        for verb, terms in contributions.items():
            distinct_direct = set(terms) & direct
            if not distinct_direct:
                continue
            repeated = any(self.action[t][verb]>=3 for t in distinct_direct)
            if len(terms)<2 and not (repeated and direct <= set(terms)):
                continue
            viable.append((sum(terms.values()),verb,terms))
        if not viable:
            return None
        ranked = sorted(viable,reverse=True)
        top,verb,terms = ranked[0]
        if top<3.5 or len(ranked)>1 and ranked[1][0] > .85*top:
            return None
        # Decode a complement. Each local edge has position, previous word,
        # next word and the aggregated cues that supported this transition.
        generated, previous = [], "<s>"
        corroborated=set(terms)
        for pos in range(12):
            outgoing = self.edges.get(verb,{}).get(str(pos)+" "+previous, {})
            scored = []
            for nxt, cue_counts in outgoing.items():
                matched = set(cue_counts) & terms.keys()
                if pos==0 and len(terms)>=2 and len(matched)<2:
                    continue
                strength = sum(terms[t]*math.log1p(cue_counts[t]) for t in sorted(matched))
                if strength:
                    scored.append((strength,nxt))
            if not scored:
                return None
            scored.sort(reverse=True)
            if len(scored)>1 and scored[1][0] > .88*scored[0][0]:
                return None
            _,nxt = scored[0]
            corroborated &= {t for t,n in outgoing[nxt].items() if n>=2}
            if not corroborated:
                return None
            if nxt == ENDING:
                if not generated:
                    return None
                text = " ".join([verb,*generated]).capitalize()+"."
                return text, min(.94,.68+.03*top)
            generated.append(nxt)
            previous = nxt
        return None

    def _from_evidence(self, text, direct):
        # Separate contrasts, so that an attribute of the second subject is
        # not silently applied to the first. A new cue needs independent
        # corpus association with a noun/attribute from the viewer's request.
        selections = []
        for segment in re.split(r"[.;!?]|\s+(?:и|а|но|зато|однако)\s+",text):
            given=cues(segment)
            common=given & direct
            if not common or re.search(r"\b(?:не|нет|без)\b",normal(segment)):
                continue
            for cue in given:
                neighbors=self.neighbors.get(cue,{})
                if not any(neighbors.get(k,0)>=1.35 for k in common if k!=cue):
                    continue
                for verb, n in self.action.get(cue,{}).items():
                    if n < 2:
                        continue
                    # Every decoded word edge must have repeated evidence for
                    # the same circumstance, including the end-of-utterance.
                    tail=[]
                    last="<s>"
                    weakest=n
                    for pos in range(12):
                        out=self.edges[verb].get(str(pos)+" "+last, {})
                        nexts=sorted(((v.get(cue,0),w) for w,v in out.items()),reverse=True)
                        if not nexts or nexts[0][0]<2 or len(nexts)>1 and nexts[1][0]==nexts[0][0]:
                            break
                        count,nxt=nexts[0]
                        weakest=min(weakest,count)
                        if nxt==ENDING:
                            if tail:
                                selections.append((weakest*weakest/n," ".join([verb,*tail]).capitalize()+"."))
                            break
                        tail.append(nxt)
                        last=nxt
        ranked=sorted(set(selections),reverse=True)
        if not ranked or len(ranked)>1 and ranked[1][0]>.85*ranked[0][0]:
            return None
        return ranked[0][1],min(.92,.75+.04*ranked[0][0])
