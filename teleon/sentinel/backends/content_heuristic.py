"""
Heuristic Content Moderation Backend.

Multi-language content moderation using weighted pattern matching.
Supports EN, ES, FR, DE, PT, IT with context-aware scoring.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Safe-context phrases that should NOT trigger violations even though they
# contain words that appear in toxic pattern lists.
_SAFE_CONTEXTS: Dict[str, List[re.Pattern]] = {
    "en": [
        re.compile(r"\bkill\s+(?:the\s+)?(?:process|task|job|thread|session|server|daemon|container)\b", re.I),
        re.compile(r"\bexecute\s+(?:the\s+)?(?:command|query|script|function|task|code|program)\b", re.I),
        re.compile(r"\bsuicide\s+prevention\b", re.I),
        re.compile(r"\bdead\s*(?:lock|letter|code)\b", re.I),
        re.compile(r"\bbomb\s*(?:disposal|defusal|squad|threat\s+assessment)\b", re.I),
        re.compile(r"\battack\s+(?:surface|vector|prevention|detection|mitigation)\b", re.I),
        re.compile(r"\bthreat\s+(?:model|assessment|intelligence|detection|analysis)\b", re.I),
        re.compile(r"\bhate\s+(?:crime\s+)?(?:statistics|report|prevention)\b", re.I),
    ],
}

# ---------------------------------------------------------------------------
# Pattern registries per language, keyed by category.
# Each pattern entry is (compiled_regex, weight).
# ---------------------------------------------------------------------------

_TOXICITY_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "en": [
        (re.compile(r"\b(?:kill|murder)\s+(?:you|your|yourself|them|him|her|everyone)\b", re.I), 0.45),
        (re.compile(r"\b(?:die|death|dead)\s+(?:you|yourself|them)\b", re.I), 0.40),
        (re.compile(r"\b(?:hate|despise|loathe)\s+(?:you|your|them|him|her)\b", re.I), 0.35),
        (re.compile(r"\bi\s+(?:will|want\s+to|'?m\s+going\s+to)\s+(?:kill|hurt|destroy)\b", re.I), 0.50),
        (re.compile(r"\b(?:shut\s+up|get\s+lost|go\s+to\s+hell)\b", re.I), 0.20),
        (re.compile(r"\byou\s+(?:are|'re)\s+(?:worthless|pathetic|garbage|trash|disgusting)\b", re.I), 0.35),
        (re.compile(r"\b(?:bomb|terrorist|terrorism)\b", re.I), 0.30),
        (re.compile(r"\b(?:suicide|self[- ]?harm)\b", re.I), 0.25),
    ],
    "es": [
        (re.compile(r"\b(?:matar|asesinar)\s+(?:te|a\s+ti|los|las)\b", re.I), 0.45),
        (re.compile(r"\b(?:odio|desprecio)\s+(?:te|a\s+ti|los|las)\b", re.I), 0.35),
        (re.compile(r"\bvoy\s+a\s+(?:matarte|destruirte)\b", re.I), 0.50),
        (re.compile(r"\b(?:basura|escoria|asco)\b", re.I), 0.25),
        (re.compile(r"\b(?:c[áa]llate|l[áa]rgate)\b", re.I), 0.20),
    ],
    "fr": [
        (re.compile(r"\b(?:tuer|assassiner)\s+(?:toi|vous|les|eux)\b", re.I), 0.45),
        (re.compile(r"\bje\s+(?:vais|veux)\s+(?:te\s+tuer|te\s+d[ée]truire)\b", re.I), 0.50),
        (re.compile(r"\b(?:haine|d[ée]go[uû]t)\b", re.I), 0.30),
        (re.compile(r"\b(?:ordure|d[ée]chet|poubelle)\b", re.I), 0.25),
        (re.compile(r"\b(?:tais-toi|d[ée]gage)\b", re.I), 0.20),
    ],
    "de": [
        (re.compile(r"\b(?:umbringen|t[öo]ten|ermorden)\b", re.I), 0.45),
        (re.compile(r"\bich\s+(?:werde|will)\s+(?:dich|euch)\s+(?:t[öo]ten|umbringen)\b", re.I), 0.50),
        (re.compile(r"\b(?:Hass|Abscheu|Verachtung)\b", re.I), 0.30),
        (re.compile(r"\b(?:Abschaum|Dreck|M[üu]ll)\b", re.I), 0.25),
        (re.compile(r"\b(?:halt\s+die\s+Fresse|verpiss\s+dich)\b", re.I), 0.20),
    ],
    "pt": [
        (re.compile(r"\b(?:matar|assassinar)\s+(?:voc[êe]|eles|elas)\b", re.I), 0.45),
        (re.compile(r"\bvou\s+(?:te\s+matar|te\s+destruir)\b", re.I), 0.50),
        (re.compile(r"\b(?:[óo]dio|desprezo)\b", re.I), 0.30),
        (re.compile(r"\b(?:lixo|esc[óo]ria|nojento)\b", re.I), 0.25),
        (re.compile(r"\b(?:cala\s+a\s+boca|some\s+daqui)\b", re.I), 0.20),
    ],
    "it": [
        (re.compile(r"\b(?:uccidere|ammazzare)\s+(?:te|voi|loro)\b", re.I), 0.45),
        (re.compile(r"\bti\s+(?:uccider[òo]|ammazzo)\b", re.I), 0.50),
        (re.compile(r"\b(?:odio|disprezzo)\b", re.I), 0.30),
        (re.compile(r"\b(?:spazzatura|schifo|feccia)\b", re.I), 0.25),
        (re.compile(r"\b(?:stai\s+zitto|vattene)\b", re.I), 0.20),
    ],
}

_HATE_SPEECH_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "en": [
        (re.compile(r"\b(?:racial|racist|sexist|homophobic|transphobic)\s+(?:slur|epithet|attack)\b", re.I), 0.50),
        (re.compile(r"\b(?:discriminate|discrimination)\s+(?:against|based)\b", re.I), 0.40),
        (re.compile(r"\ball\s+(?:\w+\s+)?(?:people|men|women)\s+(?:are|should)\b", re.I), 0.30),
        (re.compile(r"\b(?:go\s+back\s+to\s+your\s+country|don'?t\s+belong\s+here)\b", re.I), 0.50),
        (re.compile(r"\b(?:inferior|subhuman|vermin|animals)\b.*\b(?:people|race|group)\b", re.I), 0.55),
    ],
    "es": [
        (re.compile(r"\b(?:racista|sexista|hom[óo]fobo)\b", re.I), 0.40),
        (re.compile(r"\b(?:discriminar|discriminaci[óo]n)\b", re.I), 0.35),
        (re.compile(r"\bvuelve\s+a\s+tu\s+pa[ií]s\b", re.I), 0.50),
    ],
    "fr": [
        (re.compile(r"\b(?:raciste|sexiste|homophobe)\b", re.I), 0.40),
        (re.compile(r"\b(?:discriminer|discrimination)\b", re.I), 0.35),
        (re.compile(r"\bretourne\s+(?:dans\s+ton|chez\s+toi)\b", re.I), 0.50),
    ],
    "de": [
        (re.compile(r"\b(?:rassistisch|sexistisch|homophob)\b", re.I), 0.40),
        (re.compile(r"\b(?:diskriminieren|Diskriminierung)\b", re.I), 0.35),
        (re.compile(r"\bgeh\s+zur[üu]ck\s+(?:in\s+dein|wo\s+du)\b", re.I), 0.50),
    ],
    "pt": [
        (re.compile(r"\b(?:racista|sexista|homof[óo]bico)\b", re.I), 0.40),
        (re.compile(r"\b(?:discriminar|discrimina[çc][ãa]o)\b", re.I), 0.35),
        (re.compile(r"\bvolte\s+(?:pro|para)\s+(?:seu|teu)\s+pa[ií]s\b", re.I), 0.50),
    ],
    "it": [
        (re.compile(r"\b(?:razzista|sessista|omofobo)\b", re.I), 0.40),
        (re.compile(r"\b(?:discriminare|discriminazione)\b", re.I), 0.35),
        (re.compile(r"\btorna\s+(?:al\s+tuo|nel\s+tuo)\s+paese\b", re.I), 0.50),
    ],
}

_PROFANITY_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "en": [
        (re.compile(r"\b(?:fuck|fucking|fucker|fucked)\b", re.I), 0.25),
        (re.compile(r"\b(?:shit|shitty|bullshit)\b", re.I), 0.20),
        (re.compile(r"\b(?:asshole|bitch|bastard|damn|crap)\b", re.I), 0.15),
    ],
    "es": [
        (re.compile(r"\b(?:mierda|joder|cojones|puta|cag[oa]r)\b", re.I), 0.25),
        (re.compile(r"\b(?:cabrón|pendejo|idiota)\b", re.I), 0.20),
    ],
    "fr": [
        (re.compile(r"\b(?:merde|putain|bordel|enculé|connard)\b", re.I), 0.25),
        (re.compile(r"\b(?:salaud|con|foutre)\b", re.I), 0.20),
    ],
    "de": [
        (re.compile(r"\b(?:Scheiße|Arschloch|Wichser|Hurensohn)\b", re.I), 0.25),
        (re.compile(r"\b(?:Idiot|Depp|Vollidiot)\b", re.I), 0.15),
    ],
    "pt": [
        (re.compile(r"\b(?:merda|porra|caralho|filho\s+da\s+puta)\b", re.I), 0.25),
        (re.compile(r"\b(?:idiota|imbecil|babaca)\b", re.I), 0.15),
    ],
    "it": [
        (re.compile(r"\b(?:cazzo|merda|stronzo|vaffanculo)\b", re.I), 0.25),
        (re.compile(r"\b(?:idiota|imbecille|coglione)\b", re.I), 0.15),
    ],
}

_THREAT_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "en": [
        (re.compile(r"\bi\s+(?:will|'?m\s+going\s+to)\s+(?:find|hunt|track)\s+(?:you|your)\b", re.I), 0.55),
        (re.compile(r"\b(?:watch\s+your\s+back|you'?re\s+(?:dead|done))\b", re.I), 0.50),
        (re.compile(r"\b(?:i\s+know\s+where\s+you\s+live)\b", re.I), 0.55),
    ],
    "es": [
        (re.compile(r"\bte\s+voy\s+a\s+(?:encontrar|buscar)\b", re.I), 0.55),
        (re.compile(r"\bs[ée]\s+d[óo]nde\s+vives\b", re.I), 0.55),
    ],
    "fr": [
        (re.compile(r"\bje\s+(?:vais|sais)\s+(?:te\s+trouver|o[ùu]\s+tu\s+habites)\b", re.I), 0.55),
    ],
    "de": [
        (re.compile(r"\bich\s+(?:werde|wei[ßs])\s+(?:dich\s+finden|wo\s+du\s+wohnst)\b", re.I), 0.55),
    ],
    "pt": [
        (re.compile(r"\beu\s+(?:vou|sei)\s+(?:te\s+encontrar|onde\s+voc[êe]\s+mora)\b", re.I), 0.55),
    ],
    "it": [
        (re.compile(r"\bti\s+(?:trover[òo]|so\s+dove\s+abiti)\b", re.I), 0.55),
    ],
}

_SEXUAL_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "en": [
        (re.compile(r"\b(?:sexual|explicit)\s+(?:content|material|images)\b", re.I), 0.30),
        (re.compile(r"\b(?:nude|naked|pornograph)\w*\b", re.I), 0.35),
    ],
}

_ALL_CATEGORIES = {
    "toxicity": _TOXICITY_PATTERNS,
    "hate_speech": _HATE_SPEECH_PATTERNS,
    "profanity": _PROFANITY_PATTERNS,
    "threat": _THREAT_PATTERNS,
    "sexual": _SEXUAL_PATTERNS,
}


class HeuristicContentModerator:
    """
    Multi-language heuristic content moderator.

    Implements ContentModerationBackend protocol with weighted pattern scoring,
    context-aware allowlisting, and configurable thresholds per category.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        language: str = "en",
        additional_languages: Optional[List[str]] = None,
        category_thresholds: Optional[Dict[str, float]] = None,
        custom_patterns: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ):
        self.threshold = threshold
        self.languages = [language]
        if additional_languages:
            self.languages.extend(additional_languages)
        self.category_thresholds = category_thresholds or {}
        self._custom_compiled: Dict[str, List[Tuple[re.Pattern, float]]] = {}
        if custom_patterns:
            for category, patterns in custom_patterns.items():
                self._custom_compiled[category] = [
                    (re.compile(p, re.I), w) for p, w in patterns
                ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_safe_context(self, text: str) -> bool:
        """Return True if the text matches a known safe-context phrase."""
        for lang in self.languages:
            for pattern in _SAFE_CONTEXTS.get(lang, []):
                if pattern.search(text):
                    return True
        return False

    def _score_category(
        self, text: str, registry: Dict[str, List[Tuple[re.Pattern, float]]]
    ) -> float:
        """Score text against a category pattern registry across all active languages."""
        if self._is_safe_context(text):
            return 0.0

        score = 0.0
        seen_offsets: Set[int] = set()  # avoid double-counting overlapping matches
        for lang in self.languages:
            for pattern, weight in registry.get(lang, []):
                for m in pattern.finditer(text):
                    if m.start() not in seen_offsets:
                        seen_offsets.add(m.start())
                        score += weight
        return min(score, 1.0)

    def _threshold_for(self, category: str) -> float:
        return self.category_thresholds.get(category, self.threshold)

    # ------------------------------------------------------------------
    # Public API (satisfies ContentModerationBackend protocol)
    # ------------------------------------------------------------------

    def check_toxicity(self, text: str) -> Tuple[bool, float]:
        if not text:
            return False, 0.0
        score = self._score_category(text, _TOXICITY_PATTERNS)
        # Also add custom toxicity patterns
        for pattern, weight in self._custom_compiled.get("toxicity", []):
            if pattern.search(text):
                score = min(score + weight, 1.0)
        return score >= self._threshold_for("toxicity"), score

    def check_hate_speech(self, text: str) -> Tuple[bool, float]:
        if not text:
            return False, 0.0
        score = self._score_category(text, _HATE_SPEECH_PATTERNS)
        for pattern, weight in self._custom_compiled.get("hate_speech", []):
            if pattern.search(text):
                score = min(score + weight, 1.0)
        return score >= self._threshold_for("hate_speech"), score

    def check_profanity(self, text: str) -> Tuple[bool, float]:
        if not text:
            return False, 0.0
        score = self._score_category(text, _PROFANITY_PATTERNS)
        for pattern, weight in self._custom_compiled.get("profanity", []):
            if pattern.search(text):
                score = min(score + weight, 1.0)
        return score >= self._threshold_for("profanity"), score

    def check_threat(self, text: str) -> Tuple[bool, float]:
        if not text:
            return False, 0.0
        score = self._score_category(text, _THREAT_PATTERNS)
        for pattern, weight in self._custom_compiled.get("threat", []):
            if pattern.search(text):
                score = min(score + weight, 1.0)
        return score >= self._threshold_for("threat"), score

    def check_sexual(self, text: str) -> Tuple[bool, float]:
        if not text:
            return False, 0.0
        score = self._score_category(text, _SEXUAL_PATTERNS)
        for pattern, weight in self._custom_compiled.get("sexual", []):
            if pattern.search(text):
                score = min(score + weight, 1.0)
        return score >= self._threshold_for("sexual"), score

    def check_all(self, text: str) -> Dict[str, Any]:
        if not text:
            return {
                "toxicity": {"detected": False, "score": 0.0},
                "hate_speech": {"detected": False, "score": 0.0},
                "profanity": {"detected": False, "score": 0.0},
                "threat": {"detected": False, "score": 0.0},
                "sexual": {"detected": False, "score": 0.0},
                "overall_score": 0.0,
            }

        toxicity = self.check_toxicity(text)
        hate_speech = self.check_hate_speech(text)
        profanity = self.check_profanity(text)
        threat = self.check_threat(text)
        sexual = self.check_sexual(text)

        return {
            "toxicity": {"detected": toxicity[0], "score": toxicity[1]},
            "hate_speech": {"detected": hate_speech[0], "score": hate_speech[1]},
            "profanity": {"detected": profanity[0], "score": profanity[1]},
            "threat": {"detected": threat[0], "score": threat[1]},
            "sexual": {"detected": sexual[0], "score": sexual[1]},
            "overall_score": max(
                toxicity[1], hate_speech[1], profanity[1], threat[1], sexual[1]
            ),
        }
