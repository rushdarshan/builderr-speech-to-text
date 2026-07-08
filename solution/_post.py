"""Post-decode text fixes: normalize number words to digits + cross-model fusion."""
import math
import re
from difflib import SequenceMatcher

from word2number import w2n

try:
    import hindi_normalize
    _HI_NORM_AVAIL = True
except ImportError:
    _HI_NORM_AVAIL = False

_TOKEN_RE = re.compile(r"[\w'-]+")

# ── Romanized Hindi number words (1-100) ──────────────────────────────
_HI_NUMS: dict[str, int] = {
    "shunya": 0, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4,
    "paanch": 5, "panch": 5, "cheh": 6, "chhah": 6, "saat": 7, "aath": 8,
    "nau": 9, "das": 10, "gyaarah": 11, "baarah": 12, "terah": 13,
    "chaudah": 14, "pandrah": 15, "solah": 16, "satrah": 17, "athaarah": 18,
    "unees": 19, "bees": 20, "ikkis": 21, "baais": 22, "teis": 23,
    "chaubees": 24, "pachees": 25, "chhabbees": 26, "sattaais": 27,
    "atthaais": 28, "untees": 29, "tees": 30, "iktees": 31, "batees": 32,
    "taintees": 33, "chauntees": 34, "pantees": 35, "chhattees": 36,
    "saintees": 37, "adtees": 38, "untaalees": 39, "chaalees": 40,
    "iktaalees": 41, "bayaalees": 42, "tentaalees": 43, "chavvaalees": 44,
    "pantaalees": 45, "chiyalees": 46, "santaalees": 47, "adtaalees": 48,
    "unchaas": 49, "pachaas": 50, "ikyavan": 51, "bavan": 52, "tirpan": 53,
    "chauvan": 54, "pachpan": 55, "chappan": 56, "sattavan": 57,
    "atthavan": 58, "unsath": 59, "saath": 60, "iksath": 61, "baysath": 62,
    "tirsath": 63, "chaunsath": 64, "painsath": 65, "chhiyansath": 66,
    "sadsath": 67, "arsath": 68, "unhattar": 69, "sattar": 70,
    "ikhattar": 71, "bahattar": 72, "tihattar": 73, "chauhattar": 74,
    "pachhattar": 75, "chhiyattar": 76, "satattar": 77, "atthattar": 78,
    "unyasi": 79, "assi": 80, "ikyaasi": 81, "bayaasi": 82, "tiraasi": 83,
    "chauraasi": 84, "pachchaasi": 85, "chhiyaasi": 86, "sataasi": 87,
    "atthaasi": 88, "navaasi": 89, "nabbe": 90, "nabbey": 90,
    "ikyaanave": 91, "baanave": 92, "tiraanave": 93, "chauraanave": 94,
    "pachchaavan": 95, "chhiyaanave": 96, "sataanave": 97, "atthaanave": 98,
    "ninayaanave": 99,
}

# ── Devanagari number words ──────────────────────────────────────────
_HI_DEV_NUMS: dict[str, int] = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकत्तीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चव्वालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छियहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

# ── Powers (multipliers) ─────────────────────────────────────────────
_HI_POWERS: dict[str, int] = {
    "sau": 100,
    "hazaar": 1000, "hazar": 1000, "hajaar": 1000,
    "laakh": 100000, "lakh": 100000,
    "karod": 10000000, "crore": 10000000,
}

_HI_DEV_POWERS: dict[str, int] = {
    "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "लाख": 100000, "करोड़": 10000000,
}

# Build combined lookup: Roman + Devanagari single words
_HI_ALL_SINGLE: dict[str, int] = {}
_HI_ALL_SINGLE.update(_HI_NUMS)
_HI_ALL_SINGLE.update(_HI_DEV_NUMS)

_HI_ALL_POWERS: dict[str, int] = {}
_HI_ALL_POWERS.update(_HI_POWERS)
_HI_ALL_POWERS.update(_HI_DEV_POWERS)


def _parse_hi_number_phrase(words: list[str]) -> int | None:
    """Parse a Hinglish number phrase like ['do', 'sau', 'pachaas'] → 250.

    Rules (left-to-right):
    - A unit word followed by a power word: unit × power → add to total
    - A unit word on its own: add to total
    - e.g. do sau → 2×100=200
    - e.g. do sau pachaas → 2×100 + 50 = 250
    - e.g. pachaas hazaar → 50×1000 = 50000
    """
    total = 0
    pending_unit = 0
    for w in words:
        w_lower = w.lower()
        if w_lower in _HI_ALL_POWERS:
            power = _HI_ALL_POWERS[w_lower]
            if pending_unit > 0:
                total += pending_unit * power
                pending_unit = 0
            elif total == 0:
                total = power  # bare "sau" = 100
            else:
                total *= power  # stacked powers: sau hazaar = 100 * 1000
        elif w_lower in _HI_NUMS:
            n = _HI_NUMS[w_lower]
            if n > 0:
                pending_unit = n
        elif w in _HI_DEV_NUMS:
            n = _HI_DEV_NUMS[w]
            if n > 0:
                pending_unit = n
        else:
            return None

    if pending_unit > 0:
        if total > 0:
            total += pending_unit
        else:
            total = pending_unit

    return total if total > 0 else None


def _parse_multiword_number(text: str) -> str | None:
    """Try to parse a multi-word Hindi number phrase at the START of text.
    Returns the digit string + length consumed, or None.
    """
    words = text.split()
    best: tuple[int, int] | None = None  # (start, value) at end of words
    for end in range(len(words), 0, -1):
        candidate = words[:end]
        val = _parse_hi_number_phrase(candidate)
        if val is not None:
            best = (end, val)
            break
    if best is None:
        return None
    end_idx, val = best
    consumed = len(" ".join(words[:end_idx]))
    return str(val)


_DEVANAGARI_DIGIT = str.maketrans("०१२३४५६७८९", "0123456789")


def _normalize_dev_digits(text: str) -> str:
    """Convert Devanagari digit characters to ASCII digits: २५ → 25."""
    return text.translate(_DEVANAGARI_DIGIT) if _DEVANAGARI_RE.search(text) else text


def _try_num(word: str) -> str | None:
    # English via word2number
    try:
        return str(w2n.word_to_num(word))
    except Exception:
        pass
    # Romanized Hindi single word
    w_lower = word.lower()
    n = _HI_NUMS.get(w_lower)
    if n is not None:
        return str(n)
    # Devanagari single word
    n = _HI_DEV_NUMS.get(word)
    if n is not None:
        return str(n)
    # Devanagari digits: २५ → 25
    if _DEVANAGARI_RE.search(word):
        return word.translate(_DEVANAGARI_DIGIT)
    return None


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_HINDI_MARKERS = frozenset({
    "hai", "hain", "tha", "thi", "the", "thay",
    "nahi", "nahin", "na",
    "kya", "kyun", "kyon", "kaise", "kaun", "kahan", "kab",
    "matlab", "abhi", "pehle", "baad",
    "lekin", "magar", "aur",
    "karo", "karna", "karenge", "karun", "karein", "karta", "karti", "karte",
    "sikhenge", "sikh", "sikha", "sikhe",
    "dekho", "dekhna", "dekhe", "dekha",
    "chahiye", "chahta", "chahti", "chahte",
    "bhi", "hi",
    "yeh", "woh", "ye", "vo", "voh",
    "iska", "uska", "unki", "unke", "unka",
    "hum", "humko", "humne", "humara", "hamari",
    "aap", "aapko", "aapne", "aapka", "aapki",
    "tum", "tumko", "tumne", "tumhara",
    "main", "mein", "maine", "mujhe",
    "raha", "rahi", "rahe", "rahaa", "rahi",
    "gaya", "gayi", "gaye", "gaya",
    "mat",
    "kyunki", "isliye", "phir", "toh",
    "wala", "wali", "wale",
    "bata", "batao", "bataaye", "bataaiye",
    "sunna", "sunno", "suno", "suniye",
    "kijiye", "kijiyega",
    "ho", "hoga", "hogee", "honge", "hoo",
    "tha", "the", "thi", "thay",
    "aye", "aao", "aate", "aata", "aati",
    "jao", "jata", "jati", "jate", "jaa",
})


def has_hindi_signal(text: str) -> bool:
    """Check if text contains Devanagari script or Hindi lexical markers."""
    if not text:
        return False
    if _DEVANAGARI_RE.search(text):
        return True
    words = set(w.lower() for w in _TOKEN_RE.findall(text))
    return len(words & _HINDI_MARKERS) >= 2


def normalize_numbers(text: str) -> str:
    """Convert number-words to digits. Handles English, Romanized Hindi,
    and Devanagari Hindi number words.

    Single words: thirty→30, pachaas→50, पचास→50
    Multi-word compounds: do sau pachaas → 250, do सौ → 200
    Devanagari digit chars: २५ → 25
    """
    # Pass 1: Devanagari digit chars (independent of word boundaries)
    text = _normalize_dev_digits(text)

    # Pass 2: multi-word Hindi number compounds FIRST, before single words
    # are replaced (so "do sau" is still intact as "do sau", not "2 sau").
    # Match sequences of 2+ known Hindi number words.
    _HI_ALL_KNOWN = sorted(set(_HI_ALL_SINGLE.keys()) | set(_HI_ALL_POWERS.keys()),
                           key=len, reverse=True)
    word_pattern = "|".join(re.escape(w) for w in _HI_ALL_KNOWN)
    # Match 2+ consecutive known words (case-insensitive)
    multi_pat = re.compile(
        r"((?:" + word_pattern + r")(?:\s+(?:" + word_pattern + r"))+)",
        re.IGNORECASE,
    )

    def _repl_multi(m: re.Match) -> str:
        phrase = m.group(0)
        words = phrase.split()
        parsed = _parse_hi_number_phrase(words)
        return str(parsed) if parsed is not None else phrase

    text = multi_pat.sub(_repl_multi, text)

    # Pass 3: single number words (English, Hindi, Devanagari)
    def _repl_single(m: re.Match) -> str:
        n = _try_num(m.group(0))
        return n if n is not None else m.group(0)
    text = _TOKEN_RE.sub(_repl_single, text)

    if _HI_NORM_AVAIL:
        text = hindi_normalize.normalize_devanagari(text)
        text = hindi_normalize.strip_zero_width(text)

    return text


def _words_and_confs(text: str, token_ids: list[int],
                     token_logprobs: list[float],
                     tokenizer) -> tuple[list[str], list[float]]:
    """Convert token-level logprobs to per-word averaged probabilities."""
    words = text.split()
    if not words or not token_ids:
        return words, [1.0] * len(words)

    token_probs = [math.exp(lp) for lp in token_logprobs]

    char_to_tok = {}
    char_idx = 0
    for i, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid])
        for _ in tok_str:
            char_to_tok[char_idx] = i
            char_idx += 1

    word_confs = []
    search_start = 0
    for w in words:
        try:
            word_start = text.index(w, search_start)
        except ValueError:
            word_confs.append(1.0)
            continue
        word_end = word_start + len(w)
        search_start = word_end

        tok_indices = set()
        for c in range(word_start, min(word_end, char_idx)):
            if c in char_to_tok:
                tok_indices.add(char_to_tok[c])

        if tok_indices:
            word_confs.append(
                sum(token_probs[t] for t in tok_indices) / len(tok_indices)
            )
        else:
            word_confs.append(1.0)

    return words, word_confs


def fusion_merge(fast_text: str, specialist_text: str,
                 fast_words: list[str], fast_confs: list[float],
                 spec_words: list[str], spec_confs: list[float]) -> str:
    """Cross-model lattice fusion: pick higher-confidence word per aligned position."""
    if not fast_words or not spec_words:
        return specialist_text or fast_text

    matcher = SequenceMatcher(None, fast_words, spec_words)
    result = []
    for op, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if op == 'equal':
            result.extend(fast_words[a_start:a_end])
        elif op == 'replace':
            paired = min(a_end - a_start, b_end - b_start)
            for k in range(paired):
                if fast_confs[a_start + k] >= spec_confs[b_start + k]:
                    result.append(fast_words[a_start + k])
                else:
                    result.append(spec_words[b_start + k])
            if a_end - a_start > paired:
                for k in range(a_start + paired, a_end):
                    result.append(fast_words[k])
            elif b_end - b_start > paired:
                for k in range(b_start + paired, b_end):
                    result.append(spec_words[k])
        elif op == 'delete':
            result.extend(fast_words[a_start:a_end])
        elif op == 'insert':
            result.extend(spec_words[b_start:b_end])
    return " ".join(result)
