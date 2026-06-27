"""Post-decode text fixes: normalize number words to digits."""
import re
from word2number import w2n

_TOKEN_RE = re.compile(r"[\w'-]+")


def _try_num(word: str) -> str | None:
    try:
        return str(w2n.word_to_num(word))
    except Exception:
        return None


def normalize_numbers(text: str) -> str:
    """Convert English number-words to digits.

    Handles single words (thirty→30) and hyphenated (twenty-five→25).
    Skips multi-word number phrases to avoid year ambiguity (nineteen eighty → not touched).
    """
    def _repl(m: re.Match) -> str:
        n = _try_num(m.group(0))
        return n if n is not None else m.group(0)
    return _TOKEN_RE.sub(_repl, text)
