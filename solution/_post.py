"""Post-decode text fixes: normalize number words to digits + cross-model fusion."""
import math
import re
from difflib import SequenceMatcher

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
