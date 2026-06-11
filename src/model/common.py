import re
from rapidfuzz import fuzz


PROMPT_INSTRUCTIONS = "You are an assistant specialized in extracting mechanistic toxicology events (MIE, KE, AO) from scientific text.\n\nGiven the following text from a toxicology article, extract all MIE, KE and AO events with the associated chemical and a concise description.\n\nReturn one event per line in the exact format:\n\"chemical\",\"event_type\",\"description\"\n\nIf the text does not contain any MIE, KE or AO events, return an empty output.\n\nText:\n"

# Utils
def csv_quote(value: str) -> str:
    """
    Wrap a string in double quotes and escape internal quotes for CSV-like output.
    """
    if value is None:
        value = ""
    value = str(value)
    value = value.replace('"', '""')
    return f'"{value}"'

def normalize_whitespace(text: str) -> str:
    """
    Basic whitespace normalization to avoid duplicates.
    """
    return " ".join((text or "").split())

def norm(s: str) -> str:
    """
    Whitespace normalization + lowercase for simple case-insensitive matching.
    """
    return " ".join(s.lower().split())

def contains_normalized_substring(haystack: str, needle: str) -> bool:
    """
    Simple exact substring match ignoring case and extra whitespace. Useful for chemical variants and short descriptions.
    """
    if not needle:
        return False
    return norm(needle) in norm(haystack)

def contains_wordbound(haystack: str, needle: str) -> bool:
    """
    Word-boundary-ish match to avoid substring disasters.
    Works for multi-word needles too.
    """
    if not haystack or not needle:
        return False

    hs = haystack.lower()
    tokens = needle.strip().lower().split()
    if not tokens:
        return False

    # \w is unicode-aware in Python; use lookarounds to enforce boundaries
    pat = r"(?<![\w])" + r"\s+".join(re.escape(t) for t in tokens) + r"(?![\w])"
    return re.search(pat, hs, flags=re.UNICODE) is not None

def fuzzy_score(text: str, pattern: str) -> float:
    """
    Performs a fuzzy similarity check between text and pattern, returning a score in 0..1.
    """
    if not text or not pattern:
        return 0.0

    t = norm(text)
    p = norm(pattern)

    # partial_ratio is good for "pattern contained-ish in long text"
    pr = fuzz.partial_ratio(t, p) / 100.0
    ts = fuzz.token_set_ratio(t, p) / 100.0
    # blend: token_set helps paraphrase-y overlaps, partial helps near-substring
    return max(pr * 0.7 + ts * 0.3, ts * 0.6 + pr * 0.4)

def compute_score(text: str, event: dict) -> float:
    """
    Returns a score in 0..100 based on description matching only.
    Perfect (100) only if long or short description exact match.
    Chemical presence is handled separately as a boolean flag.

    desc_l is treated as a verbatim quote from the paper, so fuzz.partial_ratio
    (best-window substring alignment) is used directly for it rather than
    fuzzy_score (which blends token_set_ratio and would fire on shared domain
    vocabulary across many unrelated chunks).
    desc_s is a human-written label/paraphrase, so fuzzy_score (blend) is kept.
    """
    desc_s = event.get("event_description_short", "")
    desc_l = event.get("event_description_long", "")

    # Clean truncation artifacts from the long description
    desc_l = re.sub(r"\s*\.\.\.\s*", " ", desc_l)
    desc_l = desc_l.replace("…", " ")
    desc_l = re.sub(r"\s{2,}", " ", desc_l)

    # Perfect condition: long or short exact match
    if contains_normalized_substring(text, desc_s) or contains_normalized_substring(text, desc_l):
        return 100.0

    t = norm(text)
    p_s = norm(desc_s)
    p_l = norm(desc_l)

    # Short description: blended fuzzy (partial_ratio + token_set_ratio)
    pr_s = fuzz.partial_ratio(t, p_s) / 100.0
    ts_s = fuzz.token_set_ratio(t, p_s) / 100.0
    fs = max(pr_s * 0.7 + ts_s * 0.3, ts_s * 0.6 + pr_s * 0.4)

    # Long description: pure partial_ratio — verbatim quote should align as a
    # window inside the chunk text; token_set_ratio would produce false positives
    # from shared domain terms (firing rate, network burst, etc.)
    fl = fuzz.partial_ratio(t, p_l) / 100.0 if desc_l else fs

    score = 10.0
    score += 20.0 * fs
    score += 60.0 * fl

    return min(99.0, score)
