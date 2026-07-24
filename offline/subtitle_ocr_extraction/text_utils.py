import re
import unicodedata

from rapidfuzz import fuzz


VI_DIACRITIC_CHARS = (
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴĐ"
)

CJK_HANGUL_RE = re.compile(
    r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF"
    r"\u3400-\u4DBF\u4E00-\u9FFF\uf900-\ufaff"
    r"\u3040-\u30FF]"
)
LATIN_VI_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]")
ALLOWED_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9\s\.,!?;:'\"()\[\]\-_/&%$@#+=*…]")


def normalize_text_for_match(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\sÀ-ỹĐđ]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_text(text: str) -> str:
    return normalize_text_for_match(text)


def remove_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def similarity(a: str, b: str) -> float:
    a_norm = normalize_text_for_match(a)
    b_norm = normalize_text_for_match(b)
    strict = fuzz.ratio(a_norm, b_norm) / 100.0
    folded = fuzz.ratio(remove_diacritics(a_norm), remove_diacritics(b_norm)) / 100.0
    return max(strict, folded)


def text_quality_score(text: str) -> float:
    raw = text.strip()
    norm = normalize_text_for_match(raw)
    tokens = norm.split()
    diacritic_count = sum(1 for ch in raw if ch in VI_DIACRITIC_CHARS)
    weird_count = len(re.findall(r"[^\w\sÀ-ỹĐđ.,!?;:'\"()\-]", raw, flags=re.UNICODE))
    long_token_penalty = sum(max(0, len(tok) - 8) for tok in tokens)

    score = 0.0
    score += diacritic_count * 1.5
    score += len(tokens) * 2.0
    score += len(norm) * 0.05
    score -= weird_count * 2.0
    score -= long_token_penalty * 0.5
    return score


def choose_better_text(current_text: str, candidate_text: str) -> str:
    current_score = text_quality_score(current_text)
    candidate_score = text_quality_score(candidate_text)
    if candidate_score > current_score:
        return candidate_text
    if candidate_score < current_score:
        return current_text
    if len(candidate_text) > len(current_text):
        return candidate_text
    return current_text


def frame_to_time(frame: int, fps: float) -> str:
    t = frame / fps
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def is_vi_en_candidate(text: str, score: float, min_score: float = 0.85, min_allowed_ratio: float = 0.7) -> bool:
    text = text.strip()
    if not text:
        return False
    if score < min_score:
        return False
    if CJK_HANGUL_RE.search(text):
        return False
    if not LATIN_VI_RE.search(text):
        return False
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False
    allowed = sum(1 for ch in chars if ALLOWED_RE.fullmatch(ch))
    return (allowed / len(chars)) >= min_allowed_ratio
