import re
import unicodedata

STOPWORDS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "em", "no", "na", "nos",
    "nas", "um", "uma", "para", "por", "com", "sem", "que", "os", "as", "ao",
    "aos", "se", "sua", "seu", "suas", "seus", "ou", "the", "and", "or", "of",
    "to", "in", "on", "for", "with", "as", "is", "are", "be", "will", "you",
    "your", "we", "our", "at", "an", "this", "that", "vaga", "empresa",
    "trabalho", "experiencia", "experiencias", "conhecimento", "area",
}


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    return strip_accents(text.lower())


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if len(t) > 2 and t not in STOPWORDS]
