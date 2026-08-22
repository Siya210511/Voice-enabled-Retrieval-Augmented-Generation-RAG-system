"""
Minimal stopword lists for English and (transliterated + Devanagari) Hindi.
Used only to strip noise words before computing groundedness/topic overlap --
NOT a linguistic resource, just enough to stop "the"/"is"/"and" from
dominating overlap scores the way they did in the basic groundedness check.
"""

ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "of", "at", "by", "for", "with", "about",
    "to", "from", "in", "on", "into", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "whom", "as", "does", "do", "did", "has", "have",
    "had", "having",
}

HINDI_STOPWORDS = {
    "है", "हैं", "था", "थी", "थे", "के", "का", "की", "को", "में", "से",
    "पर", "एक", "यह", "वह", "और", "भी", "ही", "तो", "जो", "कि", "इस",
    "उस", "हो", "गया", "गयी", "गई",
}

ALL_STOPWORDS = ENGLISH_STOPWORDS | HINDI_STOPWORDS


def content_words(text: str) -> set[str]:
    """Lowercased word tokens with stopwords removed. Works for mixed
    English/Hindi text since we just tokenize on word boundaries."""
    import re
    words = re.findall(r"\w+", text.lower())
    return {w for w in words if w not in ALL_STOPWORDS and len(w) > 1}
