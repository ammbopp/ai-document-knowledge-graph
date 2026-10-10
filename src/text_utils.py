from .config import MIN_SENTENCE_LENGTH, WINDOW_OVERLAP, WINDOW_SIZE


def create_sliding_windows(sentences, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
    chunks = []
    step = max(1, window_size - overlap)
    for i in range(0, len(sentences), step):
        chunks.append(" ".join(sentences[i : i + window_size]))
        if i + window_size >= len(sentences):
            break
    return chunks


def split_sentences(nlp, text, min_len=MIN_SENTENCE_LENGTH):
    return [s.text.strip() for s in nlp(text).sents if len(s.text.strip()) >= min_len]


def split_by_chars(text, max_chars):
    """Split text into pieces <= max_chars on paragraph/line boundaries (long lines split on spaces)."""
    pieces, current = [], ""
    for para in text.split("\n"):
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            head, para = para[:cut], para[cut:].lstrip()
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if len(current) + len(para) + 1 > max_chars and current:
            pieces.append(current)
            current = ""
        current = f"{current}\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces
