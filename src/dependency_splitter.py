import spacy

nlp = spacy.load("en_core_web_sm")

def split_complex_sentence(sentence: str):
    doc = nlp(sentence)

    # ถ้าสั้นมาก ไม่ต้อง split
    if len(doc) < 12:
        return [sentence.strip()]

    clauses = []
    current_tokens = []
    has_verb = False

    def flush():
        nonlocal current_tokens, has_verb
        if current_tokens and has_verb:
            text = " ".join(current_tokens).strip()
            if len(text) > 10:
                clauses.append(text)
        current_tokens = []
        has_verb = False

    for token in doc:
        current_tokens.append(token.text)

        if token.pos_ == "VERB":
            has_verb = True

        # === จุด split เชิง syntax ===
        # 1. เจอ coordinating conjunction (and, but, or)
        if token.dep_ == "cc":
            flush()
            continue

        # 2. เจอ adverbial clause / relative clause
        if token.dep_ in {"advcl", "relcl"}:
            flush()
            continue

        # 3. punctuation ที่เป็น boundary จริง
        if token.text in {";", "."}:
            flush()
            continue

    # เก็บส่วนสุดท้าย
    flush()

    # Fallback: ถ้าแตกเละ ให้คืนประโยคเดิม
    if not clauses:
        return [sentence.strip()]

    return clauses
