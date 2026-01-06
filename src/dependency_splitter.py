import spacy

nlp = spacy.load("en_core_web_sm")


def get_subject_chunks(doc):
    """
    ดึง subject ของประโยคในรูป noun chunk
    เช่น 'Company A', 'Company B'
    """
    subjects = []
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in ("nsubj", "nsubjpass"):
            subjects.append(chunk.text)
    return subjects


def split_complex_sentence(sentence: str):
    """
    แยกประโยคซับซ้อนด้วย dependency parsing
    พร้อม subject propagation
    """
    doc = nlp(sentence)

    clauses = []

    # 1️⃣ ดึง subject เป็น noun chunk (ถูกต้องกว่า token)
    subjects = get_subject_chunks(doc)
    subject_text = " and ".join(subjects)

    # 2️⃣ แยก clause ตาม verb
    for token in doc:
        if token.pos_ == "VERB":
            subtree = list(token.subtree)
            clause_text = " ".join([t.text for t in subtree])

            # 3️⃣ ถ้า clause ไม่มี subject → เติม subject หลัก
            if subject_text and not any(
                t.dep_ in ("nsubj", "nsubjpass") for t in subtree
            ):
                clause_text = subject_text + " " + clause_text

            clauses.append(clause_text)

    # ถ้ามี year → สร้าง clause ที่ตัดเวลาออกเพิ่ม
    if any(t.ent_type_ == "DATE" for t in doc):
        no_time = " ".join(
            t.text for t in doc if t.ent_type_ != "DATE"
        )
        clauses.append(no_time)

    # 4️⃣ กันประโยคซ้ำ
    clauses = list(dict.fromkeys(clauses))

    # fallback
    if not clauses:
        clauses = [sentence]

    return clauses
