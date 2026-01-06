from transformers import pipeline

extractor = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

PROMPT = """
You are an information extraction system.

Task:
From the given sentence, identify ALL possible relationships.
If a sentence contains multiple actions or participants,
extract multiple relationship triples.

Rules:
- Do NOT summarize.
- Do NOT select only the main relation.
- Extract every possible pair of entities that are related.
- Relations do NOT need to be normalized.
- Use the exact wording that best describes the relationship.

Output:
Return the relationships as raw relation triples.

Sentence:
{}
"""

INVALID_RELATIONS = {
    "date_of_death",
    "place_of_death",
    "date_of_birth"
}

def is_valid_relation(rel):
    return rel not in INVALID_RELATIONS

def estimate_quality(head, relation_text, tail):
    # tail ยาวเกิน = น่าจะ hallucination
    if len(tail.split()) > 5:
        return "low"

    # relation เชิงเวลา / ความรู้โลก (มัก noise)
    noisy_keywords = ["DATE", "COUNTRY", "TIME", "YEAR"]
    if any(k in relation_text.upper() for k in noisy_keywords):
        return "low"

    return "medium"

def split_multi_triples(tokens):
    triples = []
    i = 0

    while i < len(tokens):
        # หา relation (มักเป็นตัวพิมพ์ใหญ่)
        if tokens[i].isupper():
            relation = tokens[i]

            # head = คำก่อนหน้า relation
            head_tokens = tokens[:i]
            head = " ".join(head_tokens).strip()

            # tail = คำหลัง relation
            tail_tokens = []
            j = i + 1
            while j < len(tokens):
                # ❗ ถ้าเจอ token ที่ซ้ำกับ head → หยุด
                if tokens[j] in head_tokens:
                    break
                # ❗ ถ้าเจอ relation ใหม่ → หยุด
                if tokens[j].isupper():
                    break

                tail_tokens.append(tokens[j])
                j += 1

            tail = " ".join(tail_tokens).strip()

            if head and tail:
                triples.append((head, relation, tail))

            # ตัด tokens ที่ใช้แล้วทิ้ง
            tokens = tokens[j:]
            i = 0
        else:
            i += 1

    return triples

def deduplicate_relations(relations):
    seen = set()
    unique = []

    for r in relations:
        key = (
            r["head"].lower(),
            r["relation_text"].lower(),
            r["tail"].lower()
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique

def extract_relations(sentences):
    print("🔥 extract_relations CALLED")
    relations = []

    for sent in sentences:
        result = extractor(PROMPT.format(sent))
        output = result[0]["generated_text"]

        print("DEBUG OUTPUT:", output)

        # 🔹 CASE: output เป็น string triple
        if isinstance(output, str):
            tokens = output.strip().split()

            # ใช้ตัวที่คุณเขียนไว้แล้ว
            triples = split_multi_triples(tokens)

            for head, relation, tail in triples:

                # กรอง junk ที่มาจาก table / context
                if any(x in head for x in ["DATE", "TABLE", "CONTEXT"]):
                    continue

                if not is_valid_relation(relation.lower()):
                    continue

                quality = estimate_quality(head, relation, tail)

                relations.append({
                    "head": head,
                    "relation_text": relation,
                    "tail": tail,
                    "source_sentence": sent,
                    "quality": quality
                })
    relations = deduplicate_relations(relations)
    return relations
