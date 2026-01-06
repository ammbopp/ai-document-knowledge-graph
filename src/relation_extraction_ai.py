import re
from transformers import pipeline

# โหลด Model
extractor = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=256
)

PROMPT = """
Task: Extract structured relationships.
Format: Subject | Verb | Object ###
Rules:
1. END every triplet with " ### ".
2. The middle part MUST be a Verb (e.g., works with, is).
3. If a sentence has multiple parts, split into multiple triplets.
4. Do NOT use nouns like "platform", "project" as relations.

Examples:
Input: Company A collaborated with Company B to develop Project X.
Output: 
Company A | collaborated with | Company B ###
Company A | developed | Project X ###

Input: John Smith is the CEO of Company A.
Output: 
John Smith | is the CEO of | Company A ###

Input: Company B works with Company C on Project Y.
Output: 
Company B | works with | Company C ###
Company B | works on | Project Y ###

Input: {}
Output:
"""

INVALID_RELATIONS = {
    "date_of_death", "place_of_death", "date_of_birth", "context", "table"
}

KNOWN_VERBS = [
    " partnered with ", " collaborates with ", " works with ", " works at ",
    " is the ", " is a ", " is ", " joined ", " founded ", " owns ", 
    " located in ", " based in ", " developed "
]

def is_valid_relation(rel):
    return rel.lower() not in INVALID_RELATIONS

def estimate_quality(head, relation_text, tail):
    bad_keywords = ["platform", "program", "infrastructure", "website", "app", "project", "system", "technology"]
    if any(k in relation_text.lower() for k in bad_keywords): return "low"
    if len(relation_text.split()) > 6: return "low"
    if "Faculty" in relation_text or "University" in relation_text: return "low"
    if len(tail.split()) > 15: return "low" 
    if len(head) < 2 or len(tail) < 2: return "low"
    return "medium"

def clean_text(text):
    """ล้างขยะและตัวคั่น"""
    return text.replace("###", "").strip()

def parse_triples(text):
    """
    🔥 2. Logic ใหม่: ตัดด้วย ### ก่อน แล้วค่อยแกะด้วย |
    รองรับทั้งแบบ 3 ช่อง (ปกติ) และ 4 ช่อง (ของแถม)
    """
    triples = []
    
    # 1. แยกแต่ละ Relation ออกจากกันด้วย ###
    segments = text.split("###")
    
    for segment in segments:
        segment = clean_text(segment)
        if not segment: continue

        # แยกด้วย |
        parts = [p.strip() for p in segment.split('|')]
        
        # ✅ Case ปกติ: A | Rel | B
        if len(parts) == 3:
            triples.append((parts[0], parts[1], parts[2]))
            
        # ✅ Case ของแถม: A | Rel | B | C (AI เผลอใส่ Context หรือ Object ที่ 2 มา)
        # ตัวอย่าง: Company B | works | Company C | Project Y
        elif len(parts) >= 4:
            # Relation 1: A -> B
            triples.append((parts[0], parts[1], parts[2]))
            # Relation 2: A -> C (สร้างเพิ่มให้อัตโนมัติ โดยใช้ Rel ตัวเดิม)
            if parts[3]: 
                # ถ้า Object ตัวที่ 2 ดูเหมือนไม่ใช่ปี (ไม่ใช่ตัวเลขล้วน)
                if not parts[3].isdigit():
                    triples.append((parts[0], parts[1], parts[3]))
        
        # ✅ Case ขาด: A | B (เติม is)
        elif len(parts) == 2:
             triples.append((parts[0], "is", parts[1]))

        # ⚠️ Fallback: ถ้าไม่มี | เลย ให้ลองหา Verb
        elif len(parts) == 1 and '|' not in segment:
            for verb in KNOWN_VERBS:
                if verb in segment:
                    p = segment.split(verb, 1)
                    if len(p) == 2:
                        triples.append((p[0].strip(), verb.strip(), p[1].strip()))
                        break
    
    return triples

def deduplicate_relations(relations):
    seen = set()
    unique = []
    for r in relations:
        key = (r["head"].lower(), r["relation_text"].lower(), r["tail"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

def extract_relations(sentences):
    print("🔥 extract_relations CALLED (V9 - Separator Logic)")
    relations = []

    for sent in sentences:
        if len(sent) < 10: continue

        try:
            result = extractor(PROMPT.format(sent))
            output = result[0]["generated_text"]

            print(f"Input: {sent[:40]}...")
            print(f"AI Output: {output}")

            triples = parse_triples(output)

            for head, relation, tail in triples:
                if not is_valid_relation(relation): continue
                quality = estimate_quality(head, relation, tail)
                if quality == "low": 
                    print(f"⚠️ Skipped low quality: {head} --[{relation}]--> {tail}")
                    continue

                relations.append({
                    "head": head,
                    "relation_text": relation,
                    "tail": tail,
                    "source_sentence": sent,
                    "quality": quality
                })

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    relations = deduplicate_relations(relations)
    print(f"✅ Extracted {len(relations)} relations.")
    return relations