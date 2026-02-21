import re
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# 1. Model Setup (Direct Loading)
# =========================
print("⏳ Loading Model (Flan-T5-Large)...")
try:
    # โหลดโมเดลตรงๆ เพื่อความเสถียร (เลิกใช้ Pipeline)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    print("💡 Suggestion: Run 'pip install sentencepiece protobuf'")
    raise e

nlp = spacy.load("en_core_web_sm")

# =========================
# 2. Prompt Engineering
# =========================
PROMPT = """
Task: Extract relationship triplets.
Format: Subject | Verb Phrase | Object ###

Rules:
1. Do NOT use nouns alone (e.g., "CEO", "Partner") as relations.
2. Do NOT invent information.
3. If A works with B on C, split it: "A | works with | B" and "A | works on | C".
4. Extract multiple relationships if present.

CRITICAL RULES:
1. Each relationship MUST have EXACTLY two pipes (|). 
2. If there are multiple relationships, separate them completely with " ### ". 
3. DO NOT chain relationships together. Create a new triplet for each fact.
4. The middle part MUST be a verb phrase (e.g., "uses", "developed", "worked with") and MUST NOT be a noun alone (e.g., "CEO", "Partner", "Platform").

Examples:
Input: John Smith is the CEO of Company A.
Output: John Smith | is the CEO of | Company A ###

Input: Company A collaborated with Company B to develop Project X.
Output: Company A | collaborated with | Company B ### Company A | developed | Project X ###

Input: Company B works with Company C on Project Y.
Output: Company B | works with | Company C ### Company B | works on | Project Y ###

Input: Dr. Somchai is the Dean of the Faculty of Engineering.
Output: Dr. Somchai | is the Dean of | Faculty of Engineering ###

Input: The robotics program works with RoboTech Company for internships.
Output: The robotics program | works with | RoboTech Company ###

Input: Funding for the project was provided by the National Innovation Agency.
Output: The project | was funded by | National Innovation Agency ###

Input: John Miller was appointed as the project manager and worked with GreenField University.
Output: John Miller | was appointed as | project manager ### John Miller | worked with | GreenField University ###

Input: {}
Output:
"""

# =========================
# 3. Helper Function (คุยกับ AI)
# =========================
def ask_llm(text):
    """
    ฟังก์ชันส่งข้อความหา AI โดยตรง (แทน extractor)
    """
    input_text = PROMPT.format(text)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate (Deterministic = ไม่สุ่ม)
    outputs = model.generate(
        **inputs, 
        max_length=128, 
        temperature=0.0, 
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# 4. Filters & Logic
# =========================
FORBIDDEN_RELATIONS = {
    "ceo", "founder", "manager", "president", "director", "head", 
    "owner", "partner", "member", "company", "project"
}

def is_valid_relation(relation):
    rel_clean = relation.strip().lower()
    if rel_clean in FORBIDDEN_RELATIONS: return False
    if relation[0].isupper() and " " not in relation: return False # ชื่อเฉพาะ
    if len(rel_clean) < 2: return False
    return True

def is_hallucination(head, tail, original_text):
    text_lower = original_text.lower()
    head_clean = head.lower().strip()
    tail_clean = tail.lower().strip()
    
    # เช็คว่ามีคำนี้ในประโยคจริงไหม
    if head_clean not in text_lower and head_clean.split()[0] not in text_lower: return True 
    if tail_clean not in text_lower and tail_clean.split()[0] not in text_lower: return True 
    return False

def parse_triples(text):
    triples = []
    # ลบ ### ที่อาจจะอยู่หน้าสุดออกก่อน
    text = text.strip()
    if text.startswith("###"):
        text = text[3:]
        
    facts = [f.strip() for f in text.split("###") if f.strip()]

    for fact in facts:
        parts = [p.strip() for p in fact.split("|") if p.strip()]
        
        # 1. สมบูรณ์แบบ (A | rel | B)
        if len(parts) == 3:
            triples.append((parts[0], parts[1], parts[2]))
            
        # 2. แบบที่ LLM ขี้เกียจเขียนประธานซ้ำ (Subj | Rel1 | Obj1 | Rel2 | Obj2)
        elif len(parts) >= 5 and len(parts) % 2 == 1:
            # ยึด parts[0] เป็นประธานเสมอ แล้วขยับไปทีละคู่
            for i in range(1, len(parts) - 1, 2):
                triples.append((parts[0], parts[i], parts[i+1]))
                
        # 3. แบบที่ LLM รวบ Object (Subj | Rel | Obj1 | Obj2)
        elif len(parts) == 4:
            # ยึดประธาน (0) และกริยา (1) แจกจ่ายให้ Object ทั้งสองตัว
            triples.append((parts[0], parts[1], parts[2]))
            triples.append((parts[0], parts[1], parts[3]))

    return triples

def deduplicate(relations):
    seen = set()
    unique = []
    for r in relations:
        key = (r["head"].lower(), r["relation"].lower(), r["tail"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

# =========================
# 5. Main Extraction Function
# =========================
def extract_relations(sentences):
    print("🔥 extract_relations CALLED (Direct Model - FIXED)")
    relations = []

    for sent in sentences:
        if len(sent) < 10: continue

        try:
            # 🔥 แก้ไขตรงนี้: เรียก ask_llm แทน extractor
            output = ask_llm(sent)
            
            # (Optional) Debug ดูผลลัพธ์ดิบ
            print(f"DEBUG: {output}")

            triples = parse_triples(output)

            for head, rel_text, tail in triples:
                if not is_valid_relation(rel_text): continue
                if is_hallucination(head, tail, sent): continue
                if head.lower() == tail.lower(): continue

                relations.append({
                    "head": head,
                    "relation": rel_text,
                    "tail": tail,
                    "source_sentence": sent,
                    "confidence": 0.9,
                    "method": "LLM+Direct"
                })

        except Exception as e:
            print(f"❌ Error processing sentence: {e}")

    final_relations = deduplicate(relations)
    print(f"✅ Extracted {len(final_relations)} valid relations.")
    return final_relations