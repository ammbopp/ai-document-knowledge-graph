import re
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# 1. Model Setup (Direct Loading)
# =========================
print("⏳ Loading Model (Flan-T5-XL)...")
try:
    model_name = "google/flan-t5-xl"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    print("💡 Suggestion: Run 'pip install sentencepiece protobuf accelerate'")
    raise e

nlp = spacy.load("en_core_web_sm")

# =========================
# 2. Prompt Engineering
# =========================

PROMPT = """
Task: Extract clear and concise relationship triplets from the text.
Format: Entity 1 | Verb Phrase | Entity 2 ###

CRITICAL RULES:
1. Output MUST strictly follow this pattern: Subject | Verb Phrase | Object ###
2. You MUST use EXACTLY TWO pipes (|) per relationship. Never combine the verb and the object.
3. Keep [Entity 2] short and precise (1-5 words). Do NOT copy long descriptive phrases.
4. Extract the CORE action. Ignore speech tags like "said that".
5. Do not use generic placeholder words like "Subject" or "Object" in your output. Use the actual names from the text.
6. Do NOT use brackets, quotes, or any special punctuation around the words.
7. DO NOT extract years, dates, or times (e.g., 2022, 2025, Monday) as entities. Ignore them completely.

Examples:
Input: John Miller was appointed as the project manager and worked with GreenField University.
Output: John Miller | was appointed as | project manager ### John Miller | worked with | GreenField University ###

Input: TechCorp partnered with InnovateX to build an AI platform in 2023.
Output: TechCorp | partnered with | InnovateX ### TechCorp | built | AI platform ###

Input: US President Donald Trump said on Friday that he will impose global tariffs of 15%.
Output: Donald Trump | will impose | global tariffs ### 

Input: That law allows these new tariffs to stay in place before the administration must seek congressional approval.
Output: law | allows | new tariffs ### administration | must seek | congressional approval ###

Input: Major studios like Disney quickly accused ByteDance of copyright infringement.
Output: Disney | accused | ByteDance ### Disney | accused of | copyright infringement ###

Input: Seedance 2.0 can generate cinema-quality video, complete with sound effects and dialogue.
Output: Seedance 2.0 | can generate | cinema-quality video ### Seedance 2.0 | complete with | sound effects ###

Input: In 2022, the Facility of Engineering collaborated with the Facility of Science.
Output: Facility of Engineering | collaborated with | Facility of Science ###

Input: Dr. Somchai is the Dean of the Faculty of Engineering.
Output: Dr. Somchai | is the Dean of | Faculty of Engineering ###

Input: {}
Output:
"""
# =========================
# 3. Helper Function (คุยกับ AI)
# =========================
def ask_llm(text):
    input_text = PROMPT.format(text)
    inputs = tokenizer(input_text,max_length=1024, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            num_beams=1, # สำคัญ: ห้ามใช้ beam search กับ XL บนเครื่องส่วนตัว เพราะจะช้ามาก
            do_sample=False
        )
    
    # เคลียร์แรมหลังใช้งานเสร็จ (ป้องกันอาการค้าง)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # ล้างขยะที่ AI อาจจะแอบใส่มา
    result = result.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return result

# =========================
# 4. Filters & Logic
# =========================
FORBIDDEN_RELATIONS = {
    "ceo", "founder", "manager", "president", "director", "head", 
    "owner", "partner", "member", "company", "project",
    "said", "says", "announced", "stated", "told", "added", "reported", "mentioned"
}

def is_valid_relation(relation):
    rel_clean = relation.strip().lower()
    if any(word in rel_clean.split() for word in FORBIDDEN_RELATIONS): 
        return False
        
    if relation[0].isupper() and " " not in relation: return False
    if len(rel_clean) < 2: return False
    if len(rel_clean.split()) > 4: return False
        
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
    
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
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

def is_valid_entity(entity):
    ent_clean = entity.strip().lower()
    
    # ห้ามเป็นตัวเลขล้วน
    if ent_clean.isdigit(): return False
    # ห้ามสั้นเกิน 1 ตัวอักษร
    if len(ent_clean) < 2: return False
        
    # 🔥 แบนวัน เวลา และสรรพนาม (Stop words / Pronouns)
    forbidden_words = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "today", "yesterday", "tomorrow",
        "he", "she", "it", "they", "this", "that", "these", "those", "we", "i", "you"
    }
    
    # ถ้า entity ตรงกับคำต้องห้ามเป๊ะๆ ให้เตะทิ้ง
    if ent_clean in forbidden_words:
        return False
        
    return True

def is_valid_relation(relation):
    rel_clean = relation.strip().lower()
    if rel_clean in FORBIDDEN_RELATIONS: return False
    if relation[0].isupper() and " " not in relation: return False
    if len(rel_clean) < 2: return False
    
    # 🔥 เพิ่มกฎใหม่: Relation ต้องเป็นกริยาสั้นๆ ห้ามยาวเกิน 4 คำ
    if len(rel_clean.split()) > 4: 
        return False
        
    return True

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
                if not is_valid_entity(head) or not is_valid_entity(tail): continue
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