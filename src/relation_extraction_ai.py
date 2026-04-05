import re
import requests
import json

# =========================
# 1. API Setup (Ollama)
# =========================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

print(f"⏳ Connecting to local Ollama ({MODEL_NAME})...")

# =========================
# 2. Prompt Engineering (Tuned for Llama 3)
# =========================
PROMPT = """You are an expert NLP system. Your ONLY task is to extract relationships as exact triplets.
DO NOT output any conversational text, greetings, or explanations. ONLY output the triplets.

Task: Extract clear and concise relationship triplets from the text.
Format: Entity 1 | Verb Phrase | Entity 2 ###

CRITICAL RULES:
1. Output MUST strictly follow this pattern: Subject | Verb Phrase | Object ###
2. You MUST use EXACTLY TWO pipes (|) per relationship. Never combine the verb and the object.
3. Keep Entities SHORT. Entities MUST be NOUNS (Person, Organization, Concept).
4. Keep [Entity 2] short and precise (1-5 words). Do NOT copy long descriptive phrases. 
5. Extract the CORE action. Ignore speech tags like "said that".
6. SEPARATE TITLES FROM ORGANIZATIONS: Put job titles in the Verb Phrase. Example: BAD = `John | is | CEO of Apple`. GOOD = `John | is CEO of | Apple`.
7. CLEAN NAMES: Remove titles (like CTO, CEO, Dr., Mr.) from person names. Example: Use `Alice Johnson` instead of `CTO Alice Johnson`.
8. Do not use generic placeholder words like "Subject" or "Object" in your output. Use the actual names from the text.
9. Do NOT use brackets, quotes, or any special punctuation around the words.
10. DO NOT extract years, dates, or times (e.g., 2022, 2025, Monday) as entities. Ignore them completely.
11. NEVER use verbs or actions as Entities (e.g., "to launch"). Turn them into relationships instead.
12. AVOID REDUNDANCY: Do NOT output multiple slightly different triplets for the same meaning. Just pick the best one.
13. NO CHAT. NO EXPLANATIONS. Start immediately with the first triplet.

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

Input: Dr. Sarah is the Director of the Medical Institute.
Output: Sarah | is the Director of | Medical Institute ###


Input: {}
Output:
"""

# =========================
# 3. Helper Function (คุยกับ Ollama API)
# =========================
def ask_llm(text):
    input_text = PROMPT.format(text)
    
    payload = {
        "model": MODEL_NAME,
        "prompt": input_text,
        "stream": False,
        "options": {
            "temperature": 0.0,#ไม่เพ้อ
            "num_predict": 150, #ไม่ยาว
            "stop": ["Input:", "\n\n", "Input"]  # หยุดทันที ป้องกันการไหลของข้อความ
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status() # เช็คว่า Request ผ่านหรือไม่
        
        result = response.json().get("response", "")
        
        # ล้างขยะที่ AI อาจจะแอบใส่มา
        result = result.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace("`", "").strip()
        
        # ป้องกันกรณี Llama เผลอใส่คำเกริ่นนำ
        if "Here" in result or "Output" in result:
            lines = result.split('\n')
            # หาบรรทัดที่มีท่อ | (แสดงว่าเป็นข้อมูล Triplet)
            result = " ".join([line for line in lines if "|" in line])
            
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Ollama Connection Error: {e}")
        print("💡 ตรวจสอบให้แน่ใจว่าเปิดโปรแกรม Ollama และรัน 'ollama run llama3' ไว้แล้ว")
        return ""

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
    
    if head_clean not in text_lower and head_clean.split()[0] not in text_lower: return True 
    if tail_clean not in text_lower and tail_clean.split()[0] not in text_lower: return True 
    return False

def parse_triples(text):
    triples = []
    text = text.strip()
    if text.startswith("###"):
        text = text[3:]
        
    facts = [f.strip() for f in text.split("###") if f.strip()]

    for fact in facts:
        parts = [p.strip() for p in fact.split("|") if p.strip()]
        
        if len(parts) == 3:
            triples.append((parts[0], parts[1], parts[2]))
        elif len(parts) >= 5 and len(parts) % 2 == 1:
            for i in range(1, len(parts) - 1, 2):
                triples.append((parts[0], parts[i], parts[i+1]))
        elif len(parts) == 4:
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
    
    if ent_clean.isdigit(): return False
    if len(ent_clean) < 2: return False
        
    forbidden_words = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "today", "yesterday", "tomorrow",
        "he", "she", "it", "they", "this", "that", "these", "those", "we", "i", "you",
        "the", "a", "an", "and", "or", "some", "many", "all"
    }
    
    if ent_clean in forbidden_words:
        return False
    
    if ent_clean.startswith("to "): 
        return False
        
    return True

# =========================
# 5. Main Extraction Function
# =========================
def extract_relations(sentences):
    print("🔥 extract_relations CALLED (Ollama Llama3 - FIXED)")
    relations = []

    for sent in sentences:
        if len(sent) < 10: continue

        try:
            output = ask_llm(sent)
            print(f"DEBUG Output: {output}") # สามารถเอาบรรทัดนี้ออกได้ถ้าไม่อยากให้หน้าจอรันรก

            if not output: # ถ้า API มีปัญหาหรือตอบกลับมาว่างเปล่า ให้ข้ามไป
                continue

            triples = parse_triples(output)

            for head, rel_text, tail in triples:
                # 🔥 ตัดคำนำหน้า (Article) ทิ้งอัตโนมัติ (เช่น "The robotics program" -> "robotics program")
                head = re.sub(r'^(the|a|an)\s+', '', head.strip(), flags=re.IGNORECASE)
                tail = re.sub(r'^(the|a|an)\s+', '', tail.strip(), flags=re.IGNORECASE)

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
                    "method": "Ollama_Llama3"
                })

        except Exception as e:
            print(f"❌ Error processing sentence: {e}")

    final_relations = deduplicate(relations)
    print(f"✅ Extracted {len(final_relations)} valid relations.")
    return final_relations