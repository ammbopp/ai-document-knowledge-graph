# relation_extraction_ai.py
import re
import requests
import spacy

nlp = spacy.load("en_core_web_sm")


# ==========================================
# 1. Prompt Engineering
# ==========================================
PROMPT = """You are an expert NLP relation extraction system. Your task is to extract highly accurate relationship triplets (Subject | Verb Phrase | Object) from the text.

CRITICAL RULES:
1. Output MUST strictly follow this pattern: Subject | Verb Phrase | Object ###
2. Separate triplets using '###'. You can put multiple triplets on new lines.
3. Keep Entities SHORT and concrete (1-4 words).
4. DIRECTIONAL ACCURACY: Pay close attention to who performs the action and who receives it. Do not invert subjects and objects.
5. CONTENT OVER SPEECH WRAPPERS: Never extract the act of speaking or reporting itself (e.g., "said", "told", "testified", "claimed") as a relation. Extract the core factual claim inside what was said.

=== EXAMPLES OF STRUCTURAL EXTRACTION ===
Input: Alice told investigators last week that Bob secretly stole the company assets from the vault.
Output: Bob | stole | company assets ### company assets | stored in | vault ###

Input: Several employees filed a lawsuit against the firm, claiming they were mistreated by the supervisor.
Output: employees | filed lawsuit against | firm ### supervisor | mistreated | employees ###
=== END OF EXAMPLES ===

STRICT INSTRUCTION: Extract triplets ONLY from the input text provided below. Do not copy the examples.

Input: {}
Output:"""

# ==========================================
# 2. Helper Functions (นำ \n\n ออกจาก stop tokens และกรองขยะบรรทัดต่อบรรทัด)
# ==========================================
def ask_llm(text):
    input_text = PROMPT.format(text)
    payload = {
        "model": "llama3",
        "prompt": input_text,
        "stream": False,
        "options": {
            "temperature": 0.0, 
            "num_predict": 300,
            # 🔥 ถอด "\n\n" ออก เพื่อให้โมเดลสามารถเคาะบรรทัดเปล่าก่อนส่งคำตอบได้โดยไม่โดนสั่งตัดจบ
            "stop": ["Input:", "=== END"] 
        }
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        
        # ล้างขยะสัญลักษณ์ทั่วไป
        result = result.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace("`", "").strip()
        
        # 🛡️ Dynamic Post-Processing: ถ้ามีบรรทัดบทสนทนาหลุดมา ให้กรองเก็บเฉพาะบรรทัดที่มีสัญลักษณ์ท่อ '|' เท่านั้น
        lines = result.split('\n')
        clean_lines = [line.strip() for line in lines if "|" in line]
        
        return " ".join(clean_lines)
    except Exception as e:
        print(f"\n❌ Ollama Error: {e}")
        return ""

# ==========================================
# 3. Dynamic Structural Filters (ตรวจด้วยหลักไวยากรณ์)
# ==========================================
def is_valid_entity(entity):
    ent_clean = entity.strip()
    if not ent_clean or ent_clean.isdigit() or len(ent_clean) < 2: 
        return False
        
    if ent_clean.lower() in {"he", "she", "it", "they", "we", "i", "you", "this", "that"}: 
        return False
        
    doc = nlp(ent_clean)
    
    # กฎข้อที่ 1: ต้องมีคำนาม หรือคำสรรพนามประกอบอยู่จริง ไม่ปล่อยให้เป็นคำเชื่อมลอยๆ
    if not any(token.pos_ in ["NOUN", "PROPN", "PRON"] for token in doc):
        return False
        
    # กฎข้อที่ 2: ห้ามมีกริยาแท้/กริยาช่วยผสมอยู่ในตัวโหนด (สกัดพวก Clause ย่อยออก)
    if any(token.pos_ in ["VERB", "AUX"] for token in doc):
        return False

    # กฎข้อที่ 3: คำสุดท้ายของโหนด ห้ามลงท้ายด้วยคำบุพบทหรือคำเชื่อม (แก้ปัญหาโหนดค้างติ่ง)
    if doc[-1].pos_ in ["ADP", "CCONJ", "SCONJ", "PART"]:
        return False
        
    return True

def is_valid_relation(relation):
    rel_clean = relation.strip().lower()
    if len(rel_clean) < 2 or len(rel_clean.split()) > 5: 
        return False
    return True

def is_hallucination(head, tail, original_text):
    text_lower = original_text.lower()
    head_clean = head.lower().strip()
    tail_clean = tail.lower().strip()
    if head_clean.split()[0] not in text_lower or tail_clean.split()[0] not in text_lower: 
        return True 
    return False

def parse_triples(text):
    triples = []
    facts = [f.strip() for f in text.split("###") if f.strip()]
    for fact in facts:
        parts = [p.strip() for p in fact.split("|") if p.strip()]
        if len(parts) == 3:
            triples.append((parts[0], parts[1], parts[2]))
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

# ==========================================
# 4. Directional & Substance Gatekeeper (ด่านตรวจความถูกต้องเชิงตรรกะ)
# ==========================================
def evaluate_triplet_cloze(head, relation, tail, source_sentence, threshold=0.6):
    """
    ด่านตรวจความถูกต้องเชิงความหมายและทิศทางประธาน-กรรม (Directional & Structural Gatekeeper)
    """
    prompt = f"""You are a strict validation logic gate for Knowledge Graphs. Verify the extracted triplet against the text.

Text: "{source_sentence}"
Triplet: [{head}] ---> ({relation}) ---> [{tail}]

CRITICAL VALIDATION RULES:
1. DIRECTIONAL CHECK: Does [{head}] actually initiate or perform the action ({relation}) upon [{tail}] in the text? If the text states or implies the reverse (i.e., [{tail}] did it to [{head}]), it is an INVERSION and completely INVALID.
2. SUBSTANCE CHECK: Is the relation a real factual event, or is it just a speech tag like "told", "said", "testified"? Speech tag triplets are strictly INVALID.

Reply with ONLY 'VALID' if it passes both rules perfectly.
Reply with ONLY 'INVALID' if there is an inversion, speech noise, or factual error.
Answer:"""
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        result = response.json().get("response", "").strip().upper()
        
        if "VALID" in result and "INVALID" not in result:
            return True, 1.0
        else:
            print(f"🚫 [Structural Filtered: REJECTED] {head} -> {relation} -> {tail}")
            return False, 0.0
            
    except Exception as e:
        print(f"⚠️ Evaluator Error: {e}")
        return True, 0.5
        
# ==========================================
# 5. Main Extraction Function
# ==========================================
def extract_relations(sentences):
    print("🔥 extract_relations CALLED (Ollama Llama3 - FIXED)")
    relations = []

    for sent in sentences:
        if len(sent) < 10: continue

        try:
            output = ask_llm(sent)
            if output:
                print("DEBUG Output:")
                # ใช้เครื่องหมาย ### ในการหั่นข้อความออกมาพิมพ์ทีละบรรทัด
                for fact in output.split("###"):
                    if fact.strip():
                        print(f"  {fact.strip()}")

            if not output: 
                continue

            triples = parse_triples(output)

            for head, rel_text, tail in triples:
                head = re.sub(r'^(the|a|an)\s+', '', head.strip(), flags=re.IGNORECASE)
                tail = re.sub(r'^(the|a|an)\s+', '', tail.strip(), flags=re.IGNORECASE)

                if not is_valid_entity(head) or not is_valid_entity(tail): continue
                if not is_valid_relation(rel_text): continue
                if is_hallucination(head, tail, sent): continue
                if head.lower() == tail.lower(): continue

                is_valid, confidence_score = evaluate_triplet_cloze(head, rel_text, tail, sent)
                if not is_valid:
                    continue

                relations.append({
                    "head": head,
                    "relation": rel_text,
                    "tail": tail,
                    "source_sentence": sent,
                    "confidence": confidence_score,
                    "method": "Ollama_Llama3_Factual_Verifier"
                })

        except Exception as e:
            print(f"❌ Error processing sentence: {e}")

    final_relations = deduplicate(relations)
    print(f"✅ Extracted {len(final_relations)} valid relations.")
    return final_relations