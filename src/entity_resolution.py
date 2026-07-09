import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import spacy

nlp = spacy.load("en_core_web_sm")

def simplify_entity_name(text):
    """
    ตัดคำที่ไม่ใช่คำนาม (เช่น Adjectives, Adverbs) ออกจากชื่อ Entity
    """
    doc = nlp(text)
    # เก็บเฉพาะคำที่เป็นคำนาม (NOUN) หรือคำนามเฉพาะ (PROPN)
    clean_tokens = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    # ถ้าตัดแล้วไม่เหลือคำนามเลย (เช่นกรณีที่เป็นคำวิเศษณ์ล้วนๆ) ให้คืนค่าเดิม
    if not clean_tokens:
        return text
    return " ".join(clean_tokens)

class EntityResolver:
    def __init__(self, threshold=0.85):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        # เปลี่ยนโครงสร้างจาก string -> string เป็น string -> {dict}
        # เก็บทั้งชื่อหลัก (canonical) และประเภท (type)
        self.resolved_entities = {}

    def _types_match(self, existing_type, new_type):
        """ตรรกะการตรวจสอบความเข้ากันได้ของ Type"""
        # ถ้ามีฝั่งไหนเป็น 'Entity' (ไม่ทราบประเภท) ให้ยอมให้รวมได้ (ยืดหยุ่น)
        if existing_type == "Entity" or new_type == "Entity":
            return True
        # ถ้าประเภทต่างกัน (เช่น Person vs Organization) ห้ามรวม!
        return existing_type == new_type

    def resolve(self, entity, entity_type="Entity"):
        # 1. ทำ Simplify ชื่อก่อน (ใช้ฟังก์ชันที่คุณเพิ่งเพิ่มเข้าไป)
        simplified = simplify_entity_name(entity)
        clean_lower = simplified.lower().strip()
        
        # 2. เช็คว่าเคยเจอคีย์นี้ไปแล้วหรือยัง
        if clean_lower in self.resolved_entities:
            existing = self.resolved_entities[clean_lower]
            if self._types_match(existing['type'], entity_type):
                return existing['canonical']
            
        # 3. ตรวจสอบกับโหนดที่มีอยู่เดิม (Substring & Similarity)
        for key, existing in list(self.resolved_entities.items()):
            # ตรวจสอบ Type ก่อนเสมอ ถ้าไม่แมตช์ ไม่ต้องเสียเวลาคำนวณ
            if not self._types_match(existing['type'], entity_type):
                continue
            
            # Substring Matching
            if (len(clean_lower) > 3 and clean_lower in key) or \
               (len(key) > 3 and key in clean_lower):
                
                if abs(len(clean_lower) - len(key)) <= 10:
                    # อัปเดตชื่อหลัก
                    new_canonical = entity.strip() if len(entity) > len(existing['canonical']) else existing['canonical']
                    self.resolved_entities[clean_lower] = {"canonical": new_canonical, "type": entity_type}
                    return new_canonical

        # 4. ใช้ AI ค้นหาความหมาย (Cosine Similarity)
        if self.resolved_entities:
            # ดึงเฉพาะรายการที่ Type ตรงกันมาเทียบ
            candidates = [(k, v) for k, v in self.resolved_entities.items() if self._types_match(v['type'], entity_type)]
            if candidates:
                keys = [c[0] for c in candidates]
                canonicals = [c[1]['canonical'] for c in candidates]
                
                embeddings1 = self.model.encode([clean_lower])
                embeddings2 = self.model.encode(keys)
                
                similarities = cosine_similarity(embeddings1, embeddings2)[0]
                best_idx = np.argmax(similarities)
                
                if similarities[best_idx] >= self.threshold:
                    best_match = canonicals[best_idx]
                    self.resolved_entities[clean_lower] = {"canonical": best_match, "type": entity_type}
                    return best_match
        
        # 5. ถ้าไม่ตรงกับใครเลย บันทึกเป็น Node ใหม่
        self.resolved_entities[clean_lower] = {"canonical": entity.strip(), "type": entity_type}
        return entity.strip()