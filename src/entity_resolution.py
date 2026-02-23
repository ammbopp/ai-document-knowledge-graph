import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EntityResolver:
    def __init__(self, threshold=0.85):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.resolved_entities = {}

    def resolve(self, entity):
        clean_entity = entity.strip()
        clean_lower = clean_entity.lower()
        
        # 1. เช็คว่าเคยเจอคำนี้เป๊ะๆ ไปแล้วหรือยัง
        if clean_lower in self.resolved_entities:
            return self.resolved_entities[clean_lower]
            
        # 2. 🔥 กฎใหม่: Substring Matching (เช็คคำซ้อนทับ)
        for existing_lower, existing_canonical in self.resolved_entities.items():
            # ถ้าคำใหม่ซ่อนอยู่ในคำเก่า หรือ คำเก่าซ่อนอยู่ในคำใหม่ (และต้องยาวกว่า 3 ตัวอักษร ป้องกันการยุบรวมคำมั่วๆ เช่น "the")
            if (len(clean_lower) > 3 and clean_lower in existing_lower) or \
               (len(existing_lower) > 3 and existing_lower in clean_lower):
                
                # ให้ยึดชื่อที่ "ยาวกว่า" (มีรายละเอียดมากกว่า) เป็นชื่อหลักเสมอ
                if len(clean_entity) > len(existing_canonical):
                    # อัปเดตข้อมูลเก่าทั้งหมดให้มาใช้ชื่อใหม่ที่ยาวกว่า
                    for k, v in self.resolved_entities.items():
                        if v == existing_canonical:
                            self.resolved_entities[k] = clean_entity
                    self.resolved_entities[clean_lower] = clean_entity
                    return clean_entity
                else:
                    self.resolved_entities[clean_lower] = existing_canonical
                    return existing_canonical

        # 3. ใช้ AI ค้นหาความหมายที่คล้ายกัน (Cosine Similarity)
        if self.resolved_entities:
            existing_canonical_list = list(set(self.resolved_entities.values()))
            embeddings1 = self.model.encode([clean_lower])
            embeddings2 = self.model.encode(existing_canonical_list)
            
            similarities = cosine_similarity(embeddings1, embeddings2)[0]
            best_idx = np.argmax(similarities)
            
            if similarities[best_idx] >= self.threshold:
                best_match = existing_canonical_list[best_idx]
                self.resolved_entities[clean_lower] = best_match
                return best_match
        
        # 4. ถ้าไม่ตรงกับใครเลย ให้บันทึกเป็น Node ใหม่
        self.resolved_entities[clean_lower] = clean_entity
        return clean_entity