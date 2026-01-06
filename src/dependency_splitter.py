import spacy

nlp = spacy.load("en_core_web_sm")

def split_complex_sentence(sentence: str):
    doc = nlp(sentence)
    
    # ถ้าประโยคสั้นอยู่แล้ว ไม่ต้องทำอะไร
    if len(doc) < 15:
        return [sentence]

    sub_sentences = []
    
    # แยกด้วย ; หรือ , ที่ตามด้วยคำเชื่อม (and, but) แบบง่ายๆ
    current_chunk = []
    
    for token in doc:
        current_chunk.append(token.text)
        
        # เงื่อนไขการตัดประโยค: เจอ ; หรือ .
        if token.text in [";", "."]:
            text = " ".join(current_chunk).strip()
            if len(text) > 5: # ป้องกันประโยคสั้นเกิน
                sub_sentences.append(text)
            current_chunk = []
            
        # เงื่อนไขพิเศษ: เจอ , และคำต่อไปเป็น cc (and, but, or)
        elif token.text == "," and (token.i + 1 < len(doc) and doc[token.i+1].pos_ == "CC"):
            text = " ".join(current_chunk[:-1]).strip() # ตัด , ออก
            if len(text) > 10: # ต้องยาวพอสมควรถึงจะตัด
                sub_sentences.append(text)
            current_chunk = []

    # เก็บตกส่วนที่เหลือ
    if current_chunk:
        text = " ".join(current_chunk).strip()
        if len(text) > 5:
            sub_sentences.append(text)

    # Fallback: ถ้าตัดไม่ได้เลย ให้ส่งคืนประโยคเดิม
    if not sub_sentences:
        return [sentence]

    return sub_sentences