# evaluate_filters_accuracy.py
import os
import re
import requests
import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

# นำเข้าฟังก์ชันจากไฟล์ระบบของคุณ
from src.relation_extraction_ai import ask_llm, parse_triples, is_valid_entity, is_valid_relation, is_hallucination, evaluate_triplet_cloze

# โหลดโมเดลสำหรับหั่นประโยค
nlp = spacy.load("en_core_web_sm")
OLLAMA_URL = "http://localhost:11434/api/generate"

# ==========================================
# 🤖 LLM-as-a-Judge: ฟังก์ชันกรรมการตรวจข้อเท็จจริง
# ==========================================
def judge_triplet_accuracy(head, relation, tail, source_sentence):
    """
    ใช้ LLM ทำหน้าที่เป็นกรรมการตรวจว่า Triplet นี้ ถูกต้องตามหลักความจริงในประโยคหรือไม่
    เพื่อนำมาคำนวณหาค่า Precision และ Hallucination Rate ที่แท้จริง
    """
    prompt = f"""You are an objective QA inspector for Knowledge Graphs. Your job is to verify if the extracted triplet is 100% accurate based on the text.

Source Text: "{source_sentence}"
Extracted Triplet: [{head}] ---> ({relation}) ---> [{tail}]

Evaluation Categories:
1. Is it FACTUALLY CORRECT and supported by the text? (Yes/No)
2. Is it a SPEECH WRAPPER (e.g., contains 'said', 'told', 'reported') or an OPINION/RUMOR? (Yes/No)
3. Is it a HALLUCINATION (contains words/facts NOT present or implied in the text)? (Yes/No)

Provide your analysis in this strict JSON format:
{{
    "factually_correct": true/false,
    "is_speech_or_opinion": true/false,
    "is_hallucination": true/false
}}
JSON:"""
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        res_json = response.json().get("response", "")
        import json
        data = json.loads(res_json)
        return data
    except Exception:
        # หากเกิดข้อผิดพลาดให้ถือว่าไม่ผ่านเพื่อความปลอดภัย
        return {"factually_correct": False, "is_speech_or_opinion": True, "is_hallucination": True}

# ==========================================
# 🧪 PIPELINE การทดลองเปรียบเทียบ 2 กลุ่ม
# ==========================================
def run_experiment_chapter4():
    print("📥 Loading BBC News Dataset (Config: 2024-01)...")
    dataset = load_dataset("RealTimeData/bbc_news_alltime", "2024-01", split="train")
    
    # ดึงมาทดสอบ 3 บทความเพื่อความรวดเร็ว (สามารถปรับเป็น 5 ได้)
    sample_size = 3 
    
    # ตัวนับสำหรับกลุ่มที่ 1: Llama 3 เปล่าๆ (ไม่มีตัวกรอง Double-Gate)
    g1_total_extracted = 0
    g1_true_positives = 0   # ถูกต้องตามจริง
    g1_hallucinations = 0   # มโน/หลอน
    
    # ตัวนับสำหรับกลุ่มที่ 2: Llama 3 + Double-Gate Filters (ระบบที่พัฒนาขึ้น)
    g2_total_extracted = 0
    g2_true_positives = 0
    g2_hallucinations = 0

    print(f"\n🚀 Starting Evaluation Pipeline on {sample_size} articles...")
    
    for idx in range(sample_size):
        print(f"\n📄 Processing Article {idx+1}/{sample_size}...")
        article_text = dataset[idx]["content"]
        
        # หั่นเป็นประโยคย่อย
        doc = nlp(article_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 15]
        
        # จำกัดจำนวนประโยคต่อบทความเพื่อไม่ให้รันนานเกินไป (เลือกมา 5 ประโยคเด่น)
        sentences = sentences[:5]
        
        for sent in sentences:
            # 1. ให้ LLM สกัดความสัมพันธ์ดิบออกมา
            output = ask_llm(sent)
            if not output:
                continue
            
            raw_triples = parse_triples(output)
            
            for head, rel_text, tail in raw_triples:
                # ทำความสะอาดคำนำหน้านามพื้นฐาน
                head_clean = re.sub(r'^(the|a|an)\s+', '', head.strip(), flags=re.IGNORECASE)
                tail_clean = re.sub(r'^(the|a|an)\s+', '', tail.strip(), flags=re.IGNORECASE)
                
                # กรองโครงสร้างไวยากรณ์พื้นฐาน (is_valid_entity, is_valid_relation)
                if not is_valid_entity(head_clean) or not is_valid_entity(tail_clean): continue
                if not is_valid_relation(rel_text): continue
                if head_clean.lower() == tail_clean.lower(): continue
                
                # -----------------------------------------------
                # กลุ่มที่ 1: แบบไม่มีตัวกรองเชิงตรรกะ (Baseline)
                # -----------------------------------------------
                g1_total_extracted += 1
                # ส่งให้กรรมการกลาง (Judge) ตรวจสอบคุณภาพ
                g1_judge = judge_triplet_accuracy(head_clean, rel_text, tail_clean, sent)
                
                if g1_judge.get("factually_correct") == True and g1_judge.get("is_speech_or_opinion") == False:
                    g1_true_positives += 1
                if g1_judge.get("is_hallucination") == True:
                    g1_hallucinations += 1
                
                # -----------------------------------------------
                # กลุ่มที่ 2: เปิดใช้งาน Double-Gate Filters (Proposed)
                # -----------------------------------------------
                # รันตัวกรอง evaluate_triplet_cloze ที่อยู่ในไฟล์ระบบของคุณ
                is_valid, _ = evaluate_triplet_cloze(head_clean, rel_text, tail_clean, sent)
                
                if is_valid: # ถ้าด่านตรวจยอมให้ผ่าน
                    g2_total_extracted += 1
                    # ส่งให้กรรมการกลาง (Judge) ตรวจสอบคุณภาพ
                    g2_judge = judge_triplet_accuracy(head_clean, rel_text, tail_clean, sent)
                    
                    if g2_judge.get("factually_correct") == True and g2_judge.get("is_speech_or_opinion") == False:
                        g2_true_positives += 1
                    if g2_judge.get("is_hallucination") == True:
                        g2_hallucinations += 1

    # ==========================================
    # 📊 คำนวณผลลัพธ์สุทธิตามสูตรคณิตศาสตร์
    # ==========================================
    # สูตร Precision = True Positives / Total Extracted
    g1_precision = (g1_true_positives / g1_total_extracted) * 100 if g1_total_extracted > 0 else 0
    g2_precision = (g2_true_positives / g2_total_extracted) * 100 if g2_total_extracted > 0 else 0
    
    # สูตร Hallucination Rate = Hallucinations / Total Extracted
    g1_halluc_rate = (g1_hallucinations / g1_total_extracted) * 100 if g1_total_extracted > 0 else 0
    g2_halluc_rate = (g2_hallucinations / g2_total_extracted) * 100 if g2_total_extracted > 0 else 0
    
    # สำหรับ Recall เนื่องจากไม่มี Ground Truth สมบูรณ์แบบ 
    # ทางวิชาการจะใช้จำนวน True Positive ของกลุ่มไม่มีตัวกรองเป็นฐานคำนวณอัตราการดึงกลับ
    g1_recall = 100.0 if g1_true_positives > 0 else 0
    g2_recall = (g2_true_positives / g1_true_positives) * 100 if g1_true_positives > 0 else 0
    
    # คำนวณ F1-Score
    g1_f1 = (2 * g1_precision * g1_recall) / (g1_precision + g1_recall) if (g1_precision + g1_recall) > 0 else 0
    g2_f1 = (2 * g2_precision * g2_recall) / (g2_precision + g2_recall) if (g2_precision + g2_recall) > 0 else 0

    # พิมพ์ตารางผลลัพธ์จริงออกมาสำหรับบทที่ 4
    print("\n" + "="*70)
    print("📊 REAL MATHEMATICAL RESULTS FOR TABLE 4.2")
    print("="*70)
    print(f"🔹 กลุ่มที่ 1: Llama 3 (ไม่มีตัวกรอง)")
    print(f"   - Total Extracted    : {g1_total_extracted} triplets")
    print(f"   - Triplet Precision  : {g1_precision:.1f}%")
    print(f"   - Triplet Recall     : {g1_recall:.1f}%")
    print(f"   - F1-Score           : {g1_f1:.1f}%")
    print(f"   - Hallucination Rate : {g1_alloc_rate if 'g1_alloc_rate' in locals() else g1_halluc_rate:.1f}%")
    print("-" * 70)
    print(f"🔥 กลุ่มที่ 2: Llama 3 + Double-Gate Filters (ระบบของคุณ)")
    print(f"   - Total Extracted    : {g2_total_extracted} triplets")
    print(f"   - Triplet Precision  : {g2_precision:.1f}%")
    print(f"   - Triplet Recall     : {g2_recall:.1f}%")
    print(f"   - F1-Score           : {g2_f1:.1f}%")
    print(f"   - Hallucination Rate : {g2_halluc_rate:.1f}%")
    print("="*70)

if __name__ == "__main__":
    run_experiment_chapter4()