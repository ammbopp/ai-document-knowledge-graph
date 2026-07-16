# evaluate_entity_resolution.py
import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

# นำเข้าโมดูลและฟังก์ชันแกนหลักจากระบบของคุณ
from src.relation_extraction_ai import extract_relations
from src.entity_resolution import EntityResolver

# ตั้งค่าสภาพแวดล้อมและโมเดลภาษา
nlp = spacy.load("en_core_web_sm")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# ==========================================
# 📋 ฟังก์ชันการทำ Text Chunking เหมือนสเปกของระบบหลัก
# ==========================================
def create_sliding_windows(sentences, window_size=3, overlap=1):
    chunks = []
    step = max(1, window_size - overlap)
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + window_size]
        chunks.append(" ".join(chunk_sentences))
        if i + window_size >= len(sentences):
            break
    return chunks

# ==========================================
# 🤖 LLM JUDGE: สร้างเฉลยกลุ่มเอนทิตีที่ถูกหลักข้อเท็จจริง
# ==========================================
def generate_gold_entity_clusters(text, unique_entities):
    """
    ให้ LLM ตรวจสอบเนื้อหาและจัดกลุ่มคำดิบ (Raw Strings) ทั้งหมดที่สกัดได้
    ว่าคำไหนหมายถึงสิ่งเดียวกันหรือคนเดียวกันในโลกจริง (Gold-Standard Grouping)
    """
    entity_list_str = "\n".join([f"- {ent} ({etype})" for ent, etype in unique_entities])
    
    prompt = f"""You are an expert NLP Entity Resolution validator. Analyze the text and the list of extracted raw entity strings. 
    Your task is to group the exact raw strings that refer to the SAME real-world object, person, location, or organization within this context.

    Text: "{text}"

    List of Raw Extracted Entities:
    {entity_list_str}

    Provide the correct semantic groups strictly in this JSON format. Every raw string from the input must belong to a group. If an entity is unique, put it in its own group:
    {{
       "Group_Canonical_Name_1": ["raw_string_A", "raw_string_B"],
       "Group_Canonical_Name_2": ["raw_string_C"]
    }}
    JSON:"""
    
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.0}}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return json.loads(response.json().get("response", "{}"))
    except Exception:
        return {}

# ==========================================
# 📐 MATHEMATICAL EVALUATION (คำนวณคู่ความสัมพันธ์เชิงสถิติ)
# ==========================================
def evaluate_pairwise_metrics(gold_clusters, pred_resolved_dict):
    """
    คำนวณประสิทธิภาพความแม่นยำของการจัดกลุ่มเอนทิตีด้วยวิธี Pairwise Matching Matrix
    """
    # 1. สร้างคู่อ้างอิงที่ถูกต้อง (Gold Pairs)
    gold_pairs = set()
    for _, items in gold_clusters.items():
        clean_items = [it.strip().lower() for it in items]
        for i in range(len(clean_items)):
            for j in range(i + 1, len(clean_items)):
                gold_pairs.add(frozenset([clean_items[i], clean_items[j]]))

    # 2. สร้างคู่ที่ระบบโมดูลทำนายออกมา (Predicted Pairs)
    pred_clusters = {}
    for raw_ent, resolved_canon in pred_resolved_dict.items():
        canon_key = resolved_canon.strip().lower()
        if canon_key not in pred_clusters:
            pred_clusters[canon_key] = []
        pred_clusters[canon_key].append(raw_ent.strip().lower())

    pred_pairs = set()
    for _, items in pred_clusters.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pred_pairs.add(frozenset([items[i], items[j]]))

    # 3. คำนวณตามสูตรคณิตศาสตร์สถิติ Precision / Recall / F1
    if not pred_pairs:
        precision = 100.0 if not gold_pairs else 0.0
    else:
        intersection = gold_pairs.intersection(pred_pairs)
        precision = (len(intersection) / len(pred_pairs)) * 100

    if not gold_pairs:
        recall = 100.0
    else:
        intersection = gold_pairs.intersection(pred_pairs)
        recall = (len(intersection) / len(gold_pairs)) * 100

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# ==========================================
# 🧪 PIPELINE การทดลองหลักแปรผัน THRESHOLD
# ==========================================
def run_threshold_experiment():
    print("📥 Loading BBC News Dataset (Config: 2025-01)...")
    dataset = load_dataset("RealTimeData/bbc_news_alltime", "2025-01", split="train")
    
    # ดึงข้อมูลรัน 3 บทความเพื่อเฉลี่ยผลสถิติที่เสถียร
    SAMPLE_SIZE = 3 
    thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    
    # ตัวแปรเก็บคะแนนสุทธิแยกตาม Threshold
    threshold_results = {t: {"p": [], "r": [], "f1": []} for t in thresholds}
    
    print(f"🚀 Starting Entity Resolution Threshold Optimization on {SAMPLE_SIZE} articles...\n")
    
    for idx in range(SAMPLE_SIZE):
        print(f"📄 Analyzing Article {idx+1}/{SAMPLE_SIZE} for Entity Resolution...")
        article_text = dataset[idx]["content"]
        
        # 1. แตกประโยคและทำ Sliding Window Chunking ตามสเปกระบบ
        doc = nlp(article_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 10]
        chunks = create_sliding_windows(sentences, window_size=3, overlap=1)
        
        # 2. รันสกัดความสัมพันธ์ดิบเพื่อเก็บรวบรวมคำนามเอนทิตีดิบทั้งหมดในกระบวนการ
        relations = []
        for chunk in chunks:
            extracted = extract_relations([chunk])
            relations.extend(extracted)
            
        # กรองข้อมูลเอนทิตีดิบที่เป็น Unique Items (เก็บทั้งชื่อและประเภท)
        raw_entities_pool = set()
        for r in relations:
            if r.get("confidence", 0.0) >= 0.7:
                raw_entities_pool.add((r["head"], r.get("head_type", "Entity")))
                raw_entities_pool.add((r["tail"], r.get("tail_type", "Entity")))
        
        raw_entities_list = list(raw_entities_pool)
        
        if not raw_entities_list:
            print("⚠️ ไม่พบเอนทิตีที่ผ่านการสกัดในบทความนี้ ข้ามไปบทความถัดไป...")
            continue
            
        # 3. ให้กรรมการกลาง (Judge) สร้างกลุ่มจัดสรรเอนทิตีที่แท้จริง
        gold_clusters = generate_gold_entity_clusters(article_text, raw_entities_list)
        
        # 4. ลูปแปรผันค่า Cosine Similarity Threshold ของตัวแปรทดลอง
        for t in thresholds:
            # ประกาศสร้างตัว Resolver ใหม่ทุกรอบเพื่อไม่ให้เกิดข้อมูลตกค้างข้ามฝั่ง (Clean State)
            resolver = EntityResolver(threshold=t)
            
            # บันทึกคำบอกจับคู่ที่สกัดได้
            pred_resolved_dict = {}
            for ent_str, ent_type in raw_entities_list:
                resolved_canonical = resolver.resolve(ent_str, ent_type)
                pred_resolved_dict[ent_str] = resolved_canonical
                
            # คำนวณหาคะแนน Matrix เชิงคู่วิเคราะห์ของเอกสารชิ้นนี้
            p, r, f1 = evaluate_pairwise_metrics(gold_clusters, pred_resolved_dict)
            
            threshold_results[t]["p"].append(p)
            threshold_results[t]["r"].append(r)
            threshold_results[t]["f1"].append(f1)

    # ==========================================
    # 📊 ประมวลผลลัพธ์และพิมพ์ตารางรายงานบทที่ 4.4
    # ==========================================
    print("\n" + "="*80)
    print("📊 REAL MATHEMATICAL RESULTS FOR TABLE 4.3 (SECTION 4.4)")
    print("="*80)
    print("| Similarity Threshold | Clustering Precision | Clustering Recall | F1-Score |")
    print("| :--- | :---: | :---: | :---: |")
    
    for t in thresholds:
        mean_p = sum(threshold_results[t]["p"]) / len(threshold_results[t]["p"]) if threshold_results[t]["p"] else 0.0
        mean_r = sum(threshold_results[t]["r"]) / len(threshold_results[t]["r"]) if threshold_results[t]["r"] else 0.0
        mean_f1 = sum(threshold_results[t]["f1"]) / len(threshold_results[t]["f1"]) if threshold_results[t]["f1"] else 0.0
        
        # ทำสัญลักษณ์ดารากำกับค่าที่ระบบเลือกใช้จริง (0.85)
        marker = " (ค่าที่ระบบเลือกใช้)" if t == 0.85 else ""
        print(f"| Threshold = {t:.2f}{marker} | {mean_p:.1f}% | {mean_r:.1f}% | {mean_f1:.1f}% |")
        
    print("="*80)

if __name__ == "__main__":
    run_threshold_experiment()