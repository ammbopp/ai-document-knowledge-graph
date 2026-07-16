# evaluate_summarization.py
import os
import re
import json
import time
import requests
import pandas as pd
import spacy
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

# นำเข้าโมดูลจากระบบหลักของคุณ
from src.relation_extraction_ai import extract_relations
from src.entity_resolution import EntityResolver

# ตั้งค่าสภาพแวดล้อม
nlp = spacy.load("en_core_web_sm")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# เรียกใช้งาน ROUGE Scorer และ Entity Resolver (Threshold 0.85)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
resolver = EntityResolver(threshold=0.85)

# ==========================================
# 📋 ฟังก์ชันตัวช่วยสร้าง Chunk และปรับปรุงระบบหลัก
# ==========================================
def create_sliding_windows(sentences, window_size=3, overlap=1):
    chunks = []
    step = max(1, window_size - overlap)
    for i in range(0, len(sentences), step):
        chunks.append(" ".join(sentences[i : i + window_size]))
        if i + window_size >= len(sentences):
            break
    return chunks

# ==========================================
# 📝 วิธีที่ 1: Baseline Text-based Summarization (สรุปจากข้อความดิบ)
# ==========================================
def generate_text_based_summary(original_text):
    prompt = f"""You are an expert AI summarizer. Your task is to generate a structured, executive-level summary of the text below using clear bullet points.
    DO NOT invent or add external knowledge. Ground every fact strictly in the input.

    === ORIGINAL TEXT ===
    {original_text}

    Summary:"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception:
        return ""

# ==========================================
# 📝 วิธีที่ 2: Proposed Graph-Augmented Summarization (สรุปผสานโครงสร้างกราฟ)
# ==========================================
def generate_graph_augmented_summary(original_text, resolved_relations):
    graph_context = ""
    grouped_relations = {}
    for r in resolved_relations:
        sent = r.get("source_sentence", "General")
        if sent not in grouped_relations:
            grouped_relations[sent] = []
        grouped_relations[sent].append(f"{r['head']} -> {r['relation']} -> {r['tail']}")

    for idx, (sent, triplets) in enumerate(grouped_relations.items()):
        graph_context += f"\n[Topic Shift {idx+1}]\n"
        for t in triplets:
            graph_context += f"- {t}\n"

    prompt = f"""You are an expert AI summarizer. Your task is to generate a highly accurate, structured summary using clear bullet points.
    You MUST synthesize the summary using BOTH the 'Original Text' and the extracted 'Knowledge Graph Context' to prevent hallucinations.
    Ensure you capture exact relationships and track topic shifts. Ground every fact strictly in the inputs.

    === ORIGINAL TEXT ===
    {original_text}

    === KNOWLEDGE GRAPH CONTEXT (Chronological Topic Shifts) ===
    {graph_context}

    Summary:"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception:
        return ""

# ==========================================
# 🤖 LLM JUDGE: ฟังก์ชันตรวจจับการมโนและความสอดคล้องข้อเท็จจริง
# ==========================================
def judge_summary_faithfulness(original_text, generated_summary):
    """
    ใช้ LLM ทำหน้าที่เป็นกรรมการตรวจหาจุดมโน (Hallucinations) ในบทสรุป
    โดยส่งกลับมาเป็นคะแนนความสอดคล้อง (Faithfulness Score: 0.0 - 1.0)
    """
    prompt = f"""You are an advanced quality assurance inspector for text summarization systems. Your job is to strictly evaluate the Faithfulness of the Generated Summary against the Original Text.
    Check for any factual contradictions, ungrounded exaggerations, or external claims introduced by the AI (Hallucinations).

    === ORIGINAL TEXT ===
    {original_text}

    === GENERATED SUMMARY ===
    {generated_summary}

    Based on the rules, judge the summary and output a JSON object containing:
    1. "faithfulness_score": a float between 0.0 (completely hallucinated/wrong) and 1.0 (perfectly faithful and grounded).
    2. "hallucination_count": an integer representing the exact number of ungrounded or invented statements found.

    Strict JSON Output Format:
    {{
        "faithfulness_score": <float>,
        "hallucination_count": <integer>
    }}
    JSON:"""
    
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.0}}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        data = json.loads(response.json().get("response", "{}"))
        return data.get("faithfulness_score", 0.5), data.get("hallucination_count", 0)
    except Exception:
        return 0.5, 1

# ==========================================
# 🧪 PIPELINE การทดลองหลัก
# ==========================================
def run_summarization_experiment():
    print("📥 Loading BBC News Dataset (Config: 2025-01)...")
    dataset = load_dataset("RealTimeData/bbc_news_alltime", "2025-01", split="train")
    
    SAMPLE_SIZE = 3 # รันจำนวน 3 บทความเพื่อเฉลี่ยผลลัพธ์
    
    # ตัวแปรเก็บคะแนนรายกลุ่ม
    g1_metrics = {"faith": [], "r1": [], "r2": [], "rl": []}
    g2_metrics = {"faith": [], "r1": [], "r2": [], "rl": []}
    
    print(f"🚀 Running Summarization Quality Experiment on {SAMPLE_SIZE} articles...\n")
    
    for idx in range(SAMPLE_SIZE):
        print(f"📄 Analyzing Article {idx+1}/{SAMPLE_SIZE} for Summarization Modules...")
        article_text = dataset[idx]["content"]
        ground_truth = dataset[idx]["title"] # ใช้พาดหัวข่าวเป็น Ground-Truth อ้างอิง
        
        # --- เตรียมข้อมูลล่วงหน้าและการขึ้นโครงกราฟความรู้สำหรับกลุ่ม Proposed ---
        doc = nlp(article_text)
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 10]
        chunks = create_sliding_windows(sentences, window_size=3, overlap=1)
        
        relations = []
        for chunk in chunks:
            extracted = extract_relations([chunk])
            relations.extend(extracted)
            
        # ทำ Entity Resolution 
        resolved_relations = []
        seen_pairs = set()
        for r in relations:
            if r.get("confidence", 0.0) >= 0.7:
                h = resolver.resolve(r["head"], r.get("head_type", "Entity"))
                t = resolver.resolve(r["tail"], r.get("tail_type", "Entity"))
                if h.lower() == t.lower(): continue
                pair = frozenset([h.lower(), t.lower()])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    nr = r.copy()
                    nr["head"], nr["tail"] = h, t
                    resolved_relations.append(nr)
                    
        # --------------------------------------------------
        # กลุ่มที่ 1: Text-based Summarization (Baseline)
        # --------------------------------------------------
        g1_summary = generate_text_based_summary(article_text)
        g1_faith, _ = judge_summary_faithfulness(article_text, g1_summary)
        g1_rouge = scorer.score(ground_truth, g1_summary)
        
        g1_metrics["faith"].append(g1_faith)
        g1_metrics["r1"].append(g1_rouge['rouge1'].fmeasure)
        g1_metrics["r2"].append(g1_rouge['rouge2'].fmeasure)
        g1_metrics["rl"].append(g1_rouge['rougeL'].fmeasure)
        
        # --------------------------------------------------
        # กลุ่มที่ 2: Graph-Augmented Summarization (Proposed)
        # --------------------------------------------------
        g2_summary = generate_graph_augmented_summary(article_text, resolved_relations)
        g2_faith, _ = judge_summary_faithfulness(article_text, g2_summary)
        g2_rouge = scorer.score(ground_truth, g2_summary)
        
        g2_metrics["faith"].append(g2_faith)
        g2_metrics["r1"].append(g2_rouge['rouge1'].fmeasure)
        g2_metrics["r2"].append(g2_rouge['rouge2'].fmeasure)
        g2_metrics["rl"].append(g2_rouge['rougeL'].fmeasure)

    # ==========================================
    # 📊 ประมวลผลลัพธ์และพิมพ์ตารางรายงานบทที่ 4.5
    # ==========================================
    print("\n" + "="*80)
    print("📊 REAL MATHEMATICAL RESULTS FOR TABLE 4.5 (SECTION 4.5)")
    print("="*80)
    print("| วิธีการสร้างบทสรุป | Faithfulness Score (0-1) | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    # กลุ่มที่ 1
    m1_faith = sum(g1_metrics["faith"]) / SAMPLE_SIZE
    m1_r1 = sum(g1_metrics["r1"]) / SAMPLE_SIZE
    m1_r2 = sum(g1_metrics["r2"]) / SAMPLE_SIZE
    m1_rl = sum(g1_metrics["rl"]) / SAMPLE_SIZE
    print(f"| Text-based Summarization | {m1_faith:.2f} | {m1_r1:.4f} | {m1_r2:.4f} | {m1_rl:.4f} |")
    
    # กลุ่มที่ 2
    m2_faith = sum(g2_metrics["faith"]) / SAMPLE_SIZE
    m2_r1 = sum(g2_metrics["r1"]) / SAMPLE_SIZE
    m2_r2 = sum(g2_metrics["r2"]) / SAMPLE_SIZE
    m2_rl = sum(g2_metrics["rl"]) / SAMPLE_SIZE
    print(f"| Graph-Augmented Summarization | {m2_faith:.2f} | {m2_r1:.4f} | {m2_r2:.4f} | {m2_rl:.4f} |")
    
    print("="*80)

if __name__ == "__main__":
    run_summarization_experiment()