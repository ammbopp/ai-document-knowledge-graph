# evaluate_system.py
import os
import time
import json
import pandas as pd
import networkx as nx
import requests
import spacy
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer

# นำเข้าโมดูลระบบเดิมจากโปรเจกต์ของคุณ
from src.relation_extraction_ai import extract_relations
from src.entity_resolution import EntityResolver
from src.graph_builder import build_graph, visualize_graph
from src.graph_insight import analyze_graph

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
SAMPLE_SIZE = 5  # จำนวนเอกสารที่ต้องการดึงมาทดสอบ (ปรับเพิ่ม-ลดได้ตามต้องการ)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# โหลดโมเดลภาษาสำหรับทำ Sentence Segmentation 
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# เรียกใช้งาน Entity Resolver จากโครงงานเดิม (Threshold 0.85)
resolver = EntityResolver(threshold=0.85)
# เรียกตัววัดผล ROUGE Score สำหรับวัดคุณภาพของ Summary
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# ==========================================
# HELPER FUNCTIONS (ดึงมาจากสถาปัตยกรรมระบบหลัก)
# ==========================================
def resolve_coreferences_with_llm(text):
    prompt = f"""You are a precise NLP Coreference Resolution engine. 
    Your task is to rewrite the text by replacing pronouns (he, she, it, they, his, her, their) with their exact explicit noun antecedents.

    CRITICAL SAFETY RULES:
    1. STRICT PLURAL MATCHING: If the pronoun is plural ("they", "their"), it MUST be replaced by a plural group.
    2. CONTEXTUAL LOGIC: Read the entire sentence to ensure the replacement makes logical sense. Do not blindly assign all pronouns to the most frequent name.

    Original Text: {text}
    Rewritten Text:"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get("response", text).strip()
    except Exception as e:
        print(f"⚠️ Coref Failed: {e}")
        return text

def create_sliding_windows(sentences, window_size=3, overlap=1):
    chunks = []
    step = max(1, window_size - overlap) 
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + window_size]
        chunks.append(" ".join(chunk_sentences))
        if i + window_size >= len(sentences):
            break
    return chunks

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

    prompt = f"""You are an expert AI summarizer. Your task is to generate a highly accurate, structured summary.
    To prevent hallucinations and capture topic shifts perfectly, you MUST synthesize the summary using BOTH the 'Original Text' and the extracted 'Knowledge Graph Context'.

    === ORIGINAL TEXT ===
    {original_text}

    === KNOWLEDGE GRAPH CONTEXT (Chronological Topic Shifts) ===
    {graph_context}

    === INSTRUCTIONS ===
    1. Write a clear, executive-level summary of the text.
    2. Use the Knowledge Graph Context to ensure you capture the exact relationships.
    3. DO NOT invent, hallucinate, or add external knowledge. Ground every fact in the provided inputs.

    Summary:"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ Summary Failed: {e}")
        return "Error generating summary"


# ==========================================
# MAIN EVALUATION PIPELINE
# ==========================================
def run_evaluation():
    print("📥 Loading BBC News dataset ('RealTimeData/bbc_news_alltime') from Hugging Face...")
    
    # ระบุชื่อ Sub-config 
    dataset = load_dataset("RealTimeData/bbc_news_alltime", "2025-01", split="train")

    # --- ตรวจจับหาคอลัมน์โดยอัตโนมัติ (Dynamic Column Detection) ---
    all_cols = dataset.column_names
    text_candidates = ["content", "text", "article", "story", "body"]
    summary_candidates = ["summary", "highlights", "title", "description", "headline"]
    
    text_col = next((c for c in text_candidates if c in all_cols), None)
    summary_col = next((c for c in summary_candidates if c in all_cols and c != text_col), None)
    
    # กรณีหาไม่เจอตาม Candidate ให้ใช้ Default สองคอลัมน์แรก
    if not text_col:
        text_col = all_cols[0]
    if not summary_col:
        summary_col = all_cols[1] if len(all_cols) > 1 else text_col
        
    print(f"🔑 Dynamic Mapping -> Text Column: '{text_col}' | Ground-Truth Column: '{summary_col}'\n")
    
    results = []
    print(f"🚀 Starting evaluation pipeline on {SAMPLE_SIZE} sampled BBC documents...\n")
    
    for idx in range(SAMPLE_SIZE):
        print(f"📄 Processing Document {idx+1}/{SAMPLE_SIZE}...")
        article = dataset[idx][text_col]
        ground_truth_highlight = dataset[idx][summary_col]
        
        word_count = len(article.split())
        
        # --- [1] TEST: Coreference Resolution Latency ---
        start_time = time.time()
        resolved_text = resolve_coreferences_with_llm(article)
        coref_latency = time.time() - start_time
        
        # --- [2] TEST: Chunking Process ---
        start_time = time.time()
        doc = nlp(resolved_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 10]
        chunks = create_sliding_windows(sentences, window_size=3, overlap=1)
        chunking_latency = time.time() - start_time
        
        # --- [3] TEST: Relation Extraction & Double-Gate Filter ---
        start_time = time.time()
        relations = []
        for chunk in chunks:
            extracted = extract_relations([chunk])
            relations.extend(extracted)
        extraction_latency = time.time() - start_time
        raw_triplet_count = len(relations)
        
        # กรองตัวที่มีความมั่นใจผ่านเกณฑ์ของระบบ (Double-Gate Filters)
        usable_relations = [r for r in relations if r.get("confidence", 0.0) >= 0.7]
        filtered_triplet_count = len(usable_relations)
        
        # --- [4] TEST: 2-Pass Hybrid Entity Resolution ---
        start_time = time.time()
        resolved_relations = []
        seen_entity_pairs = set()
        for r in usable_relations:
            resolved_head = resolver.resolve(r["head"], r.get("head_type", "Entity"))
            resolved_tail = resolver.resolve(r["tail"], r.get("tail_type", "Entity"))
            
            if resolved_head.lower() == resolved_tail.lower():
                continue
            pair_key = frozenset([resolved_head.lower(), resolved_tail.lower()])
            if pair_key not in seen_entity_pairs:
                seen_entity_pairs.add(pair_key)
                new_r = r.copy()
                new_r["head"] = resolved_head
                new_r["tail"] = resolved_tail
                resolved_relations.append(new_r)
        entity_res_latency = time.time() - start_time
        
        # --- [5] TEST: Graph Construction & Insights ---
        G = build_graph(resolved_relations)
        insights = analyze_graph(G)
        
        node_count = insights.get("total_entities", 0)
        edge_count = insights.get("total_relations", 0)
        graph_density = insights.get("density", 0.0)
        
        # --- [6] TEST: Graph-Augmented Summarization & ROUGE Score ---
        start_time = time.time()
        generated_summary = generate_graph_augmented_summary(article, resolved_relations)
        summary_latency = time.time() - start_time
        
        # คำนวณ ROUGE score เปรียบเทียบกับ Ground-Truth ของข่าว BBC
        rouge_scores = scorer.score(ground_truth_highlight, generated_summary)
        
        # บันทึกผลลัพธ์
        doc_result = {
            "doc_index": idx + 1,
            "word_count": word_count,
            "num_chunks": len(chunks),
            # Latency Metrics (Seconds)
            "latency_coref_sec": round(coref_latency, 2),
            "latency_chunking_sec": round(chunking_latency, 2),
            "latency_extraction_sec": round(extraction_latency, 2),
            "latency_entity_res_sec": round(entity_res_latency, 2),
            "latency_summary_sec": round(summary_latency, 2),
            "total_latency_sec": round(coref_latency + chunking_latency + extraction_latency + entity_res_latency + summary_latency, 2),
            # Graph Architecture Metrics
            "raw_triplets_extracted": raw_triplet_count,
            "filtered_triplets_passed": filtered_triplet_count,
            "final_graph_nodes": node_count,
            "final_graph_edges": edge_count,
            "graph_density": graph_density,
            # ROUGE Quality Metrics (F1-Score)
            "rouge1_f1": round(rouge_scores['rouge1'].fmeasure, 4),
            "rouge2_f1": round(rouge_scores['rouge2'].fmeasure, 4),
            "rougeL_f1": round(rouge_scores['rougeL'].fmeasure, 4)
        }
        results.append(doc_result)
        print(f"✨ Finished Doc {idx+1} | Total Time: {doc_result['total_latency_sec']}s | ROUGE-L: {doc_result['rougeL_f1']}")

    # ==========================================
    # DATA AGGREGATION & REPORTING
    # ==========================================
    df = pd.DataFrame(results)
    
    # บันทึกข้อมูลดิบลงไฟล์ CSV เพื่อให้คุณนำไปทำแผนภูมิเส้นหรือแท่งประกอบบทที่ 4
    output_csv = "evaluation_bbc_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved raw evaluation data to '{output_csv}'")
    
    # สรุปค่าเฉลี่ยทางสถิติสำหรับเขียนบทที่ 4
    print("\n" + "="*60)
    print("📊 FINAL SYSTEM EVALUATION SUMMARY (BBC NEWS DATASET)")
    print("="*60)
    print(f"📄 Total Test Samples      : {SAMPLE_SIZE} articles")
    print(f"⏱️ Avg Coref Latency      : {df['latency_coref_sec'].mean():.2f} sec")
    print(f"⏱️ Avg Extraction Latency : {df['latency_extraction_sec'].mean():.2f} sec")
    print(f"⏱️ Avg Total Pipeline Time: {df['total_latency_sec'].mean():.2f} sec")
    print("-" * 60)
    print(f"📐 Avg Raw Triplet Extracted: {df['raw_triplets_extracted'].mean():.1f}")
    print(f"📐 Avg Triplets Passed Gate : {df['filtered_triplets_passed'].mean():.1f}")
    print(f"🕸️ Avg Graph Nodes (Entities): {df['final_graph_nodes'].mean():.1f}")
    print(f"🕸️ Avg Graph Edges (Relations): {df['final_graph_edges'].mean():.1f}")
    print(f"🕸️ Avg Graph Density       : {df['graph_density'].mean():.4f}")
    print("-" * 60)
    print(f"📝 Avg ROUGE-1 F1-Score   : {df['rouge1_f1'].mean():.4f}")
    print(f"📝 Avg ROUGE-2 F1-Score   : {df['rouge2_f1'].mean():.4f}")
    print(f"📝 Avg ROUGE-L F1-Score   : {df['rougeL_f1'].mean():.4f}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()