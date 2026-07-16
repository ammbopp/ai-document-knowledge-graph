import os
import warnings
import logging
import requests

# 🔥 ปิดคำเตือนกวนใจทั้งหมด
os.environ["TRANSFORMERS_VERBOSITY"] = "error" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import spacy
import pandas as pd
import time
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

from src.relation_extraction_ai import extract_relations
from src.entity_resolution import EntityResolver
from src.graph_builder import build_graph, visualize_graph
from src.graph_insight import analyze_graph

# 1. ตั้งค่าหน้าเว็บ (Page Config)
st.set_page_config(page_title="AI Document Graph", layout="wide", page_icon="🧠")

# ==========================================
# 🧠 เริ่มต้นระบบความจำอย่างเสถียร (Initialize Session State)
# ==========================================
if "main_text" not in st.session_state:
    st.session_state.main_text = ""
    st.session_state.analyzed = False
    st.session_state.resolved_relations = []
    st.session_state.final_summary = ""
    st.session_state.insights = {}
    st.session_state.html_data = ""
    st.session_state.download_filename = "knowledge_graph.html"
    
    # ตัวจำค่าสำหรับการเช็คเปลี่ยนอินพุตเพื่อ Auto-Reset
    st.session_state.last_input_method = ""
    st.session_state.last_url = ""
    st.session_state.last_uploaded_file_name = ""
    st.session_state.last_pasted_text = ""

# ==========================================
# โหลดโมเดล (ใช้ Cache เพื่อความรวดเร็ว)
# ==========================================
st.title("🧠 AI Knowledge Graph Extractor")
st.markdown("วิเคราะห์สกัดความสัมพันธ์จากข้อความและสร้างเป็น Mind Map แบบ Interactive")

# 2. โหลดโมเดล Entity Resolution
@st.cache_resource
def get_resolver():
    return EntityResolver(threshold=0.85)

resolver = get_resolver()

# ==========================================
# ฟังก์ชันตัวช่วย: Web Scraping ดึงเนื้อหาจาก URL
# ==========================================
def fetch_content_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
            
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if not text.strip():
            text = soup.get_text(separator="\n").strip()
            
        return text
    except Exception as e:
        st.error(f"❌ ไม่สามารถดึงเนื้อหาจากลิงก์ได้: {e}")
        return ""

# ==========================================
# ฟังก์ชันตัวช่วย: LLM Coreference Resolution
# ==========================================
def resolve_coreferences_with_llm(text):
    prompt = f"""You are a precise NLP Coreference Resolution engine. 
    Your task is to rewrite the text by replacing pronouns (he, she, it, they, his, her, their) with their exact explicit noun antecedents.

    CRITICAL SAFETY RULES:
    1. STRICT PLURAL MATCHING: If the pronoun is plural ("they", "their"), it MUST be replaced by a plural group (e.g., "victims", "survivors", "lawmakers"), NEVER by a single individual's name.
    2. CONTEXTUAL LOGIC: Read the entire sentence to ensure the replacement makes logical sense. Do not blindly assign all pronouns to the most frequent name.
    3. Keep all other words and sentence structures exactly unchanged. Do not add commentary.

    Original Text: {text}
    Rewritten Text:"""
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        resolved_text = response.json().get("response", text).strip()
        return resolved_text
    except Exception as e:
        print(f"⚠️ LLM Coreference Failed: {e}")
        return text

# ==========================================
# ฟังก์ชันตัวช่วย: Sliding Window
# ==========================================
def create_sliding_windows(sentences, window_size=3, overlap=1):
    chunks = []
    step = max(1, window_size - overlap) 
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + window_size]
        chunk_text = " ".join(chunk_sentences)
        chunks.append(chunk_text)
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
2. Use the Knowledge Graph Context to ensure you capture the exact relationships and track how the topic shifts from beginning to end.
3. DO NOT invent, hallucinate, or add external knowledge. Ground every fact in the provided inputs.
4. Format the output with clear bullet points or short paragraphs for readability.

Summary:"""
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        summary = response.json().get("response", "").strip()
        return summary
    except Exception as e:
        print(f"⚠️ Graph-Augmented Summarization Failed: {e}")
        return "ไม่สามารถสร้างคำสรุปได้เนื่องจากเกิดข้อผิดพลาดในการเชื่อมต่อกับ LLM"

# ==========================================
# ส่วนรับข้อมูล (Input Section)
# ==========================================

input_method = st.radio(
    "👉 เลือกวิธีใส่ข้อมูล (Choose Input Method):",
    ("📝 วางข้อความ (Paste Text)", "📂 อัปโหลดไฟล์ (Upload .txt File)", "🌐 ดึงเนื้อหาจากลิงก์ (Fetch from URL)"),
    horizontal=True
)

# รีเซ็ตสถานะเมื่อสลับประเภท Input Method
if input_method != st.session_state.last_input_method:
    st.session_state.main_text = ""
    st.session_state.analyzed = False
    st.session_state.last_input_method = input_method

# 📂 การจัดการไฟล์อัปโหลด
if input_method == "📂 อัปโหลดไฟล์ (Upload .txt File)":
    uploaded_file = st.file_uploader("Upload your .txt file here", type=["txt"])
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_uploaded_file_name:
            st.session_state.main_text = uploaded_file.getvalue().decode("utf-8")
            st.session_state.last_uploaded_file_name = uploaded_file.name
            st.session_state.download_filename = f"{uploaded_file.name.replace('.txt', '')}_graph.html"
            st.session_state.analyzed = False
    else:
        if st.session_state.main_text != "":
            st.session_state.main_text = ""
            st.session_state.last_uploaded_file_name = ""
            st.session_state.analyzed = False

# 🌐 การจัดการดึงเนื้อหาจากลิงก์ (รันสแครปปิ้งเฉพาะตอน URL เปลี่ยนแปลงเท่านั้น)
elif input_method == "🌐 ดึงเนื้อหาจากลิงก์ (Fetch from URL)":
    url_input = st.text_input("Paste news or article URL here", placeholder="https://example.com/news-article")
    if url_input.strip():
        if url_input.strip() != st.session_state.last_url:
            with st.spinner("กำลังดึงข้อมูลและคัดกรองเนื้อหาจากลิงก์... 🌐"):
                fetched_text = fetch_content_from_url(url_input.strip())
                if fetched_text:
                    st.session_state.main_text = fetched_text
                    st.session_state.last_url = url_input.strip()
                    st.session_state.download_filename = "news_url_graph.html"
                    st.session_state.analyzed = False
                    st.success("✅ ดึงเนื้อหาสำเร็จ!")
    else:
        if st.session_state.main_text != "":
            st.session_state.main_text = ""
            st.session_state.last_url = ""
            st.session_state.analyzed = False

# 📝 การจัดการวางข้อความดิบ
else:
    pasted_text = st.text_area("Paste your document text here", height=200, placeholder="พิมพ์หรือวางข้อความภาษาอังกฤษที่นี่...")
    if pasted_text.strip():
        if pasted_text.strip() != st.session_state.last_pasted_text:
            st.session_state.main_text = pasted_text.strip()
            st.session_state.last_pasted_text = pasted_text.strip()
            st.session_state.download_filename = "custom_text_graph.html"
            st.session_state.analyzed = False
    else:
        if st.session_state.main_text != "":
            st.session_state.main_text = ""
            st.session_state.last_pasted_text = ""
            st.session_state.analyzed = False

# ==========================================
# ส่วนวิเคราะห์และสร้างกราฟ (รันผ่านหน่วยความจำหลัก)
# ==========================================
if st.session_state.main_text:
    word_count = len(st.session_state.main_text.split())
    char_count = len(st.session_state.main_text)
    
    col_word, col_char = st.columns(2)
    col_word.metric("📝 จำนวนคำทั้งหมด (Word Count)", f"{word_count:,}")
    col_char.metric("🔤 จำนวนตัวอักษรทั้งหมด (Character Count)", f"{char_count:,}")

    with st.expander("📄 ดูเนื้อหาที่ถูกดึงมาวิเคราะห์ (Extracted Content / Original Document)"):
        st.write(st.session_state.main_text)

    # ปุ่มสั่งรัน AI
    if st.button("🚀 Analyze & Generate Graph", type="primary", use_container_width=True):
        start_time = time.time()
        progress_bar = st.progress(0, text="เตรียมการวิเคราะห์...")
        
        with st.status("AI is processing the document... Please wait ⏳", expanded=True) as status:
            
            # --- ขั้นตอนที่ 1: LLM Coreference Resolution ---
            st.write("🔍 1. กำลังให้ AI แก้คำสรรพนาม (LLM Coreference Resolution)...")
            resolved_text = resolve_coreferences_with_llm(st.session_state.main_text)
            progress_bar.progress(5, text="แก้คำสรรพนามเสร็จสิ้น (5%)")
            
            # --- ขั้นตอนที่ 2: Sentence Segmentation & Chunking ---
            st.write("✂️ 2. กำลังตัดคำ แบ่งประโยค และทำ Text Chunking...")
            nlp = spacy.load("en_core_web_sm")
            
            doc = nlp(resolved_text) 
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 10]
            chunks = create_sliding_windows(sentences, window_size=3, overlap=1)
            
            print("\n" + "="*50)
            print("📦 [Console] 2. Text Chunking (Sliding Window)")
            print("="*50)
            print(f"📦 Total Chunks: {len(chunks)}")
            
            progress_bar.progress(10, text=f"แบ่งข้อมูลเป็น {len(chunks)} Chunks (10%)")
            
            # --- ขั้นตอนที่ 3: Relation Extraction ---
            st.write("🤖 3. กำลังให้ LLM สกัดความสัมพันธ์ (Relation Extraction)...")
            relations = []
            total_chunks = len(chunks)
            
            print("\n" + "="*50)
            print("🤖 [Console] 3. Relation Extraction (via LLM)")
            print("="*50)
            
            if total_chunks > 0:
                for i, chunk in enumerate(chunks):
                    extracted = extract_relations([chunk])
                    relations.extend(extracted)
                    
                    current_percent = 10 + int(70 * ((i + 1) / total_chunks))
                    progress_bar.progress(current_percent, text=f"กำลังสกัดความสัมพันธ์... Chunk ({i+1}/{total_chunks}) - {current_percent}%")
            else:
                progress_bar.progress(80, text="ข้ามการสกัดความสัมพันธ์ (80%)")

            usable_relations = [r for r in relations if r.get("confidence", 0.0) >= 0.7]
            
            # --- ขั้นตอนที่ 4: Entity Resolution ---
            st.write("🔍 4. กำลังยุบรวมคำที่ความหมายเหมือนกัน (Entity Resolution)...")
            resolved_relations = []
            seen_entity_pairs = set() 
            total_usable = len(usable_relations)

            if total_usable > 0:
                print("\n" + "="*50)
                print("📊 [Console] 4. สรุปเส้นความสัมพันธ์ (Final Edges)")
                print("="*50)
                
                for i, r in enumerate(usable_relations):
                    # รันและหาชื่อ canonical พร้อมระบุ Type ทันทีในการเรียกครั้งเดียว
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
                        print(f"🔗 [Node] {new_r['head']}  --({new_r['relation']})-->  [Node] {new_r['tail']}")
                    
                    current_percent = 80 + int(15 * ((i + 1) / total_usable))
                    progress_bar.progress(current_percent, text=f"กำลังคลีนข้อมูล... ({i+1}/{total_usable}) - {current_percent}%")
                print("="*50 + "\n")

            # --- ขั้นตอนที่ 5: Build Graph ---
            st.write("🎨 5. กำลังวิเคราะห์ศูนย์กลางและสร้าง Knowledge Graph...")
            G = build_graph(resolved_relations)
            output_path = "output/web_graph.html"
            os.makedirs("output", exist_ok=True)
            visualize_graph(G, output_file=output_path)
            
            insights = analyze_graph(G)
            progress_bar.progress(90, text="สร้าง Knowledge Graph เสร็จสิ้น (90%)")

            # --- ขั้นตอนที่ 6: Graph-Augmented Summarization ---
            st.write("📝 6. กำลังสรุปความด้วย Graph-Augmented LLM...")
            final_summary = generate_graph_augmented_summary(st.session_state.main_text, resolved_relations)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes} นาที {int(seconds)} วินาที" if minutes > 0 else f"{seconds:.2f} วินาที"
            
            progress_bar.progress(100, text=f"✅ วิเคราะห์ข้อมูลเสร็จสิ้นสมบูรณ์ (ใช้เวลาทั้งหมด ⏱️ {time_str})")
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # บันทึกค่าลงความจำถาวรหลังทำงานเสร็จสิ้น
        st.session_state.resolved_relations = resolved_relations
        st.session_state.final_summary = final_summary
        st.session_state.insights = insights
        with open(output_path, "r", encoding="utf-8") as f:
            st.session_state.html_data = f.read()
        st.session_state.analyzed = True
        
        # บังคับอัปเดตหน้าจอทันทีเพื่อให้ Tabs แสดงผลทันใจ
        st.rerun()

    # ==========================================
    # การแสดงผลลัพธ์แยกตาม Tabs (ทำงานอย่างมั่นคงผ่านความจำหลัก)
    # ==========================================
    if st.session_state.analyzed:
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Knowledge Graph", "📈 Graph Insights", "📝 Extracted Data", "📑 Executive Summary"])
        
        # 🟢 Tab 1: แสดงกราฟ และ ปุ่มดาวน์โหลด
        with tab1:
            st.subheader("Interactive Knowledge Graph")
            st.download_button(
                label="💾 Download Graph (HTML)",
                data=st.session_state.html_data,
                file_name=st.session_state.download_filename,
                mime="text/html",
                type="secondary"
            )
            components.html(st.session_state.html_data, height=600, scrolling=False)

        # 🟢 Tab 2: แสดงสถิติเชิงลึก
        with tab2:
            st.subheader("Graph Analysis & Insights")
            insights_data = st.session_state.insights
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Entities (Nodes)", insights_data.get("total_entities", 0))
            col2.metric("Total Relations (Edges)", insights_data.get("total_relations", 0))
            col3.metric("Graph Density", insights_data.get("density", 0))
            
            st.markdown("#### 🌟 Top Entities")
            if "top_entity_by_degree" in insights_data:
                st.info(f"**Main Entity (Hub):** {insights_data['top_entity_by_degree']['entity']} (Score: {insights_data['top_entity_by_degree']['score']})")
                
            with st.expander("ดูข้อมูลวิเคราะห์เชิงลึกแบบ JSON"):
                st.json(insights_data)

        # 🟢 Tab 3: แสดงตารางข้อมูลดิบ
        with tab3:
            st.subheader("Extracted Relations Table")
            if st.session_state.resolved_relations:
                df = pd.DataFrame(st.session_state.resolved_relations)
                if not df.empty and all(k in df.columns for k in ["head", "relation", "tail", "source_sentence"]):
                    df_display = df[["head", "relation", "tail", "source_sentence"]]
                    df_display.columns = ["Subject (Head)", "Relation", "Object (Tail)", "Source Sentence"]
                    st.dataframe(df_display, use_container_width=True)
            else:
                st.warning("No relations were extracted. ลองปรับข้อความให้เป็นประโยคที่สมบูรณ์ขึ้นครับ")

        # 🟢 Tab 4: แสดงคำสรุปจาก Dual Encoders
        with tab4:
            st.subheader("Graph-Augmented Executive Summary")
            st.info("💡 คำสรุปนี้ถูกสร้างขึ้นโดยใช้เนื้อหาต้นฉบับร่วมกับ Knowledge Graph เพื่อรักษาบริบทและป้องกันการบิดเบือนข้อมูล (Zero Hallucination)")
            st.write(st.session_state.final_summary)