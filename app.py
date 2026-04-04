import streamlit as st
import os
import spacy
import pandas as pd
import streamlit.components.v1 as components

from src.relation_extraction_ai import extract_relations
from src.entity_resolution import EntityResolver
from src.graph_builder import build_graph, visualize_graph
from src.graph_insight import analyze_graph

# 1. ตั้งค่าหน้าเว็บ (Page Config)
st.set_page_config(page_title="AI Document Graph", layout="wide", page_icon="🧠")

# ==========================================
# หน้าจอหลัก (Main Content)
# ==========================================
st.title("🧠 AI Knowledge Graph Extractor")
st.markdown("วิเคราะห์สกัดความสัมพันธ์จากข้อความและสร้างเป็น Mind Map แบบ Interactive")

# 2. โหลดโมเดล Entity Resolution
@st.cache_resource
def get_resolver():
    return EntityResolver(threshold=0.85)

resolver = get_resolver()

# ==========================================
# 3. ส่วนรับข้อมูล (Input Section) - เลือกได้ 2 โหมด
# ==========================================
input_method = st.radio(
    "👉 เลือกวิธีใส่ข้อมูล (Choose Input Method):",
    ("📝 วางข้อความ (Paste Text)", "📂 อัปโหลดไฟล์ (Upload .txt File)"),
    horizontal=True
)

document_text = ""
download_filename = "knowledge_graph.html"

if input_method == "📂 อัปโหลดไฟล์ (Upload .txt File)":
    uploaded_file = st.file_uploader("Upload your .txt file here", type=["txt"])
    if uploaded_file is not None:
        document_text = uploaded_file.getvalue().decode("utf-8")
        download_filename = f"{uploaded_file.name.replace('.txt', '')}_graph.html"
else:
    pasted_text = st.text_area("Paste your document text here", height=200, placeholder="พิมพ์หรือวางข้อความภาษาอังกฤษที่นี่...")
    if pasted_text.strip():
        document_text = pasted_text
        download_filename = "custom_text_graph.html"

# ==========================================
# 4. ส่วนวิเคราะห์และสร้างกราฟ
# ==========================================
if document_text:
    with st.expander("📄 ดูเนื้อหาต้นฉบับ (Original Document)"):
        st.write(document_text)

    # ปุ่มกดเริ่มวิเคราะห์
    if st.button("🚀 Analyze & Generate Graph", type="primary", use_container_width=True):
        
        # สร้าง Progress Bar ไว้ด้านบน
        progress_bar = st.progress(0, text="เตรียมการวิเคราะห์...")
        
        with st.status("AI is processing the document... Please wait ⏳", expanded=True) as status:
            
            # --- ขั้นตอนที่ 1 (0% -> 5%) ---
            st.write("✂️ 1. กำลังตัดคำและแบ่งประโยค...")
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(document_text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 10]
            
            progress_bar.progress(5, text="แบ่งประโยคเสร็จสิ้น (5%)")
            
            # --- ขั้นตอนที่ 2 (5% -> 80%) ---
            st.write("🤖 2. กำลังให้ LLM สกัดความสัมพันธ์ (Relation Extraction)...")
            relations = []
            total_sentences = len(sentences)
            
            if total_sentences > 0:
                for i, sent in enumerate(sentences):
                    # ทยอยส่งไปสกัดความสัมพันธ์ทีละ 1 ประโยค (เพื่อให้แถบโหลดขยับได้)
                    extracted = extract_relations([sent])
                    relations.extend(extracted)
                    
                    # คำนวณ % ปัจจุบัน (เริ่มที่ 5% และบวกเพิ่มสูงสุด 75%)
                    current_percent = 5 + int(75 * ((i + 1) / total_sentences))
                    progress_bar.progress(current_percent, text=f"กำลังสกัดความสัมพันธ์... ({i+1}/{total_sentences}) - {current_percent}%")
            else:
                progress_bar.progress(80, text="ข้ามการสกัดความสัมพันธ์ (80%)")

            usable_relations = [r for r in relations if r.get("quality", "medium") != "low"]
            
            # --- ขั้นตอนที่ 3 (80% -> 95%) ---
            st.write("🔍 3. กำลังยุบรวมคำที่ความหมายเหมือนกัน (Entity Resolution)...")
            resolved_relations = []
            seen_edges = set()
            total_usable = len(usable_relations)
            
            if total_usable > 0:
                for i, r in enumerate(usable_relations):
                    resolved_head = resolver.resolve(r["head"])
                    resolved_tail = resolver.resolve(r["tail"])
                    edge_key = (resolved_head.lower(), r["relation"].lower(), resolved_tail.lower())
                    
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        new_r = r.copy()
                        new_r["head"] = resolved_head
                        new_r["tail"] = resolved_tail
                        resolved_relations.append(new_r)
                    
                    # คำนวณ % ปัจจุบัน (เริ่มที่ 80% และบวกเพิ่มสูงสุด 15%)
                    current_percent = 80 + int(15 * ((i + 1) / total_usable))
                    progress_bar.progress(current_percent, text=f"กำลังคลีนข้อมูล... ({i+1}/{total_usable}) - {current_percent}%")
            else:
                progress_bar.progress(95, text="ข้ามการคลีนข้อมูล (95%)")

            # --- ขั้นตอนที่ 4 (95% -> 100%) ---
            st.write("🎨 4. กำลังวิเคราะห์ศูนย์กลางและสร้าง Knowledge Graph...")
            G = build_graph(resolved_relations)
            
            output_path = "output/web_graph.html"
            os.makedirs("output", exist_ok=True)
            visualize_graph(G, output_file=output_path)
            
            progress_bar.progress(100, text="✅ วิเคราะห์ข้อมูลเสร็จสิ้นสมบูรณ์ (100%)")
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # ==========================================
        # แสดงผลลัพธ์แบบแยก Tabs
        # ==========================================
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Knowledge Graph", "📈 Graph Insights", "📝 Extracted Data"])
        
        # 🟢 Tab 1: แสดงกราฟ และ ปุ่มดาวน์โหลด
        with tab1:
            st.subheader("Interactive Knowledge Graph")
            
            with open(output_path, "r", encoding="utf-8") as f:
                html_data = f.read()
            
            st.download_button(
                label="💾 Download Graph (HTML)",
                data=html_data,
                file_name=download_filename,
                mime="text/html",
                type="secondary"
            )
            
            components.html(html_data, height=600, scrolling=False)

        # 🟢 Tab 2: แสดงสถิติเชิงลึก
        with tab2:
            st.subheader("Graph Analysis & Insights")
            insights = analyze_graph(G)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Entities (Nodes)", insights.get("total_entities", 0))
            col2.metric("Total Relations (Edges)", insights.get("total_relations", 0))
            col3.metric("Graph Density", insights.get("density", 0))
            
            st.markdown("#### 🌟 Top Entities")
            if "top_entity_by_degree" in insights:
                st.info(f"**Main Entity (Hub):** {insights['top_entity_by_degree']['entity']} (Score: {insights['top_entity_by_degree']['score']})")
                
            with st.expander("ดูข้อมูลวิเคราะห์เชิงลึกแบบ JSON"):
                st.json(insights)

        # 🟢 Tab 3: แสดงตารางข้อมูลดิบ
        with tab3:
            st.subheader("Extracted Relations Table")
            if resolved_relations:
                df = pd.DataFrame(resolved_relations)
                if not df.empty and all(k in df.columns for k in ["head", "relation", "tail", "source_sentence"]):
                    df_display = df[["head", "relation", "tail", "source_sentence"]]
                    df_display.columns = ["Subject (Head)", "Relation", "Object (Tail)", "Source Sentence"]
                    st.dataframe(df_display, use_container_width=True)
            else:
                st.warning("No relations were extracted. ลองปรับข้อความให้เป็นประโยคที่สมบูรณ์ขึ้นครับ")