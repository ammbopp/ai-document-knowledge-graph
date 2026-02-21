import streamlit as st
import os
import spacy
import streamlit.components.v1 as components

# นำเข้าฟังก์ชันจากไฟล์ในโปรเจกต์ของเรา
from src.relation_extraction_ai import extract_relations
from src.entity_resolution import EntityResolver
from src.graph_builder import build_graph, visualize_graph
from src.graph_insight import analyze_graph

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Document Graph", layout="wide", page_icon="🧠")
st.title("🧠 AI Knowledge Graph Extractor")
st.markdown("อัปโหลดไฟล์เอกสาร (Text) เพื่อให้ AI วิเคราะห์และสร้าง Mind Map ความสัมพันธ์")

# 2. โหลดโมเดล Entity Resolution (ใช้ Cache จะได้ไม่โหลดซ้ำเวลากดปุ่ม)
@st.cache_resource
def get_resolver():
    return EntityResolver(threshold=0.85)

resolver = get_resolver()

# 3. ส่วนรับไฟล์อัปโหลด
uploaded_file = st.file_uploader("Upload your .txt file here", type=["txt"])

if uploaded_file is not None:
    # อ่านข้อความจากไฟล์
    document_text = uploaded_file.getvalue().decode("utf-8")
    
    with st.expander("📄 ดูเนื้อหาไฟล์ต้นฉบับ"):
        st.text_area("Document Content", document_text, height=200)

    # ปุ่มกดเริ่มวิเคราะห์
    if st.button("🚀 Analyze & Generate Graph", type="primary"):
        
        # ใช้ st.status เพื่อโชว์สถานะการทำงานทีละสเต็ปให้ดูเท่ๆ
        with st.status("AI is processing the document...", expanded=True) as status:
            
            st.write("✂️ กำลังตัดคำและแบ่งประโยค...")
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(document_text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 10]
            
            st.write("🤖 กำลังให้ Flan-T5 สกัดความสัมพันธ์ (Relation Extraction)...")
            relations = extract_relations(sentences)
            usable_relations = [r for r in relations if r.get("quality", "medium") != "low"]
            
            st.write("🔍 กำลังยุบรวมคำที่ความหมายเหมือนกัน (Entity Resolution)...")
            resolved_relations = []
            seen_edges = set()
            for r in usable_relations:
                resolved_head = resolver.resolve(r["head"])
                resolved_tail = resolver.resolve(r["tail"])
                edge_key = (resolved_head.lower(), r["relation"].lower(), resolved_tail.lower())
                
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    new_r = r.copy()
                    new_r["head"] = resolved_head
                    new_r["tail"] = resolved_tail
                    resolved_relations.append(new_r)
                    
            st.write("🎨 กำลังสร้าง Knowledge Graph...")
            topic_name = uploaded_file.name.replace(".txt", "").replace("_", " ").title()
            G = build_graph(resolved_relations, root_name=topic_name)
            
            output_path = "output/web_graph.html"
            os.makedirs("output", exist_ok=True)
            visualize_graph(G, output_file=output_path)
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # 4. แสดงผลกราฟบนหน้าเว็บ
        st.subheader("📊 Interactive Knowledge Graph")
        with open(output_path, "r", encoding="utf-8") as f:
            html_data = f.read()
            
        # ใช้ iframe ในการ render HTML ของ PyVis
        components.html(html_data, height=825, scrolling=False)
        
        # 5. แสดง Insights สถิติของกราฟ
        st.subheader("📈 Graph Insights")
        insights = analyze_graph(G)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Entities", insights.get("total_entities", 0))
        col2.metric("Total Relations", insights.get("total_relations", 0))
        col3.metric("Density", insights.get("density", 0))
        
        with st.expander("ดูข้อมูลวิเคราะห์เชิงลึก (Deep Insights)"):
            st.json(insights)