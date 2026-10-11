import os
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import hashlib
import inspect
import json
import pathlib
import re
import traceback

import pandas as pd
import spacy
import streamlit as st
import streamlit.components.v1 as components

from src.config import MIN_CONFIDENCE, RESOLVER_THRESHOLD, WINDOW_SIZE
from src.entity_resolution import load_embedder
from src.graph_builder import NODE_COLORS
from src.llm import health
from src.pipeline import analyze_document, estimate_seconds
from src.themes import DEFAULT_PALETTE, PALETTES, accent, css_vars
from src.web_fetch import decode_upload, fetch_text_from_url

st.set_page_config(page_title="Knowledge Graph", layout="wide", page_icon=":material/hub:",
                   initial_sidebar_state="expanded")

# Streamlit renamed use_container_width -> width="stretch" and deprecated components.html -> st.iframe
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})


def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


PASTE, UPLOAD, URL = "วางข้อความ", "อัปโหลดไฟล์ .txt", "ลิงก์ข่าว"
NODE_LABELS = {
    "Person": "บุคคล", "Organization": "องค์กร", "Location": "สถานที่", "Time": "เวลา",
    "Work": "ผลงาน", "Entity": "อื่นๆ", "Topic": "หัวข้อหลัก (ศูนย์กลาง)",
}
SAMPLE_DIR = pathlib.Path("data")

for key, default in {
    "input_method": PASTE, "result": None, "result_hash": "", "result_name": "document",
    "url_text": "", "url_text_for": "", "url_error": "", "sample_name": "", "sample_text": "",
    "palette": DEFAULT_PALETTE,
}.items():
    st.session_state.setdefault(key, default)


def is_dark_theme():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark_theme()
_css = pathlib.Path("assets/style.css")
if _css.exists():
    st.markdown(f"<style>{css_vars(st.session_state.palette, DARK)}\n{_css.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ==========================================
# Cached resources
# ==========================================
@st.cache_resource(show_spinner="กำลังโหลดโมเดลภาษา (ครั้งแรกอาจใช้เวลาสักครู่)...")
def get_embedder():
    return load_embedder()


@st.cache_resource
def get_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_data(ttl=15, show_spinner=False)
def check_ollama():
    return health()


def list_samples():
    samples = []
    for path in sorted(SAMPLE_DIR.glob("sample*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            samples.append((path.stem, text))
    return samples[:4]


def load_sample(name, text):
    st.session_state.input_method = PASTE
    st.session_state.paste_box = text
    st.session_state.sample_name, st.session_state.sample_text = name, text


def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def fmt_duration(seconds, short=False):
    seconds = int(seconds)
    if short:  # compact form for metric values: 2:16 นาที
        return f"{seconds // 60}:{seconds % 60:02d} นาที" if seconds >= 60 else f"{seconds} วินาที"
    return f"{seconds // 60} นาที {seconds % 60} วินาที" if seconds >= 60 else f"{seconds} วินาที"


# ==========================================
# Sidebar
# ==========================================
ollama_ok, ollama_msg = check_ollama()
with st.sidebar:
    st.markdown("<div class='brand'>Knowledge Graph<small>document → relations</small></div>",
                unsafe_allow_html=True)

    st.markdown("<div class='label'>สถานะ</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status'><span class='dot {'ok' if ollama_ok else 'bad'}'></span>"
                f"{ollama_msg.split(' — ')[0]}</div>", unsafe_allow_html=True)
    if not ollama_ok:
        st.caption(ollama_msg.split(" — ")[-1])
        if st.button("ตรวจสอบอีกครั้ง", icon=":material/refresh:", **STRETCH):
            check_ollama.clear()
            st.rerun()

    st.markdown("<div class='label'>ธีมสี</div>", unsafe_allow_html=True)
    st.selectbox("ธีมสี", list(PALETTES), key="palette", label_visibility="collapsed")
    st.markdown("<div class='swatches'>" + "".join(
        f"<span class='sw{' on' if name == st.session_state.palette else ''}' title='{name}' "
        f"style='background:{accent(name, DARK)}'></span>" for name in PALETTES) + "</div>",
        unsafe_allow_html=True)
    st.caption("โหมดสว่าง/มืด: เมนู ⋮ มุมขวาบน › Settings")

    st.markdown("<div class='label'>การวิเคราะห์</div>", unsafe_allow_html=True)
    with st.expander("ตั้งค่าขั้นสูง"):
        min_conf = st.slider("ความมั่นใจขั้นต่ำของความสัมพันธ์", 0.6, 1.0, float(MIN_CONFIDENCE), 0.1,
                             help="สูง = แม่นยำขึ้นแต่ได้ความสัมพันธ์น้อยลง")
        threshold = st.slider("ความคล้ายเพื่อรวมเอนทิตี", 0.70, 0.99, float(RESOLVER_THRESHOLD), 0.01,
                              help="ต่ำ = รวมชื่อที่คล้ายกันมากขึ้น (เสี่ยงรวมผิด)")
        window_size = st.slider("จำนวนประโยคต่อกลุ่ม", 1, 5, WINDOW_SIZE,
                                help="กลุ่มใหญ่ = บริบทมากขึ้นแต่ LLM ทำงานช้าลง")
        max_chars = st.slider("วิเคราะห์ไม่เกิน (ตัวอักษร)", 1000, 20000, 6000, 500,
                              help="เอกสารที่ยาวกว่านี้จะถูกตัดที่ท้ายประโยค เพื่อไม่ให้รอนานเกินไป")

    if st.session_state.result is not None and st.button("ล้างผลลัพธ์", icon=":material/delete:", **STRETCH):
        st.session_state.result = None
        st.rerun()

    st.divider()
    st.caption("รองรับเอกสารภาษาอังกฤษเท่านั้น ประมวลผลบนเครื่องด้วย Ollama "
               "ข้อมูลไม่ถูกส่งออกไปภายนอก ยกเว้นเมื่อดึงเนื้อหาจากลิงก์")


# ==========================================
# Header + input
# ==========================================
st.markdown(
    """<div class="masthead"><div class="kicker">DOCUMENT KNOWLEDGE GRAPH</div>
    <h1>จากบทความ สู่แผนผังความสัมพันธ์</h1>
    <p>วางข้อความหรือลิงก์ข่าวภาษาอังกฤษ ระบบจะดึงความสัมพันธ์ระหว่างบุคคล องค์กร และเหตุการณ์
    ออกมาเป็นกราฟที่สำรวจได้ พร้อมตารางหลักฐานและบทสรุป</p></div>""", unsafe_allow_html=True)

method = st.segmented_control("วิธีใส่ข้อมูล", [PASTE, UPLOAD, URL], key="input_method",
                              label_visibility="collapsed") or PASTE

main_text, doc_name = "", "document"
if method == PASTE:
    main_text = st.text_area("ข้อความ", key="paste_box", height=220, label_visibility="collapsed",
                             placeholder="วางบทความหรือข้อความภาษาอังกฤษที่นี่ (แนะนำ 3-30 ประโยค)").strip()
    doc_name = st.session_state.sample_name if main_text == st.session_state.sample_text else "custom_text"
elif method == UPLOAD:
    uploaded = st.file_uploader("ไฟล์ .txt", type=["txt"], label_visibility="collapsed")
    if uploaded is not None:
        main_text = decode_upload(uploaded.getvalue()).strip()
        doc_name = pathlib.Path(uploaded.name).stem
else:
    col_url, col_btn = st.columns([6, 1], vertical_alignment="bottom")
    url = col_url.text_input("ลิงก์บทความ", placeholder="https://www.bbc.com/news/...",
                             label_visibility="collapsed").strip()
    retry = col_btn.button("ดึงเนื้อหา", **STRETCH)
    if url and (url != st.session_state.url_text_for or retry):
        with st.spinner("กำลังดึงและคัดกรองเนื้อหาจากลิงก์..."):
            try:
                st.session_state.url_text = fetch_text_from_url(url).strip()
                st.session_state.url_error = "" if st.session_state.url_text else "ไม่พบเนื้อหาข้อความในหน้านี้"
            except Exception as e:
                st.session_state.url_text, st.session_state.url_error = "", str(e)
            st.session_state.url_text_for = url
    if url and url == st.session_state.url_text_for:
        if st.session_state.url_error:
            st.error(f"ดึงเนื้อหาไม่สำเร็จ: {st.session_state.url_error}")
        else:
            main_text = st.session_state.url_text
            doc_name = "news_url"

# ==========================================
# Empty state
# ==========================================
if not main_text:
    st.markdown(
        """<div class="steps">
          <div class="step"><span class="n">01</span><b>ใส่เอกสาร</b>
            <span class="d">วางข้อความ อัปโหลดไฟล์ .txt หรือใส่ลิงก์ข่าวภาษาอังกฤษ</span></div>
          <div class="step"><span class="n">02</span><b>วิเคราะห์</b>
            <span class="d">สกัดความสัมพันธ์ ตรวจทิศทางของแต่ละคู่ และรวมชื่อเอนทิตีที่ซ้ำกัน</span></div>
          <div class="step"><span class="n">03</span><b>สำรวจผล</b>
            <span class="d">ดูกราฟ ตารางพร้อมประโยคต้นฉบับ ข้อมูลเชิงลึก และบทสรุป</span></div>
        </div>""", unsafe_allow_html=True)
    samples = list_samples()
    if samples:
        st.markdown("<div class='label'>เอกสารตัวอย่าง</div>", unsafe_allow_html=True)
        for name, text in samples:
            c_name, c_text, c_btn = st.columns([2, 7, 1.3], vertical_alignment="center")
            c_name.markdown(f"<div class='sample-name'>{name}</div>", unsafe_allow_html=True)
            c_text.markdown(f"<div class='sample-text'>{text[:140].replace('<', '&lt;')}</div>",
                            unsafe_allow_html=True)
            c_btn.button("ใช้", key=f"sample_{name}", on_click=load_sample, args=(name, text), **STRETCH)
            st.divider()
    st.stop()

# ==========================================
# Document summary + analyse
# ==========================================
words, chars = len(main_text.split()), len(main_text)
cur_hash = text_hash(main_text)
will_truncate = chars > max_chars

c1, c2, c3 = st.columns(3)
c1.metric("จำนวนคำ", f"{words:,}")
c2.metric("จำนวนตัวอักษร", f"{chars:,}")
c3.metric("เวลาที่ใช้โดยประมาณ",
          fmt_duration(estimate_seconds(main_text[:max_chars], window_size), short=True),
          help="ประมาณจากความยาวเอกสาร เวลาจริงขึ้นกับเครื่องและโมเดล")

if len(re.findall(r"[฀-๿]", main_text)) > 0.3 * max(chars, 1):
    st.warning("เอกสารส่วนใหญ่เป็นภาษาไทย — ระบบรองรับเฉพาะภาษาอังกฤษ ผลลัพธ์อาจไม่ถูกต้อง")
if will_truncate:
    st.info(f"เอกสารยาว {chars:,} ตัวอักษร จะวิเคราะห์เฉพาะ {max_chars:,} ตัวอักษรแรก "
            "(ปรับได้ที่แถบด้านข้าง › ตั้งค่าขั้นสูง)")

with st.expander("ดูเนื้อหาที่จะวิเคราะห์"):
    st.text(main_text[:20000] + ("\n…" if chars > 20000 else ""))

run = st.button("วิเคราะห์เอกสาร", type="primary", icon=":material/play_arrow:", **STRETCH,
                disabled=not ollama_ok, help=None if ollama_ok else "ต้องเปิด Ollama ก่อน (ดูแถบด้านข้าง)")

if run:
    progress = st.progress(0, text="เตรียมการวิเคราะห์...")
    with st.status("กำลังวิเคราะห์เอกสาร...", expanded=True) as status:
        def on_progress(pct, msg, done):
            progress.progress(min(pct, 100), text=f"{msg} ({pct}%)")
            if done:
                st.write(f"{pct:>3}%  {msg}")
            else:
                status.update(label=msg)
        try:
            result = analyze_document(main_text, get_nlp(), get_embedder(), min_confidence=min_conf,
                                      threshold=threshold, window_size=window_size, max_chars=max_chars,
                                      on_progress=on_progress)
        except Exception as e:
            status.update(label="เกิดข้อผิดพลาด", state="error", expanded=False)
            failure = (e, traceback.format_exc())
        else:
            failure = None
            status.update(label=f"วิเคราะห์เสร็จสิ้น · ใช้เวลา {fmt_duration(result.stats['elapsed'])}",
                          state="complete", expanded=False)
    if failure:  # shown outside the (collapsed) status box so it is always visible
        progress.empty()
        st.error(f"วิเคราะห์ไม่สำเร็จ: {failure[0]}")
        with st.expander("รายละเอียดข้อผิดพลาด"):
            st.code(failure[1])
        st.stop()
    progress.empty()
    st.session_state.result, st.session_state.result_hash = result, cur_hash
    st.session_state.result_name = re.sub(r"[^\w\-]+", "_", doc_name) or "document"
    st.rerun()

# ==========================================
# Results
# ==========================================
result = st.session_state.result
if result is None:
    st.stop()
if st.session_state.result_hash != cur_hash:
    st.info("ผลลัพธ์ที่เก็บไว้เป็นของเอกสารอื่น — กด “วิเคราะห์เอกสาร” เพื่อวิเคราะห์เอกสารนี้")
    st.stop()

insights, relations, stats = result.insights, result.relations, result.stats
st.markdown(f"<div class='result-head'><h2>ผลการวิเคราะห์</h2><code>{st.session_state.result_name}</code></div>",
            unsafe_allow_html=True)

if not relations:
    st.warning("ไม่พบความสัมพันธ์ที่ผ่านการตรวจสอบ ลอง: ใช้ประโยคที่สมบูรณ์ขึ้น, ลดความมั่นใจขั้นต่ำ "
               "หรือเพิ่มจำนวนประโยคต่อกลุ่มในแถบด้านข้าง")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("เอนทิตี (โหนด)", insights.get("total_entities", 0))
k2.metric("ความสัมพันธ์ (เส้น)", insights.get("total_relations", 0))
k3.metric("กลุ่มที่เชื่อมกันเอง", insights.get("num_connected_components", 0),
          help="จำนวนกลุ่มเนื้อหาที่ไม่เชื่อมถึงกันในข้อความจริง (ไม่นับเส้นที่ระบบเชื่อมให้)")
k4.metric("เวลาที่ใช้", fmt_duration(stats["elapsed"], short=True))
if stats["truncated"]:
    st.caption(f"วิเคราะห์เฉพาะ {stats['chars']:,} ตัวอักษรแรกของเอกสาร")

st.write("")
tab_graph, tab_table, tab_insight, tab_summary = st.tabs(["กราฟ", "ความสัมพันธ์", "ข้อมูลเชิงลึก", "สรุป"])

with tab_graph:
    used = {"Topic"} | {r.get(k, "Entity") for r in relations for k in ("head_type", "tail_type")}
    legend = "".join(
        f"<span><i style='background:{NODE_COLORS[t]['background']}'></i>{NODE_LABELS[t]}</span>"
        for t in NODE_LABELS if t in used)
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)
    st.caption("ลากเพื่อย้ายโหนด · เลื่อนเมาส์เพื่อซูม · คลิกโหนดเพื่อดูรายละเอียด · "
               "เส้นประสีแดงคือเส้นที่ระบบเชื่อมกลุ่มที่แยกกัน ไม่ได้มาจากข้อความ")
    show_html(result.html, 680)
    st.download_button("ดาวน์โหลดกราฟ (HTML)", result.html, icon=":material/download:",
                       file_name=f"{st.session_state.result_name}_graph.html", mime="text/html")

with tab_table:
    df = pd.DataFrame([{
        "ประธาน": r["head"], "ความสัมพันธ์": r["relation"], "กรรม": r["tail"],
        "ชนิดประธาน": NODE_LABELS.get(r.get("head_type", "Entity"), "อื่นๆ"),
        "ชนิดกรรม": NODE_LABELS.get(r.get("tail_type", "Entity"), "อื่นๆ"),
        "ความมั่นใจ": r.get("confidence", 0.0), "ประโยคต้นฉบับ": r.get("source_sentence", ""),
    } for r in relations])

    query = st.text_input("ค้นหา", placeholder="ค้นหาเอนทิตี ความสัมพันธ์ หรือประโยค",
                          label_visibility="collapsed").strip().lower()
    if query:
        df = df[df.apply(lambda row: query in " ".join(map(str, row.values)).lower(), axis=1)]
    st.caption(f"แสดง {len(df)} จาก {len(relations)} รายการ · คลิกแถวเพื่อดูประโยคต้นฉบับ")

    event = st.dataframe(
        df, **STRETCH, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_order=["ประธาน", "ความสัมพันธ์", "กรรม", "ความมั่นใจ", "ประโยคต้นฉบับ"],
        column_config={
            "ความมั่นใจ": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.1f"),
            "ประโยคต้นฉบับ": st.column_config.TextColumn(width="large"),
        })
    if event.selection.rows and event.selection.rows[0] < len(df):
        row = df.iloc[event.selection.rows[0]]
        with st.container(border=True):
            st.markdown(f"**{row['ประธาน']}** → *{row['ความสัมพันธ์']}* → **{row['กรรม']}**")
            st.caption(f"“{row['ประโยคต้นฉบับ']}”")

    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("CSV", df.to_csv(index=False).encode("utf-8-sig"), icon=":material/download:",
                       file_name=f"{st.session_state.result_name}_relations.csv", mime="text/csv",
                       **STRETCH)
    d2.download_button("JSON", json.dumps(relations, ensure_ascii=False, indent=2), icon=":material/download:",
                       file_name=f"{st.session_state.result_name}_relations.json", mime="application/json",
                       **STRETCH)

with tab_insight:
    i1, i2, i3 = st.columns(3)
    i1.metric("ความหนาแน่นของกราฟ", insights.get("density", 0),
              help="สัดส่วนเส้นที่มีจริงเทียบกับเส้นที่เป็นไปได้ทั้งหมด ยิ่งต่ำยิ่งกระจาย")
    i2.metric("จำนวนกลุ่มเนื้อหา (community)", insights.get("num_communities", 0))
    i3.metric("กลุ่มที่เชื่อมกันเอง", insights.get("num_connected_components", 0))

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='label'>เอนทิตีศูนย์กลาง</div>", unsafe_allow_html=True)
        top = insights.get("top_entity_by_degree")
        if top:
            st.markdown(f"### {top['entity']}")
            st.caption("เชื่อมกับเอนทิตีอื่นมากที่สุด")
    with right:
        st.markdown("<div class='label'>เอนทิตีตัวเชื่อม</div>", unsafe_allow_html=True)
        bridge = insights.get("top_entity_by_betweenness")
        if bridge and bridge["score"] > 0:
            st.markdown(f"### {bridge['entity']}")
            st.caption("เป็นทางผ่านระหว่างกลุ่มเนื้อหาต่างๆ")
        else:
            st.caption("ไม่มีเอนทิตีที่ทำหน้าที่เชื่อมระหว่างกลุ่ม")

    top_rel = insights.get("top_relations", [])
    if top_rel:
        st.markdown("<div class='label'>ความสัมพันธ์ที่พบบ่อย</div>", unsafe_allow_html=True)
        rel_df = pd.DataFrame([(name.replace("_", " "), n) for name, n in top_rel],
                              columns=["ความสัมพันธ์", "จำนวน"]).set_index("ความสัมพันธ์")
        st.bar_chart(rel_df, horizontal=True, height=40 * len(rel_df) + 60,
                     color=accent(st.session_state.palette, DARK))

    communities = insights.get("sample_communities", [])
    if communities:
        st.markdown("<div class='label'>กลุ่มเนื้อหาที่พบ</div>", unsafe_allow_html=True)
        for i, members in enumerate(communities, 1):
            st.markdown(f"**กลุ่ม {i}** · " + " · ".join(members))
    with st.expander("ข้อมูลดิบ (JSON)"):
        st.json(insights)

with tab_summary:
    # LLMs often emit "• a • b" on one line; turn inline bullets into a real Markdown list
    summary_md = re.sub(r"\s*•\s*", "\n\n- ", result.summary) if result.summary else ""
    st.markdown(summary_md or "ไม่สามารถสร้างคำสรุปได้")
    st.caption("สรุปโดย LLM จากเนื้อหาต้นฉบับร่วมกับกราฟความรู้ เพื่อลดการบิดเบือน แต่ไม่รับประกันว่าถูกต้อง 100% "
               "ควรตรวจสอบกับต้นฉบับ")
    if result.summary:
        st.download_button("ดาวน์โหลดสรุป (.md)", result.summary, icon=":material/download:",
                           file_name=f"{st.session_state.result_name}_summary.md", mime="text/markdown")
