from .config import SUMMARY_GRAPH_CHARS, SUMMARY_TEXT_CHARS
from .llm import generate

PROMPT = """You are an expert AI summarizer. Your task is to generate a highly accurate, structured summary.

To prevent hallucinations and capture topic shifts perfectly, you MUST synthesize the summary using BOTH the 'Original Text' and the extracted 'Knowledge Graph Context'.

=== ORIGINAL TEXT ===
{text}

=== KNOWLEDGE GRAPH CONTEXT (Chronological Topic Shifts) ===
{graph}

=== INSTRUCTIONS ===
1. Write a clear, executive-level summary of the text.
2. Use the Knowledge Graph Context to ensure you capture the exact relationships and track how the topic shifts from beginning to end.
3. DO NOT invent, hallucinate, or add external knowledge. Ground every fact in the provided inputs.
4. Format the output with clear bullet points or short paragraphs for readability.

Summary:"""


def generate_graph_augmented_summary(original_text, resolved_relations):
    grouped = {}
    for r in resolved_relations:
        grouped.setdefault(r.get("source_sentence", "General"), []).append(
            f"{r['head']} -> {r['relation']} -> {r['tail']}")

    graph_context = ""
    for idx, triplets in enumerate(grouped.values()):
        graph_context += f"\n[Topic Shift {idx + 1}]\n" + "".join(f"- {t}\n" for t in triplets)

    prompt = PROMPT.format(text=original_text[:SUMMARY_TEXT_CHARS], graph=graph_context[:SUMMARY_GRAPH_CHARS])
    try:
        return generate(prompt, temperature=0.1)
    except Exception as e:
        print(f"⚠️ Graph-Augmented Summarization Failed: {e}")
        return "ไม่สามารถสร้างคำสรุปได้เนื่องจากเกิดข้อผิดพลาดในการเชื่อมต่อกับ LLM"
