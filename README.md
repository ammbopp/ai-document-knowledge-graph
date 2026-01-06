# AI Document Knowledge Graph

This project extracts open-ended relationships from documents using Large Language Models (LLMs) and constructs a knowledge graph for structural analysis and visualization.

## Features
- Open Information Extraction (no fixed schema)
- Multi-triple relation parsing
- Boundary-aware post-processing
- Quality-based relation filtering
- Knowledge graph construction and analysis
- Supports English documents (Thai support in progress)

## Pipeline
Text → Sentence Decomposition → LLM-based Relation Extraction → 
Post-processing → Knowledge Graph → Graph Insights

## Tech Stack
- Python
- HuggingFace Transformers
- NetworkX
- PyVis

## Status
- Core extraction & graph pipeline: ✅ Completed
- OCR integration (web-based): 🔜 Planned
