#  SemiDoctor: Multi-Agent Clinical Decision Support (Prototype)

**Status:** Halted. Because it couldn't read image properly. Maybe needs more planning and introspection.

##  Project Overview
SemiDoctor is a **Multi-Modal AI System** designed to assist medical professionals in triage and diagnosis. It utilizes a swarm of AI Agents to analyze patient history, medical imaging (X-Ray/MRI), and medical knowledge graphs simultaneously.

##  Tech Stack (The "200k" Architecture)
* **Groq LPU:** Real-time inference using `Llama-3.3-70b` (Reasoning) and `Llama-3.2-11b` (Vision).
* **Neo4j GraphRAG:** Structured knowledge graph for "hallucination-free" medical fact-checking.
* **OpenCV (Computer Vision):** Implemented CLAHE (Contrast Limited Adaptive Histogram Equalization) for MRI enhancement.
* **LangGraph:** State-based multi-agent orchestration (Radiologist -> Researcher -> Diagnostician -> Critic).
* **Tavily AI:** Real-time web search for hospital geolocation and emergency dispatch simulation.

##  Safety & Compliance
This project is a **technical proof-of-concept**. Development was halted to prioritize the design of HIPAA-compliant data pipelines and liability guardrails. It is **not** for clinical use.

##  Key Features
1.  **Context-Aware Vision:** Specialized prompts for Brain MRI vs. Chest X-Ray.
2.  **"One-Touch" Rescue:** Geolocation-based emergency service dispatcher.
3.  **Hybrid RAG:** Combines deterministic Graph DBs with live web search.
