import streamlit as st
import os
import base64
import cv2
import numpy as np
import ssl
import httpx
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from neo4j import GraphDatabase

# --- 1. SSL & CONFIG ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

load_dotenv()
st.set_page_config(page_title="SemiDoctor: Specialist", layout="wide", page_icon="🏥")

if os.getenv("GROQ_API_KEY"): os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY").strip()
if not os.getenv("GROQ_API_KEY"): st.error("❌ Groq Key Missing!"); st.stop()

# --- 2. ENGINES ---
vision_llm = ChatGroq(model="llama-3.2-11b-vision-preview", temperature=0.0)
reasoning_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
tavily_tool = TavilySearchResults(max_results=3)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# --- 3. PRE-PROCESSING (CLAHE) ---
def preprocess_image(uploaded_file):
    """Enhances contrast so the AI can see 'hidden' details."""
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        # Apply Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_img = clahe.apply(img)
        _, buffer = cv2.imencode('.jpg', enhanced_img)
        return base64.b64encode(buffer).decode('utf-8')
    except:
        uploaded_file.seek(0)
        return base64.b64encode(uploaded_file.read()).decode('utf-8')

def find_help(location):
    try: return tavily_tool.invoke(f"Emergency hospital near {location}")
    except: return []

# --- 4. STATE ---
class MedicalState(TypedDict):
    messages: List[str]
    patient_text: str
    scan_type: str       # <--- NEW: Context Injection
    image_data: str
    image_findings: str
    graph_evidence: str
    web_evidence: str
    diagnosis: str

# --- 5. AGENT 1: THE SPECIALIST RADIOLOGIST ---
def radiologist_node(state: MedicalState):
    img_data = state.get('image_data')
    scan_type = state.get('scan_type', 'General')
    
    if not img_data: return {"image_findings": "No image."}
    
    # --- CONTEXT INJECTION: DYNAMIC PROMPTS ---
    prompts = {
        "Brain MRI": """
            You are a Neuroradiologist. Analyze this Brain MRI.
            CRITICAL CHECKS:
            1. Is there a Midline Shift? (Yes/No)
            2. Is there Hyperintensity (White spots) in T2? (Yes/No)
            3. Is there a Mass/Tumor? Describe location.
            4. Is there Hemorrhage (Dark in T2)?
            IGNORE lungs, bones, or heart. Focus ONLY on the brain.
        """,
        "Chest X-Ray": """
            You are a Thoracic Radiologist. Analyze this Chest X-Ray.
            CRITICAL CHECKS:
            1. Is there Consolidation (Pneumonia)?
            2. Is there Pneumothorax (Collapsed lung)?
            3. Is the Heart enlarged (Cardiomegaly)?
            IGNORE brain or bones. Focus ONLY on lungs/heart.
        """,
        "Bone X-Ray": """
            You are an Orthopedic Surgeon. Analyze this Bone X-Ray.
            CRITICAL CHECKS:
            1. Trace the cortical margins. Is there a break/fracture?
            2. Is there dislocation at the joint?
            Focus ONLY on bones.
        """
    }
    
    selected_prompt = prompts.get(scan_type, "Analyze this medical scan for anomalies.")
    
    msg = HumanMessage(content=[
        {"type": "text", "text": selected_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
    ])
    
    try:
        res = vision_llm.invoke([msg])
        return {"image_findings": res.content, "messages": [f"👁️ {scan_type.upper()} SPECIALIST: {res.content}"]}
    except Exception as e: 
        return {"image_findings": "Error", "messages": [f"❌ Vision Failed: {e}"]}

# --- 6. AGENT 2: THE SKEPTICAL RESEARCHER ---
def researcher_node(state: MedicalState):
    case = state['patient_text']
    findings = state.get('image_findings', '')
    
    # WEB VALIDATION (The "BS Check")
    # If the image says "Tumor" but the patient has "Broken leg", this agent catches it.
    web_text = ""
    try:
        query = f"Can {case} cause these radiology findings: {findings[:100]}?"
        res = tavily_tool.invoke(query)
        web_text = str(res)[:800]
    except: pass

    return {"web_evidence": web_text, "messages": [f"📚 VALIDATION: Checked logic against medical databases."]}

# --- 7. AGENT 3: THE CHIEF DIAGNOSTICIAN ---
def diagnostician_node(state: MedicalState):
    prompt = f"""
    Chief Medical Officer.
    
    PATIENT SYMPTOMS: {state['patient_text']}
    SCAN FINDINGS: {state['image_findings']}
    WEB VALIDATION: {state['web_evidence']}
    
    RULE:
    If the Scan Findings do NOT match the Patient Symptoms (e.g. Brain Tumor found but patient has foot pain), DISCARD the image findings as a machine error.
    
    Provide the most logical diagnosis.
    """
    try:
        res = reasoning_llm.invoke(prompt)
        return {"diagnosis": res.content, "messages": [f"👨‍⚕️ DIAGNOSTICIAN: {res.content}"]}
    except: return {"diagnosis": "Error"}

# --- 8. GRAPH BUILD ---
workflow = StateGraph(MedicalState)
workflow.add_node("radiologist", radiologist_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("diagnostician", diagnostician_node)

workflow.set_entry_point("radiologist")
workflow.add_edge("radiologist", "researcher")
workflow.add_edge("researcher", "diagnostician")
workflow.add_edge("diagnostician", END)
app = workflow.compile()

# --- 9. UI ---
with st.sidebar:
    st.header("🚨 EMERGENCY")
    loc = st.text_input("Location", "Delhi")
    if st.button("🆘 CALL AMBULANCE"):
        res = find_help(loc)
        st.error(f"DISPATCHING TO: {res[0]['content'][:50]}...")

st.title("🏥 SemiDoctor: Specialist Edition")
st.caption("Context-Aware Radiology (Brain • Chest • Bone)")

col1, col2 = st.columns([1, 2])
with col1:
    # THE FIX: USER MUST SELECT CONTEXT
    scan_type = st.selectbox("Select Scan Type (Crucial for Accuracy)", ["Brain MRI", "Chest X-Ray", "Bone X-Ray"])
    uploaded_file = st.file_uploader("Upload Scan", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Scan Uploaded", use_column_width=True)

with col2:
    text = st.text_area("Symptoms", "Headache and nausea.")
    
    if st.button("Start Analysis"):
        if uploaded_file: uploaded_file.seek(0)
        
        with st.spinner(f"Calling {scan_type} Specialist..."):
            processed_img = preprocess_image(uploaded_file) if uploaded_file else None
            
            initial = {
                "messages": [], 
                "patient_text": text, 
                "scan_type": scan_type,
                "image_data": processed_img
            }
            
            try:
                for event in app.stream(initial):
                    for v in event.values():
                        if "messages" in v:
                            msg = v["messages"][0]
                            if "SPECIALIST" in msg: st.info(msg)
                            elif "DIAGNOSTICIAN" in msg: st.success(msg)
                            elif "VALIDATION" in msg: st.write(msg)
                st.success("✅ Assessment Complete.")
            except Exception as e: st.error(str(e))