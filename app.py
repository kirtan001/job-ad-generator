import streamlit as st
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- CONFIGURATION ---
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "jobs.index")
META_FILE = os.path.join(INDEX_DIR, "jobs_meta.json")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# CORRECT MODEL NAME FOR GROQ (Llama 3 70B)
GROQ_MODEL = "openai/gpt-oss-20b" 

TOP_K = 4
N_VARIANTS = 3
MAX_CONTEXT_CHARS = 1200

# --- PAGE SETUP ---
st.set_page_config(page_title="Job Ad Generator", layout="wide")
st.title("🤖 Job Ad Generato")

# --- 1. LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    with st.spinner("Loading AI Models..."):
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None, None, None

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    return embed_model, index, records

embed_model, index, records = load_resources()

if index is None:
    st.error(f"⚠️ Could not find FAISS files! Check '{INDEX_DIR}' folder.")
    st.stop()


# --- 2. SETUP GROQ CLIENT ---
api_key = None

# Try loading from local config.py
try:
    import config
    if hasattr(config, 'GROQ_API_KEY') and config.GROQ_API_KEY:
        api_key = config.GROQ_API_KEY
except ImportError:
    pass

# If not local, check Secrets (Hugging Face / Streamlit)
if not api_key:
    api_key = os.getenv("GROQ_API_KEY") 
    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.error("⚠️ Groq API Key not found! Set 'GROQ_API_KEY' in config.py or Secrets.")
    st.stop()

client = Groq(api_key=api_key)


# --- 3. HELPER FUNCTIONS ---
def retrieve(query, index, records, embed_model, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(records):
            continue
        rec = records[idx].copy()
        rec["_score"] = float(score)
        if not rec.get("summary"):
            txt = rec.get("combined_text","")
            rec["summary"] = (txt[:400].rsplit(" ",1)[0] + " ...") if len(txt) > 400 else txt
        hits.append(rec)
    return hits

def build_prompt(user_input, retrieved, n_variants):
    # Constructing the message for Llama 3
    context_block = ""
    total_chars = 0
    for i, r in enumerate(retrieved, 1):
        summary = r.get("summary","")
        allowed = summary if (total_chars + len(summary)) <= MAX_CONTEXT_CHARS else summary[:max(50, MAX_CONTEXT_CHARS-total_chars)]
        context_block += f"{i}) {r.get('Company Name','')} | {r.get('Location','')} | {allowed}\n"
        total_chars += len(allowed)
        if total_chars >= MAX_CONTEXT_CHARS: break

    system_msg = (
        "You are an expert HR copywriter. Your task is to generate job ads based on context. "
        "You MUST return ONLY valid JSON. No preamble, no markdown formatting (like ```json), just the raw JSON string."
    )
    
    # --- UPDATED PROMPT WITH NEW INPUTS ---
    user_msg = (
        f"Generate {n_variants} professional job ad variants for:\n"
        f"Role: {user_input.get('role')}\n"
        f"Company: {user_input.get('company')}\n"
        f"Location: {user_input.get('location')}\n"
        f"Salary Offered: {user_input.get('salary')}\n"
        f"Key Tech Stack Required: {user_input.get('tech_stack')}\n\n"
        f"Context from similar jobs (for inspiration):\n{context_block}\n\n"
        "Instructions:\n"
        "1. Ensure the 'Key Tech Stack' is listed prominently in requirements.\n"
        "2. Mention the Salary if it is competitive.\n"
        "Output Format: A single JSON array of objects. Each object must have keys: "
        "'title', 'intro', 'responsibilities' (list), 'requirements' (list), 'perks'."
    )
    return system_msg, user_msg

def try_parse_json(s):
    if not s: return None
    s = s.strip()
    # Clean markdown if Llama adds it
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"): lines = lines[1:]
        if lines[-1].startswith("```"): lines = lines[:-1]
        s = "\n".join(lines)
    try:
        return json.loads(s)
    except Exception:
        return None


# --- 4. STREAMLIT UI ---
with st.sidebar:
    st.header("Job Details")
    
    # --- NEW INPUT FIELDS ---
    role = st.text_input("Role", value="Data Scientist")
    company = st.text_input("Company", value="Tech Corp")
    location = st.text_input("Location", value="Remote")
    
    st.markdown("---")
    salary = st.text_input("Salary / Package", value="Competitive")
    tech_stack = st.text_area("Key Tech Stack (comma separated)", value="Python, SQL, AWS, Machine Learning")
    
    generate_btn = st.button("Generate Ads 🚀", type="primary")

if generate_btn:
    st.info(f"🔍 Searching DB for '{role}'...")
    query = f"{role} {location} {company} {tech_stack}"
    retrieved = retrieve(query, index, records, embed_model)
    
    st.info("⚡ Calling Groq (Llama 3)...")
    
    # Pass all new inputs to the builder
    user_inputs = {
        "role": role, 
        "company": company, 
        "location": location,
        "salary": salary,
        "tech_stack": tech_stack
    }
    
    system_msg, user_msg = build_prompt(user_inputs, retrieved, N_VARIANTS)
    
    try:
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            model=GROQ_MODEL,
            temperature=0.5,
        )
        
        resp_text = chat_completion.choices[0].message.content
        parsed_json = try_parse_json(resp_text)
        
        if parsed_json:
            st.success("✅ Generated!")
            cols = st.columns(3)
            for idx, job in enumerate(parsed_json):
                with cols[idx]:
                    st.subheader(f"Option {idx+1}")
                    st.markdown(f"**{job.get('title')}**")
                    st.write(job.get('intro'))
                    
                    st.markdown("#### Responsibilities")
                    for r in job.get('responsibilities', []): 
                        st.markdown(f"- {r}")
                        
                    st.markdown("#### Requirements")
                    for r in job.get('requirements', []): 
                        st.markdown(f"- {r}")
                        
                    st.markdown(f"**Perks:** {job.get('perks')}")
        else:
            st.error("JSON Error. Raw Output:")
            st.code(resp_text)
            
    except Exception as e:
        st.error(f"Groq Error: {e}")
