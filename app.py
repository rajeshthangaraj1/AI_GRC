import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi

# =========================
# CONFIG
# =========================
QDRANT_PATH = "./data"
COLLECTION_NAME = "grc_docs"
EMBED_MODEL_PATH = "/home/rmrobot/Desktop/Rajesh/AI_GRC/models/bge-large"
RERANK_MODEL    = "/home/rmrobot/Desktop/Rajesh/AI_GRC/models/bge-reranker-large"
OLLAMA_MODEL    = "mistral-small3.2"
TOP_K_RETRIEVE  = 40
TOP_K_RERANK    = 10
RERANK_MIN_SCORE = 0.0   # docs below this reranker score are dropped
MAX_SCROLL       = 20000  # warn if collection grows beyond this
HISTORY_TURNS    = 4      # number of prior conversation turns passed to LLM

# Keywords for auto-detecting framework from query text
FRAMEWORK_KEYWORDS = {
    "EU AI Act":   ["eu ai act", "eu act", "prohibited ai", "high-risk ai", "article 6", "article 9"],
    "NIST AI RMF": ["nist", "rmf", "ai rmf", "govern function", "map function", "measure function", "manage function"],
    "ISO":         ["iso", "42001", "42k", "annex a", "aims", "iso/iec"],
}

FRAMEWORK_COLORS = {
    "EU AI Act":    "#1565C0",
    "NIST AI RMF":  "#2E7D32",
    "ISO":          "#6A1B9A",
    "General":      "#424242",
}

SAMPLE_QUESTIONS = {
    "EU AI Act": [
        "What are the prohibited AI practices under EU AI Act?",
        "What transparency obligations apply to AI systems?",
        "What are high-risk AI system requirements?",
    ],
    "NIST AI RMF": [
        "What does NIST AI RMF GOVERN function focus on?",
        "How does NIST approach AI risk measurement?",
        "What is the MANAGE function in NIST AI RMF?",
    ],
    "ISO 42001": [
        "What are the main requirements of ISO 42001?",
        "Give me clause 4 of ISO 42K.",
        "What does ISO 42K Annex A A.2.2 require?",
        "What are the AI policy requirements under ISO?",
        "List all controls for ISO 42001 Annex A",
        "Give the topic coverage in Annex A",
        "Give Controls applicable to Developer/Provider",
        "Give Controls applicable to AI Consumer/User",
        "Provide assessment questions for Annex A section A.2",
        "Give me the management clauses for ISO 42001",
        "What are the assessment questions for auditing management clauses",
        "How many controls does ISO 42001 have?",
        "Provide terms and definitions for ISO 42001",
        "What are the AI objectives for an organization under ISO 42001",
        "What are the risk sources for AI in an enterprise",
    ],
    "ISO Implementation": [
        "Give me an ISO 42001 Overview",
        "Walk me through the Implementation Phases for ISO 42001",
        "Tell me about Phase 1: Preparation & Current State Assessment",
        "What is AI Risk Assessment Methodology",
        "Understanding the ISO Certification Timeline",
        "Tell me about Phase 5: Certification Audit",
        "What are the Benefits of ISO/IEC 42001 Implementation",
        "What is AI Ethics and Responsible AI",
    ],
    "Cross-Framework": [
        "If I build a facial recognition system, what compliance requirements apply?",
        "What are the similarities between NIST and ISO AI governance frameworks?",
    ],
}

# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="AI GRC Compliance Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main header */
    .grc-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B2838 60%, #162032 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #2A3F5F;
    }
    .grc-header h1 {
        color: #E8F4FD;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .grc-header p {
        color: #90CAF9;
        font-size: 0.85rem;
        margin: 0;
    }

    /* Framework badge */
    .fw-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        color: white;
        margin: 2px 3px;
    }

    /* Source block */
    .source-block {
        background: #F8F9FA;
        border-left: 3px solid #1565C0;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin-top: 0.8rem;
        font-size: 0.78rem;
        color: #555;
    }

    /* Stat card */
    .stat-card {
        background: #F0F4FF;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        text-align: center;
        border: 1px solid #DDEEFF;
    }
    .stat-number { font-size: 1.4rem; font-weight: 700; color: #1565C0; }
    .stat-label  { font-size: 0.72rem; color: #666; }

    /* Sidebar brand */
    .sidebar-brand {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #E0E0E0;
        margin-bottom: 1rem;
    }
    .sidebar-brand h2 { font-size: 1.1rem; font-weight: 700; color: #0D1B2A; margin: 0.3rem 0 0.1rem 0; }
    .sidebar-brand p  { font-size: 0.72rem; color: #888; margin: 0; }

    /* Sample question button */
    div[data-testid="stButton"] button {
        text-align: left;
        font-size: 0.78rem;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource(show_spinner="Loading AI models — please wait...")
def load_models():
    embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")
    reranker    = CrossEncoder(RERANK_MODEL)
    qdrant      = QdrantClient(path=QDRANT_PATH)
    llm         = OllamaLLM(model=OLLAMA_MODEL)

    scroll = qdrant.scroll(collection_name=COLLECTION_NAME, limit=MAX_SCROLL)
    if len(scroll[0]) == MAX_SCROLL:
        import warnings
        warnings.warn(f"BM25 index scroll hit limit of {MAX_SCROLL}. Some documents may be missing from BM25 index.")
    all_docs = []
    framework_counts = {}

    for point in scroll[0]:
        p = point.payload
        bm25_text = p.get("bm25_text", p["text"])
        fw = p.get("framework", "General")
        framework_counts[fw] = framework_counts.get(fw, 0) + 1
        all_docs.append({
            "text":      p["text"],
            "bm25_text": bm25_text,
            "framework": fw,
            "sheet":     p.get("sheet", ""),
        })

    tokenized = [doc["bm25_text"].split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized)

    return embed_model, reranker, qdrant, llm, bm25, all_docs, framework_counts


embed_model, reranker, qdrant_client, llm, bm25, all_docs, framework_counts = load_models()
total_controls = len(all_docs)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# =========================
# RAG PIPELINE
# =========================
def refine_query(query):
    return query + " compliance control risk requirement"

def detect_framework(query):
    """Return a framework filter list if the query clearly targets one framework.
    Returns None if query is ambiguous or targets multiple frameworks."""
    q = query.lower()
    matched = []
    for fw, keywords in FRAMEWORK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matched.append(fw)
    return matched if len(matched) == 1 else None

def hybrid_retrieve(query, framework_filter=None):
    # Auto-detect framework if user hasn't manually restricted
    if not framework_filter:
        framework_filter = detect_framework(query)

    refined = refine_query(query)

    # BM25
    tokenized_q  = refined.lower().split()
    bm25_scores  = bm25.get_scores(tokenized_q)
    bm25_idx     = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVE]
    bm25_docs    = [all_docs[i] for i in bm25_idx]

    # Vector search (with optional framework filter)
    query_vec = embed_model.encode(f"query: {refined}", normalize_embeddings=True)

    qdrant_filter = None
    if framework_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="framework", match=MatchAny(any=framework_filter))]
        )

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        limit=TOP_K_RETRIEVE,
        query_filter=qdrant_filter,
    ).points
    vector_docs = [{"text": r.payload["text"], "framework": r.payload.get("framework",""), "sheet": r.payload.get("sheet","")} for r in results]

    # Apply framework filter to BM25 results too
    if framework_filter:
        bm25_docs = [d for d in bm25_docs if d["framework"] in framework_filter]

    # Merge & deduplicate
    combined, seen = [], set()
    for doc in bm25_docs + vector_docs:
        if doc["text"] not in seen:
            combined.append(doc)
            seen.add(doc["text"])

    return combined, refined

def rerank_docs(query, docs):
    if not docs:
        return []
    pairs  = [[query, d["text"]] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    # Keep only docs above the minimum score threshold
    filtered = [d for s, d in ranked if s > RERANK_MIN_SCORE]
    # Always return at least top-3 as fallback even if all scores are low
    return filtered[:TOP_K_RERANK] if filtered else [d for _, d in ranked[:3]]

def build_context(docs):
    return "\n\n---\n\n".join(d["text"] for d in docs)

def generate_answer(query, context, history=None):
    # Build conversation history block (last N turns, truncated to avoid token bloat)
    history_block = ""
    if history:
        lines = []
        for msg in history[-(HISTORY_TURNS * 2):]:   # last N user+assistant pairs
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:400]}")
        history_block = "CONVERSATION HISTORY (recent exchanges):\n" + "\n".join(lines) + "\n\n"

    prompt = f"""You are an AI Governance, Risk, and Compliance (GRC) expert assistant.

Rules:
- Answer using ONLY the provided context below
- Do NOT hallucinate or guess
- If the answer is not in the context, say: "This information is not available in the loaded compliance documents."
- NEVER include placeholder values like TBD, TBA, N/A, "Not Assessed" in your answer — skip those fields entirely
- Be clear, structured, and professional
- If the question refers to a previous answer (e.g. "tell me more", "what about point 3"), use the conversation history above

If a checklist or list is requested:
- Use bullet points or numbered lists
- Include clause/control ID, requirement, and risk level where available

----------------
{history_block}CONTEXT:
{context}

----------------
QUESTION:
{query}

ANSWER:"""
    return llm.invoke(prompt)

def rag_pipeline(query, framework_filter=None, history=None):
    docs, refined = hybrid_retrieve(query, framework_filter)
    if not docs:
        return "No documents retrieved.", [], refined

    docs = rerank_docs(refined, docs)
    if not docs:
        return "No relevant documents found after reranking.", [], refined

    context = build_context(docs)
    answer  = generate_answer(query, context, history=history)

    # Extract unique sources for attribution
    sources = []
    seen_src = set()
    for d in docs:
        key = (d["framework"], d["sheet"])
        if key not in seen_src:
            sources.append({"framework": d["framework"], "sheet": d["sheet"]})
            seen_src.add(key)

    return answer, sources, refined

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2rem">🛡️</div>
        <h2>AI GRC Advisor</h2>
        <p>Powered by Hybrid RAG + Reranker</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Stats ---
    st.markdown("**Coverage**")
    cols = st.columns(2)
    cols[0].markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{total_controls:,}</div>
        <div class="stat-label">Controls Indexed</div>
    </div>""", unsafe_allow_html=True)
    cols[1].markdown(f"""
    <div class="stat-card">
        <div class="stat-number">3</div>
        <div class="stat-label">Frameworks</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    for fw, color in FRAMEWORK_COLORS.items():
        if fw in framework_counts and fw != "General":
            count = framework_counts.get(fw, 0)
            st.markdown(
                f'<span class="fw-badge" style="background:{color}">{fw}</span> '
                f'<span style="font-size:0.8rem;color:#555">{count} controls</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Framework Filter ---
    st.markdown("**Filter by Framework**")
    all_frameworks = ["EU AI Act", "NIST AI RMF", "ISO"]
    selected_frameworks = []
    for fw in all_frameworks:
        color = FRAMEWORK_COLORS.get(fw, "#424242")
        if st.checkbox(fw, value=True, key=f"fw_{fw}"):
            selected_frameworks.append(fw)

    framework_filter = selected_frameworks if len(selected_frameworks) < 3 else None

    st.divider()

    # --- Sample Questions ---
    st.markdown("**Sample Questions**")
    for category, questions in SAMPLE_QUESTIONS.items():
        with st.expander(category, expanded=False):
            for q in questions:
                if st.button(q, key=f"sq_{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    st.divider()

    # --- Controls ---
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<div style='font-size:0.7rem;color:#aaa;text-align:center;margin-top:1rem'>"
        "Local LLM · No data leaves your environment"
        "</div>",
        unsafe_allow_html=True,
    )

# =========================
# MAIN AREA
# =========================
st.markdown("""
<div class="grc-header">
    <h1>🛡️ AI GRC Compliance Advisor</h1>
    <p>
        Ask questions across <strong>EU AI Act</strong> · <strong>NIST AI RMF</strong> · <strong>ISO 42001</strong>
        &nbsp;|&nbsp; Hybrid Search · Neural Reranking · Local LLM
    </p>
</div>
""", unsafe_allow_html=True)

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            src_html = "<div class='source-block'><strong>Sources:</strong> "
            for s in msg["sources"]:
                color = FRAMEWORK_COLORS.get(s["framework"], "#424242")
                src_html += f'<span class="fw-badge" style="background:{color}">{s["framework"]}</span> '
                if s["sheet"]:
                    src_html += f'<span style="font-size:0.72rem;color:#666">{s["sheet"]}</span>  '
            src_html += "</div>"
            st.markdown(src_html, unsafe_allow_html=True)

# --- Handle input (typed or sample question button) ---
typed_input = st.chat_input("Ask a compliance question...")
user_input  = st.session_state.pending_question or typed_input

if st.session_state.pending_question:
    st.session_state.pending_question = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing compliance requirements..."):
            try:
                # Pass all messages except the current one as conversation history
                history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else None
                answer, sources, refined = rag_pipeline(user_input, framework_filter, history=history)

                st.markdown(answer)

                # Source attribution
                if sources:
                    src_html = "<div class='source-block'><strong>Sources:</strong> "
                    for s in sources:
                        color = FRAMEWORK_COLORS.get(s["framework"], "#424242")
                        src_html += f'<span class="fw-badge" style="background:{color}">{s["framework"]}</span> '
                        if s["sheet"]:
                            src_html += f'<span style="font-size:0.72rem;color:#666">{s["sheet"]}</span>  '
                    src_html += "</div>"
                    st.markdown(src_html, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
