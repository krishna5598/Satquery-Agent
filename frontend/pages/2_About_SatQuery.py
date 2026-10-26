import streamlit as st

from components.navbar import render_navbar


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="About SatQuery AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
/* Hide Streamlit default header */
header[data-testid="stHeader"] {
    display: none;
}

/* Remove extra top spacing */
.block-container {
    padding-top: 1rem !important;
}

/* Hide Streamlit decoration */
[data-testid="stDecoration"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)
st.html("""
<style>

.stApp {

    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(99,102,241,0.08),
            transparent 28%
        ),

        radial-gradient(
            circle at 100% 10%,
            rgba(14,165,233,0.08),
            transparent 30%
        ),

        #f8fafc;
}


.block-container {

    max-width:1200px;

    padding-top:1.2rem;

    padding-bottom:4rem;
}


#MainMenu {
    visibility:hidden;
}


footer {
    visibility:hidden;
}


.hero {

    padding:55px;

    border-radius:28px;

    background:
        linear-gradient(
            135deg,
            #0f172a,
            #312e81,
            #0369a1
        );

    color:white;

    box-shadow:
        0 25px 60px rgba(15,23,42,0.16);

    margin-bottom:45px;
}


.kicker {

    color:#bae6fd;

    font-size:11px;

    font-weight:800;

    letter-spacing:1.5px;

    text-transform:uppercase;

    margin-bottom:12px;
}


.title {

    font-size:45px;

    font-weight:850;

    letter-spacing:-1.5px;

    margin-bottom:15px;
}


.description {

    max-width:780px;

    color:rgba(255,255,255,0.75);

    font-size:15px;

    line-height:1.75;
}


.section-title {

    margin-top:40px;

    margin-bottom:12px;

    color:#111827;

    font-size:28px;

    font-weight:850;
}


.section-text {

    color:#64748b;

    font-size:14px;

    line-height:1.8;
}


.info-card {

    min-height:160px;

    padding:25px;

    border-radius:18px;

    background:white;

    border:1px solid #e2e8f0;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.05);
}


.info-title {

    margin-bottom:9px;

    color:#111827;

    font-size:17px;

    font-weight:800;
}


.info-text {

    color:#64748b;

    font-size:12px;

    line-height:1.7;
}


.footer {

    margin-top:55px;

    padding-top:20px;

    border-top:1px solid #e2e8f0;

    text-align:center;

    color:#94a3b8;

    font-size:11px;
}

</style>
""")


# ============================================================
# NAVBAR
# ============================================================

render_navbar()


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="kicker">
        🛰️ TECHNAUTS · SATQUERY AI
    </div>

    <div class="title">
        About SatQuery AI
    </div>

    <div class="description">
        SatQuery AI is an agentic vision-language system
        designed to make satellite-image analysis accessible
        through natural-language interaction.
    </div>

</div>
""")


# ============================================================
# OVERVIEW
# ============================================================

st.html("""
<div class="section-title">
    Project Overview
</div>

<div class="section-text">
    SatQuery AI combines vision-language models,
    semantic retrieval, agentic orchestration and
    visual grounding into a unified satellite-image
    analysis workflow.
</div>
""")


# ============================================================
# CAPABILITIES
# ============================================================

st.html("""
<div class="section-title">
    Core Capabilities
</div>
""")


col1, col2, col3 = st.columns(
    3,
    gap="large",
)


with col1:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            🧠 Vision-Language AI
        </div>

        <div class="info-text">
            Analyse satellite imagery using specialised
            vision-language models and natural-language
            questions.
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            🔎 Semantic Retrieval
        </div>

        <div class="info-text">
            Retrieve image regions that are semantically
            relevant to the user's query.
        </div>

    </div>
    """)


with col3:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            📍 Visual Grounding
        </div>

        <div class="info-text">
            Map retrieved regions back onto the original
            image to provide interpretable visual evidence.
        </div>

    </div>
    """)


# ============================================================
# ARCHITECTURE
# ============================================================

st.html("""
<div class="section-title">
    System Architecture
</div>

<div class="section-text">
    SatQuery AI follows a modular pipeline connecting
    image understanding, semantic retrieval, agentic
    routing, specialist analysis and evidence generation.
</div>
""")


st.code(
    """
Satellite Image + Natural-Language Query
                    │
                    ▼
              Query Router
                    │
                    ▼
            Semantic Retrieval
                    │
                    ▼
          Relevant Image Patches
                    │
                    ▼
          Specialist Vision Model
                    │
                    ▼
             Evidence Layer
                    │
                    ▼
              Final Answer
                    │
                    ▼
          Visual Grounding
    """,
    language="text",
)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.html("""
<div class="section-title">
    Technology Stack
</div>
""")


col1, col2, col3 = st.columns(
    3,
    gap="large",
)


with col1:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            🤖 AI / ML
        </div>

        <div class="info-text">
            PyTorch<br>
            Transformers<br>
            Vision-Language Models<br>
            LoRA / PEFT<br>
            CLIP<br>
            FAISS
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            🔗 Agentic AI
        </div>

        <div class="info-text">
            LangGraph<br>
            Query Routing<br>
            Tool Selection<br>
            Evidence Aggregation<br>
            LLM Response Generation
        </div>

    </div>
    """)


with col3:

    st.html("""
    <div class="info-card">

        <div class="info-title">
            🌐 Application
        </div>

        <div class="info-text">
            Python<br>
            Streamlit<br>
            FastAPI<br>
            Pillow<br>
            REST APIs
        </div>

    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    🛰️ SatQuery AI · TechNauts
</div>
""")