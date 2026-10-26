import streamlit as st

from components.navbar import render_navbar


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

header[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

#MainMenu {
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}

[data-testid="stSidebar"] {
    display: none !important;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(99,102,241,0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 15%,
            rgba(14,165,233,0.10),
            transparent 28%
        ),
        #f8fafc !important;
}

.block-container {
    max-width: 1250px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;
    min-height: 470px;
    display: flex;
    align-items: center;
    padding: 65px;
    border-radius: 30px;
    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #1e1b4b 35%,
            #3730a3 65%,
            #0369a1 100%
        );
    box-shadow:
        0 25px 65px rgba(15,23,42,0.20);
    box-sizing: border-box;
}

.hero::before {
    content: "";
    position: absolute;
    width: 500px;
    height: 500px;
    right: -220px;
    top: -250px;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            rgba(255,255,255,0.15),
            transparent 70%
        );
}

.hero::after {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    left: 55%;
    bottom: -350px;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            rgba(34,211,238,0.20),
            transparent 70%
        );
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 780px;
}

.hero-kicker {
    margin-bottom: 18px;
    color: #bae6fd;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}

.hero-title {
    margin: 0 0 20px 0;
    color: white;
    font-size: 58px;
    line-height: 1.04;
    font-weight: 850;
    letter-spacing: -2.5px;
}

.hero-highlight {
    background:
        linear-gradient(
            90deg,
            #a5b4fc,
            #67e8f9
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    max-width: 700px;
    margin-bottom: 28px;
    color: rgba(255,255,255,0.76);
    font-size: 16px;
    line-height: 1.75;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
}

.hero-tag {
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: #e0f2fe;
    font-size: 10px;
    font-weight: 750;
}


/* ============================================================
   SECTIONS
   ============================================================ */

.section {
    margin-top: 65px;
}

.section-title {
    margin-bottom: 10px;
    color: #111827;
    font-size: 30px;
    font-weight: 850;
    letter-spacing: -0.8px;
}

.section-description {
    max-width: 760px;
    margin-bottom: 30px;
    color: #64748b;
    font-size: 14px;
    line-height: 1.75;
}


/* ============================================================
   FEATURE CARDS
   ============================================================ */

.feature-card {
    min-height: 220px;
    padding: 28px;
    border-radius: 20px;
    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fafc
        );
    border: 1px solid #e2e8f0;
    box-shadow:
        0 12px 35px rgba(15,23,42,0.06);
    box-sizing: border-box;
}

.feature-icon {
    margin-bottom: 15px;
    font-size: 30px;
}

.feature-title {
    margin-bottom: 10px;
    color: #111827;
    font-size: 18px;
    font-weight: 800;
}

.feature-description {
    color: #64748b;
    font-size: 13px;
    line-height: 1.75;
}


/* ============================================================
   PIPELINE
   ============================================================ */

.pipeline-box {
    padding: 35px;
    border-radius: 23px;
    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #ecfeff
        );
    border: 1px solid #c7d2fe;
    box-sizing: border-box;
}

.pipeline-title {
    margin-bottom: 28px;
    text-align: center;
    color: #312e81;
    font-size: 19px;
    font-weight: 800;
}

.pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    flex-wrap: wrap;
}

.pipeline-step {
    min-width: 105px;
    padding: 15px 13px;
    text-align: center;
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 14px;
}

.pipeline-number {
    width: 31px;
    height: 31px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 8px;
    border-radius: 50%;
    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #0891b2
        );
    color: white;
    font-size: 11px;
    font-weight: 800;
}

.pipeline-name {
    color: #334155;
    font-size: 10px;
    font-weight: 750;
}

.pipeline-arrow {
    color: #64748b;
    font-size: 19px;
}


/* ============================================================
   CTA
   ============================================================ */

.cta {
    margin-top: 65px;
    padding: 50px 35px;
    text-align: center;
    border-radius: 25px;
    background:
        linear-gradient(
            135deg,
            #111827,
            #312e81,
            #0369a1
        );
    box-shadow:
        0 20px 50px rgba(15,23,42,0.16);
}

.cta-title {
    margin-bottom: 10px;
    color: white;
    font-size: 29px;
    font-weight: 850;
}

.cta-description {
    max-width: 650px;
    margin: 0 auto;
    color: rgba(255,255,255,0.72);
    font-size: 13px;
    line-height: 1.7;
}


/* ============================================================
   BUTTONS
   ============================================================ */

div.stButton > button {
    border-radius: 10px !important;
    min-height: 44px !important;
    font-weight: 700 !important;
    transition: all 0.18s ease !important;
}

div.stButton > button:hover {
    transform: translateY(-1px);
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 800px) {

    .hero {
        min-height: 400px;
        padding: 40px 28px;
    }

    .hero-title {
        font-size: 40px;
        letter-spacing: -1.5px;
    }

    .hero-description {
        font-size: 14px;
    }

    .pipeline-arrow {
        display: none;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NAVBAR
# ============================================================

render_navbar()


# ============================================================
# HERO
# ============================================================

hero_html = """
<div class="hero">
<div class="hero-content">
<div class="hero-kicker">🛰️ AGENTIC REMOTE SENSING INTELLIGENCE</div>
<div class="hero-title">Understand Satellite Imagery <span class="hero-highlight">with AI.</span></div>
<div class="hero-description">SatQuery AI transforms natural-language questions into evidence-grounded satellite-image analysis. Combining vision-language intelligence, semantic retrieval, agentic reasoning and visual grounding into one unified remote-sensing platform.</div>
<div class="hero-tags">
<div class="hero-tag">VISION</div>
<div class="hero-tag">SEMANTIC RETRIEVAL</div>
<div class="hero-tag">AGENTIC AI</div>
<div class="hero-tag">VISUAL GROUNDING</div>
</div>
</div>
</div>
"""

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# ============================================================
# HERO BUTTONS
# ============================================================

col1, col2, col3 = st.columns(
    [1.4, 1.4, 2.2],
    gap="medium",
)

with col1:
    if st.button(
        "🚀 Start AI Analysis",
        use_container_width=True,
        type="primary",
    ):
        st.switch_page(
            "pages/1_AI_Analysis.py"
        )

with col2:
    if st.button(
        "📖 Explore SatQuery",
        use_container_width=True,
    ):
        st.switch_page(
            "pages/2_About_SatQuery.py"
        )


# ============================================================
# WHAT IS SATQUERY
# ============================================================

what_is_html = """
<div class="section">
<div class="section-title">What is SatQuery AI?</div>
<div class="section-description">SatQuery AI is a query-driven agentic remote-sensing intelligence system that allows users to interact with satellite imagery using natural language. Instead of requiring users to understand complex remote-sensing workflows, the system automatically routes their request through the appropriate AI analysis pipeline.</div>
</div>
"""

st.markdown(
    what_is_html,
    unsafe_allow_html=True,
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(
    3,
    gap="large",
)

with col1:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🧠</div>
<div class="feature-title">Vision-Language Intelligence</div>
<div class="feature-description">Vision-language models analyse satellite imagery and answer natural-language questions about the visual scene.</div>
</div>
""",
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">🔎</div>
<div class="feature-title">Semantic Retrieval</div>
<div class="feature-description">Satellite imagery is divided into visual patches. CLIP embeddings and FAISS retrieval identify regions relevant to the user's question.</div>
</div>
""",
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
<div class="feature-card">
<div class="feature-icon">📍</div>
<div class="feature-title">Visual Grounding</div>
<div class="feature-description">Relevant regions can be mapped back onto the original satellite image to provide interpretable visual evidence.</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# PIPELINE
# ============================================================

pipeline_html = """
<div class="section">
<div class="section-title">How SatQuery AI Works</div>
<div class="section-description">A user's image and natural-language question pass through specialised components before the final response is produced.</div>

<div class="pipeline-box">
<div class="pipeline-title">SatQuery AI Analysis Pipeline</div>

<div class="pipeline">

<div class="pipeline-step">
<div class="pipeline-number">1</div>
<div class="pipeline-name">Image</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">2</div>
<div class="pipeline-name">Query</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">3</div>
<div class="pipeline-name">Router</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">4</div>
<div class="pipeline-name">Retrieval</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">5</div>
<div class="pipeline-name">Specialist AI</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">6</div>
<div class="pipeline-name">Evidence</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">7</div>
<div class="pipeline-name">Answer</div>
</div>

</div>
</div>
</div>
"""

st.markdown(
    pipeline_html,
    unsafe_allow_html=True,
)


# ============================================================
# CTA
# ============================================================

cta_html = """
<div class="cta">
<div class="cta-title">Ready to explore your satellite imagery?</div>
<div class="cta-description">Upload a satellite image, ask a question in natural language and let SatQuery AI perform the analysis.</div>
</div>
"""

st.markdown(
    cta_html,
    unsafe_allow_html=True,
)


# ============================================================
# CTA BUTTONS
# ============================================================

col1, col2, col3 = st.columns(
    [1.3, 1.3, 2.4],
    gap="medium",
)

with col1:

    if st.button(
        "🛰️ Analyze an Image",
        use_container_width=True,
        type="primary",
    ):
        st.switch_page(
            "pages/1_AI_Analysis.py"
        )


with col2:

    if st.button(
        "📚 Learn More",
        use_container_width=True,
    ):
        st.switch_page(
            "pages/2_About_SatQuery.py"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div style="
    margin-top:55px;
    padding-top:22px;
    border-top:1px solid #e2e8f0;
    text-align:center;
    color:#94a3b8;
    font-size:11px;
">
🛰️ SatQuery AI · TechNauts
<br>
Agentic Vision-Language Satellite Analysis
</div>
""",
    unsafe_allow_html=True,
)