import html
import io
import requests
import streamlit as st

from PIL import Image, ImageOps
from components.navbar import render_navbar

render_navbar()
# =========================================================
# CONFIG
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"

IMAGE_WIDTH = 420
IMAGE_HEIGHT = 260

REQUEST_TIMEOUT = 600


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

/* =====================================================
   GLOBAL
   ===================================================== */

header[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding-top: 1rem !important;
}

[data-testid="stDecoration"] {
    display: none;
}


/* =====================================================
   APP BACKGROUND
   ===================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(99, 102, 241, 0.08),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(14, 165, 233, 0.08),
            transparent 28%
        ),
        #f7f9fc;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #111827 0%,
            #1e1b4b 42%,
            #312e81 72%,
            #0369a1 100%
        );

    border-radius: 24px;

    padding: 34px 38px;

    margin-bottom: 20px;

    box-shadow:
        0 18px 45px rgba(30, 41, 59, 0.18);
}

.hero::before {
    content: "";

    position: absolute;

    width: 350px;
    height: 350px;

    right: -120px;
    top: -190px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255,255,255,0.15),
            transparent 68%
        );
}

.hero::after {
    content: "";

    position: absolute;

    width: 280px;
    height: 280px;

    left: 45%;
    bottom: -240px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(56,189,248,0.18),
            transparent 68%
        );
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    color: white;

    font-size: 38px;

    font-weight: 800;

    letter-spacing: -1px;

    margin: 0;
}

.hero-subtitle {
    color: rgba(255,255,255,0.78);

    font-size: 14px;

    margin-top: 8px;
}

.hero-pill {
    display: inline-block;

    margin-top: 17px;

    padding: 6px 13px;

    border-radius: 999px;

    background: rgba(255,255,255,0.11);

    border: 1px solid rgba(255,255,255,0.20);

    color: #e0f2fe;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 0.6px;
}


/* =====================================================
   PIPELINE
   ===================================================== */

.pipeline {
    background: rgba(255,255,255,0.86);

    border: 1px solid #e2e8f0;

    border-radius: 16px;

    padding: 13px 18px;

    margin-bottom: 30px;

    box-shadow:
        0 6px 20px rgba(15,23,42,0.04);
}

.pipeline-inner {
    display: flex;

    align-items: center;

    justify-content: center;

    gap: 12px;
}

.pipeline-step {
    display: flex;

    align-items: center;

    gap: 7px;

    color: #475569;

    font-size: 11px;

    font-weight: 700;

    white-space: nowrap;
}

.pipeline-icon {
    width: 26px;
    height: 26px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 50%;

    color: white;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #0891b2
        );

    font-size: 11px;
}

.pipeline-arrow {
    color: #94a3b8;

    font-size: 14px;

    font-weight: 700;
}


/* =====================================================
   SECTION
   ===================================================== */

.section-title {
    font-size: 24px;

    font-weight: 800;

    color: #111827;

    letter-spacing: -0.4px;
}

.section-subtitle {
    color: #64748b;

    font-size: 13px;

    margin-top: 3px;

    margin-bottom: 20px;
}


/* =====================================================
   INPUT CARDS
   ===================================================== */

.input-card-title {
    color: #111827;

    font-size: 16px;

    font-weight: 750;
}

.input-card-subtitle {
    color: #94a3b8;

    font-size: 12px;

    margin-top: 4px;

    margin-bottom: 14px;
}


/* =====================================================
   UPLOAD
   ===================================================== */

section[data-testid="stFileUploaderDropzone"] {

    background:
        linear-gradient(
            135deg,
            #f8faff 0%,
            #f0f9ff 100%
        ) !important;

    border: 1px dashed #818cf8 !important;

    border-radius: 14px !important;

    min-height: 100px;
}

section[data-testid="stFileUploaderDropzone"]:hover {

    border-color: #4f46e5 !important;

    background:
        linear-gradient(
            135deg,
            #eef2ff 0%,
            #ecfeff 100%
        ) !important;
}


/* =====================================================
   QUERY
   ===================================================== */

.query-label {
    font-size: 16px;

    font-weight: 750;

    color: #111827;
}

.query-hint {
    font-size: 12px;

    color: #64748b;

    margin-top: 4px;

    margin-bottom: 13px;
}

textarea {

    border-radius: 12px !important;

    border: 1px solid #cbd5e1 !important;

    background: #f8fafc !important;
}

textarea:focus {

    border-color: #6366f1 !important;

    box-shadow:
        0 0 0 2px rgba(99,102,241,0.12) !important;
}


/* =====================================================
   ANALYZE BUTTON
   ===================================================== */

.stButton > button {

    border: none !important;

    border-radius: 12px !important;

    height: 50px !important;

    font-size: 14px !important;

    font-weight: 750 !important;

    color: white !important;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #2563eb,
            #0891b2
        ) !important;

    box-shadow:
        0 9px 24px rgba(37,99,235,0.25) !important;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease !important;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 13px 30px rgba(37,99,235,0.32) !important;
}


/* =====================================================
   EMPTY RESULT
   ===================================================== */

.empty-result {

    min-height: 350px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    text-align: center;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.94),
            rgba(248,250,252,0.94)
        );

    border: 1px solid #e2e8f0;

    border-radius: 20px;

    box-shadow:
        0 10px 35px rgba(15,23,42,0.04);

    padding: 40px;
}

.empty-icon {

    width: 70px;
    height: 70px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #e0f2fe
        );

    font-size: 31px;

    margin-bottom: 18px;
}

.empty-title {

    font-size: 18px;

    font-weight: 750;

    color: #1e293b;

    margin-bottom: 7px;
}

.empty-text {

    max-width: 360px;

    color: #94a3b8;

    font-size: 13px;

    line-height: 1.6;
}


/* =====================================================
   ANSWER
   ===================================================== */

.answer-card {

    position: relative;

    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #f8faff 55%,
            #eef6ff 100%
        );

    border: 1px solid #dbeafe;

    border-radius: 16px;

    padding: 20px 22px;

    margin-bottom: 22px;

    box-shadow:
        0 8px 25px rgba(37,99,235,0.07);
}

.answer-card::before {

    content: "";

    position: absolute;

    left: 0;
    top: 0;
    bottom: 0;

    width: 5px;

    background:
        linear-gradient(
            180deg,
            #4f46e5,
            #0891b2
        );
}

.answer-label {

    font-size: 10px;

    font-weight: 800;

    color: #64748b;

    text-transform: uppercase;

    letter-spacing: 1px;

    margin-bottom: 9px;
}

.answer-text {

    color: #0f172a;

    font-size: 19px;

    font-weight: 650;

    line-height: 1.55;
}


/* =====================================================
   TASK
   ===================================================== */

.task-label {

    font-size: 11px;

    font-weight: 800;

    color: #64748b;

    text-transform: uppercase;

    letter-spacing: 0.7px;

    margin-bottom: 7px;
}

.task-badge {

    display: inline-block;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #ecfeff
        );

    color: #3730a3;

    border: 1px solid #c7d2fe;

    border-radius: 999px;

    padding: 6px 15px;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.7px;

    margin-bottom: 22px;
}


/* =====================================================
   VISUAL ANALYSIS
   ===================================================== */

.visual-heading {

    font-size: 18px;

    font-weight: 800;

    color: #111827;

    margin-bottom: 14px;
}

.visual-subtitle {

    color: #64748b;

    font-size: 11px;

    margin-bottom: 12px;
}

.image-heading {

    font-size: 12px;

    font-weight: 750;

    color: #334155;

    margin-bottom: 7px;
}


/* =====================================================
   TRACE
   ===================================================== */

.trace-heading {

    font-size: 18px;

    font-weight: 800;

    color: #111827;

    margin-top: 26px;

    margin-bottom: 12px;
}

.trace-step {

    background: #f8fafc;

    border: 1px solid #e2e8f0;

    border-radius: 8px;

    padding: 8px 12px;

    margin-bottom: 6px;

    font-size: 12px;

    color: #475569;
}


/* =====================================================
   LOADING PANEL
   ===================================================== */

.loading-card {

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8faff
        );

    border: 1px solid #dbeafe;

    border-radius: 20px;

    padding: 30px;

    margin-bottom: 22px;

    box-shadow:
        0 10px 35px rgba(37,99,235,0.08);

    text-align: center;
}

.loading-icon {

    font-size: 34px;

    margin-bottom: 10px;
}

.loading-title {

    color: #0f172a;

    font-size: 18px;

    font-weight: 800;

    margin-bottom: 6px;
}

.loading-subtitle {

    color: #64748b;

    font-size: 12px;

    margin-bottom: 22px;
}

.loading-bar {

    width: 100%;

    height: 7px;

    overflow: hidden;

    border-radius: 999px;

    background: #e2e8f0;
}

.loading-bar-inner {

    width: 45%;

    height: 100%;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #2563eb,
            #06b6d4
        );

    animation:
        loadingMove 1.5s ease-in-out infinite;
}

@keyframes loadingMove {

    0% {
        transform: translateX(-110%);
    }

    50% {
        transform: translateX(100%);
    }

    100% {
        transform: translateX(220%);
    }
}


/* =====================================================
   RESPONSIVE
   ===================================================== */

@media (max-width: 900px) {

    .hero {
        padding: 25px;
    }

    .hero-title {
        font-size: 30px;
    }

    .pipeline-inner {
        flex-wrap: wrap;
    }

    .pipeline-arrow {
        display: none;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def load_backend_image(image_url):
    """
    Load an image generated by the FastAPI backend.

    Supports:
        /artifacts/...
        http://...
        https://...
    """

    if not image_url:
        return None

    if image_url.startswith("http://") or image_url.startswith("https://"):
        url = image_url
    else:
        url = f"{BACKEND_URL.rstrip('/')}/{image_url.lstrip('/')}"

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    return Image.open(
        io.BytesIO(response.content)
    ).convert("RGB")


def fit_image(image):
    """
    Resize/crop an image for consistent frontend display.
    """

    if image is None:
        return None

    return ImageOps.fit(
        image,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def get_artifact_evidence(result):
    """
    Normalize the backend evidence structure.

    Current backend structure:

        result
        └── evidence
            └── evidence
                ├── patches
                ├── retrieved_image
                └── grounded_image

    Also supports a future flattened structure.
    """

    outer_evidence = result.get(
        "evidence",
        {},
    )

    if not isinstance(
        outer_evidence,
        dict,
    ):
        return {}

    nested_evidence = outer_evidence.get(
        "evidence"
    )

    if isinstance(
        nested_evidence,
        dict,
    ):
        return nested_evidence

    return outer_evidence


def fallback_artifact_url(
    request_id,
    filename,
):
    """
    Construct an artifact URL if the backend
    does not explicitly return one.
    """

    if not request_id:
        return None

    return (
        f"/artifacts/"
        f"{request_id}/"
        f"{filename}"
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-content">'
    '<div class="hero-title">🛰️ SatQuery AI</div>'
    '<div class="hero-subtitle">'
    'Intelligent satellite-image analysis powered by vision models, '
    'semantic retrieval, and an AI agent.'
    '</div>'
    '<div class="hero-pill">'
    'AI-POWERED REMOTE SENSING ANALYSIS'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# PIPELINE
# =========================================================

st.markdown(
    '<div class="pipeline">'
    '<div class="pipeline-inner">'

    '<div class="pipeline-step">'
    '<div class="pipeline-icon">1</div>'
    'Upload'
    '</div>'

    '<div class="pipeline-arrow">→</div>'

    '<div class="pipeline-step">'
    '<div class="pipeline-icon">2</div>'
    'Retrieve'
    '</div>'

    '<div class="pipeline-arrow">→</div>'

    '<div class="pipeline-step">'
    '<div class="pipeline-icon">3</div>'
    'Analyze'
    '</div>'

    '<div class="pipeline-arrow">→</div>'

    '<div class="pipeline-step">'
    '<div class="pipeline-icon">4</div>'
    'Ground'
    '</div>'

    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# TWO COLUMNS
# =========================================================

input_col, result_col = st.columns(
    [0.85, 1.45],
    gap="large",
)


# =========================================================
# LEFT COLUMN
# =========================================================

with input_col:

    st.markdown(
        '<div class="section-title">'
        'Analysis Input'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload one or more satellite images and ask SatQuery AI a question.'
        '</div>',
        unsafe_allow_html=True,
    )


    # =====================================================
    # IMAGE UPLOAD
    # =====================================================

    with st.container(border=True):

        st.markdown(
            '<div class="input-card-title">'
            '📡 Satellite Image'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="input-card-subtitle">'
            'Upload JPG, JPEG, PNG, TIF, or TIFF satellite imagery. Multiple images are supported.'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload satellite image(s)",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff",
            ],
            accept_multiple_files=True,
            label_visibility="visible",
            help=(
                "Upload one or more satellite images. "
                "For Optical + SAR analysis, upload both images; "
                "naming them optical.tif and sar.tif is recommended."
            ),
        )


    # =====================================================
    # IMAGE PREVIEW
    # =====================================================

    if uploaded_files:

        st.markdown(
            "### 🖼️ Image Preview"
        )

        preview_cols = st.columns(
            min(len(uploaded_files), 3),
            gap="small",
        )

        for index, uploaded_file in enumerate(uploaded_files):

            preview_col = preview_cols[index % len(preview_cols)]

            try:
                preview_image = Image.open(
                    uploaded_file
                ).convert("RGB")

                with preview_col:

                    with st.container(border=True):

                        st.image(
                            preview_image,
                            use_container_width=True,
                        )

                        width, height = preview_image.size

                        st.caption(
                            f"**{index + 1}.** {uploaded_file.name}"
                            f"  \\n📐 {width} × {height}px"
                        )

            except Exception as exc:

                with preview_col:

                    st.error(
                        f"Could not preview {uploaded_file.name}: {exc}"
                    )

    # =====================================================
    # QUERY
    # =====================================================

    with st.container(border=True):

        st.markdown(
            '<div class="query-label">'
            '🔎 Natural-Language Query'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="query-hint">'
            'Ask SatQuery AI what you want to know about the image.'
            '</div>',
            unsafe_allow_html=True,
        )

        query = st.text_area(
            "Query",
            placeholder=(
                "Example:\n"
                "What is the category of this remote sensing image?"
            ),
            height=130,
            label_visibility="collapsed",
        )


    # =====================================================
    # ANALYZE
    # =====================================================

    analyze = st.button(
        "🚀  Analyze Satellite Image",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with result_col:

    st.markdown(
        '<div class="section-title">'
        'Analysis Result'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Results generated through the SatQuery AI analysis pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )


    # =====================================================
    # EMPTY STATE
    # =====================================================

    if not analyze:

        st.markdown(
            '<div class="empty-result">'

            '<div class="empty-icon">'
            '🛰️'
            '</div>'

            '<div class="empty-title">'
            'Ready for satellite analysis'
            '</div>'

            '<div class="empty-text">'
            'Upload one or more satellite images, enter a natural-language question, '
            'and start the SatQuery AI analysis pipeline.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    # =====================================================
    # ANALYSIS
    # =====================================================

    if analyze:

        if not uploaded_files:

            st.warning(
                "Please upload at least one satellite image."
            )

        elif not query.strip():

            st.warning(
                "Please enter a query."
            )

        else:

            # =================================================
            # LOADING EXPERIENCE
            # =================================================

            loading_placeholder = st.empty()

            loading_placeholder.markdown(
                '<div class="loading-card">'

                '<div class="loading-icon">'
                '🛰️'
                '</div>'

                '<div class="loading-title">'
                'SatQuery AI is analyzing your image(s)'
                '</div>'

                '<div class="loading-subtitle">'
                'Retrieving relevant regions and running the '
                'vision-language analysis pipeline...'
                '</div>'

                '<div class="loading-bar">'
                '<div class="loading-bar-inner"></div>'
                '</div>'

                '</div>',
                unsafe_allow_html=True,
            )


            # =================================================
            # BACKEND REQUEST
            # =================================================

            try:

                # FastAPI expects repeated "images" fields.
                # This allows one or many files in the same request.
                request_files = []

                for uploaded_file in uploaded_files:

                    mime_type = (
                        uploaded_file.type
                        or "application/octet-stream"
                    )

                    request_files.append(
                        (
                            "images",
                            (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                mime_type,
                            ),
                        )
                    )

                response = requests.post(
                    f"{BACKEND_URL}/analyze",

                    files=request_files,

                    data={
                        "query": query.strip(),
                    },

                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                result = response.json()

            except requests.RequestException as exc:

                loading_placeholder.empty()

                st.error(
                    f"Backend request failed: {exc}"
                )

                result = None

            except Exception as exc:

                loading_placeholder.empty()

                st.error(
                    f"Unexpected error: {exc}"
                )

                result = None


            # =================================================
            # REMOVE LOADING
            # =================================================

            if result:

                loading_placeholder.empty()


                # =============================================
                # BASIC RESPONSE DATA
                # =============================================

                answer = result.get(
                    "answer",
                    "No answer returned.",
                )

                task = str(
                    result.get(
                        "task",
                        "unknown",
                    )
                ).lower()

                request_id = result.get(
                    "request_id"
                )


                # =============================================
                # FINAL ANSWER
                # =============================================

                st.markdown(
                    "### ✨ Final Answer"
                )

                safe_answer = html.escape(
                    str(answer)
                )

                st.markdown(
                    '<div class="answer-card">'
                    '<div class="answer-label">'
                    'SATQUERY AI RESPONSE'
                    '</div>'
                    f'<div class="answer-text">'
                    f'{safe_answer}'
                    f'</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )


                # =============================================
                # TASK
                # =============================================

                st.markdown(
                    '<div class="task-label">'
                    'Analysis Task'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="task-badge">'
                    f'{html.escape(task.upper())}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                image_count = result.get(
                    "image_count",
                    len(uploaded_files),
                )

                st.caption(
                    f"📡 Images processed: **{image_count}**"
                )


                # =============================================
                # VISUAL ANALYSIS
                # =============================================

                st.markdown(
                    '<div class="visual-heading">'
                    '🖼️ Visual Analysis'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if task == "grounding":

                    visual_description = (
                        "Original imagery, retrieved regions, "
                        "and grounded analysis result."
                    )

                else:

                    visual_description = (
                        "Original imagery and the regions "
                        "retrieved as evidence for the analysis."
                    )

                st.markdown(
                    f'<div class="visual-subtitle">'
                    f'{visual_description}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


                # =============================================
                # NORMALIZE EVIDENCE
                # =============================================

                evidence = get_artifact_evidence(
                    result
                )


                # =============================================
                # ARTIFACT URLS
                # =============================================

                retrieved_url = evidence.get(
                    "retrieved_image"
                )

                grounded_url = evidence.get(
                    "grounded_image"
                )


                # -------------------------------------------------
                # FALLBACK: retrieved patches
                # -------------------------------------------------

                if not retrieved_url:

                    retrieved_url = fallback_artifact_url(
                        request_id,
                        "retrieved_patches.jpg",
                    )


                # -------------------------------------------------
                # FALLBACK: grounded result
                # -------------------------------------------------

                if task == "grounding" and not grounded_url:

                    grounded_url = fallback_artifact_url(
                        request_id,
                        "grounded_result.jpg",
                    )


                # =============================================
                # ORIGINAL IMAGES
                # =============================================

                original_images = []

                for uploaded_file in uploaded_files:

                    try:
                        original_image = Image.open(
                            io.BytesIO(
                                uploaded_file.getvalue()
                            )
                        ).convert("RGB")

                        original_images.append(
                            (
                                uploaded_file.name,
                                fit_image(original_image),
                            )
                        )

                    except Exception:
                        original_images.append(
                            (
                                uploaded_file.name,
                                None,
                            )
                        )


                # =============================================
                # RETRIEVED IMAGE
                # =============================================

                retrieved_image = None

                if retrieved_url:

                    try:

                        retrieved_image = load_backend_image(
                            retrieved_url
                        )

                        retrieved_image = fit_image(
                            retrieved_image
                        )

                    except Exception:

                        retrieved_image = None


                # =============================================
                # GROUNDED IMAGE
                #
                # ONLY LOAD FOR GROUNDING
                # =============================================

                grounded_image = None

                if task == "grounding" and grounded_url:

                    try:

                        grounded_image = load_backend_image(
                            grounded_url
                        )

                        grounded_image = fit_image(
                            grounded_image
                        )

                    except Exception:

                        grounded_image = None


                # =============================================
                # IMAGE LAYOUT
                #
                # GROUNDING:
                #   Original | Retrieved | Grounded
                #
                # OTHER TASKS:
                #   Original | Retrieved
                # =============================================

                if task == "grounding":

                    original_col, retrieved_col, grounded_col = st.columns(
                        3,
                        gap="small",
                    )

                else:

                    original_col, retrieved_col = st.columns(
                        2,
                        gap="small",
                    )


                # =============================================
                # ORIGINAL IMAGES
                # =============================================

                with original_col:

                    st.markdown(
                        '<div class="image-heading">'
                        '01 · Original Image(s)'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    with st.container(
                        border=True
                    ):

                        if original_images:

                            preview_display_cols = st.columns(
                                min(len(original_images), 2),
                                gap="small",
                            )

                            for index, (filename, original_image) in enumerate(
                                original_images
                            ):

                                display_col = preview_display_cols[
                                    index % len(preview_display_cols)
                                ]

                                with display_col:

                                    st.caption(
                                        f"**{index + 1}.** {filename}"
                                    )

                                    if original_image:

                                        st.image(
                                            original_image,
                                            use_container_width=True,
                                        )

                                    else:

                                        st.info(
                                            "Image unavailable."
                                        )

                        else:

                            st.info(
                                "Original image unavailable."
                            )


                # =============================================
                # RETRIEVED PATCHES
                # =============================================

                with retrieved_col:

                    st.markdown(
                        '<div class="image-heading">'
                        '02 · Retrieved Patches'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    with st.container(
                        border=True
                    ):

                        if retrieved_image:

                            st.image(
                                retrieved_image,
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                "Retrieved patches unavailable."
                            )


                # =============================================
                # GROUNDED RESULT
                #
                # ONLY FOR GROUNDING
                # =============================================

                if task == "grounding":

                    with grounded_col:

                        st.markdown(
                            '<div class="image-heading">'
                            '03 · Grounded Result'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        with st.container(
                            border=True
                        ):

                            if grounded_image:

                                st.image(
                                    grounded_image,
                                    use_container_width=True,
                                )

                            else:

                                st.info(
                                    "Grounded image unavailable."
                                )


                # =============================================
                # RETRIEVED PATCH METADATA
                # =============================================

                patches = evidence.get(
                    "patches",
                    [],
                )

                if patches:

                    with st.expander(
                        "View retrieved regions",
                        expanded=False,
                    ):

                        for rank, patch in enumerate(
                            patches,
                            start=1,
                        ):

                            patch_id = patch.get(
                                "id",
                                "?",
                            )

                            score = patch.get(
                                "score",
                                0.0,
                            )

                            bbox = patch.get(
                                "bbox",
                                [],
                            )

                            st.write(
                                f"**#{rank} — Patch {patch_id}**  "
                                f"| score: `{score:.4f}`  "
                                f"| bbox: `{bbox}`"
                            )


                # =============================================
                # TRACE
                # =============================================

                st.markdown(
                    '<div class="trace-heading">'
                    '⚙️ Agent Execution Trace'
                    '</div>',
                    unsafe_allow_html=True,
                )

                trace = result.get(
                    "trace",
                    [],
                )

                with st.expander(
                    "View agent execution trace",
                    expanded=False,
                ):

                    if trace:

                        for index, step in enumerate(
                            trace,
                            start=1,
                        ):

                            safe_step = html.escape(
                                str(step)
                            )

                            st.markdown(
                                f'<div class="trace-step">'
                                f'<b>{index:02d}</b>&nbsp;&nbsp;'
                                f'✓ {safe_step}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    else:

                        st.info(
                            "No execution trace available."
                        )