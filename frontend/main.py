from pathlib import Path

import streamlit as st


# ============================================================
# PATH CONFIGURATION
# ============================================================

FRONTEND_DIR = Path(__file__).resolve().parent

HOME_PAGE = FRONTEND_DIR / "app.py"

AI_ANALYSIS_PAGE = (
    FRONTEND_DIR
    / "pages"
    / "1_AI_Analysis.py"
)

ABOUT_PAGE = (
    FRONTEND_DIR
    / "pages"
    / "2_About_SatQuery.py"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# GLOBAL STREAMLIT CLEANUP
# ============================================================

st.markdown(
    """
    <style>

    /* Hide Streamlit header */
    header[data-testid="stHeader"] {
        display: none !important;
    }


    /* Hide decoration */
    [data-testid="stDecoration"] {
        display: none !important;
    }


    /* Hide menu */
    #MainMenu {
        visibility: hidden !important;
    }


    /* Hide footer */
    footer {
        visibility: hidden !important;
    }


    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }


    /* Main content spacing */
    .block-container {

        padding-top:
            1.2rem !important;

        padding-bottom:
            3rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VERIFY PAGE FILES
# ============================================================

missing_pages = []


if not HOME_PAGE.exists():

    missing_pages.append(
        str(HOME_PAGE)
    )


if not AI_ANALYSIS_PAGE.exists():

    missing_pages.append(
        str(AI_ANALYSIS_PAGE)
    )


if not ABOUT_PAGE.exists():

    missing_pages.append(
        str(ABOUT_PAGE)
    )


if missing_pages:

    st.error(
        "SatQuery AI frontend pages could not be found."
    )

    st.write(
        "The following files are missing:"
    )

    for page in missing_pages:

        st.code(page)

    st.info(
        "Make sure the frontend folder contains "
        "app.py and the pages folder contains "
        "1_AI_Analysis.py and 2_About_SatQuery.py."
    )

    st.stop()


# ============================================================
# PAGE DEFINITIONS
# ============================================================

home_page = st.Page(
    str(HOME_PAGE),
    title="Home",
    icon="🏠",
    default=True,
)


ai_analysis_page = st.Page(
    str(AI_ANALYSIS_PAGE),
    title="AI Analysis",
    icon="🤖",
)


about_page = st.Page(
    str(ABOUT_PAGE),
    title="About SatQuery",
    icon="📖",
)


# ============================================================
# STREAMLIT INTERNAL NAVIGATION
# ============================================================

pg = st.navigation(
    {
        "SatQuery AI": [
            home_page,
            ai_analysis_page,
            about_page,
        ]
    },
    position="sidebar",
)


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()