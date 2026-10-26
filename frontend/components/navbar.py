import streamlit as st


def render_navbar():
    """
    SatQuery AI compact navigation bar.

    Uses Streamlit-native components for all visible content.
    No HTML is used for the navbar content, preventing raw
    HTML from appearing in the UI.
    """

    # ============================================================
    # NAVBAR STYLE
    # ============================================================

    st.markdown(
        """
        <style>

        /* ========================================================
           STREAMLIT CLEANUP
           ======================================================== */

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


        /* ========================================================
           COMPACT NAVBAR
           ======================================================== */

        div[data-testid="stHorizontalBlock"]:has(
            [data-testid="stPageLink"]
        ) {

            position: fixed !important;

            top: 0 !important;
            left: 0 !important;
            right: 0 !important;

            width: 100% !important;

            height: 58px !important;

            z-index: 999999 !important;

            display: flex !important;

            align-items: center !important;

            padding:
                0 42px !important;

            box-sizing: border-box !important;

            background:
                rgba(255, 255, 255, 0.97) !important;

            border-bottom:
                1px solid #e5e7eb !important;

            box-shadow:
                0 2px 10px
                rgba(15, 23, 42, 0.05) !important;

            backdrop-filter:
                blur(12px) !important;

            -webkit-backdrop-filter:
                blur(12px) !important;

            margin:
                0 !important;
        }


        /* ========================================================
           COLUMN RESET
           ======================================================== */

        div[data-testid="stHorizontalBlock"]:has(
            [data-testid="stPageLink"]
        ) > div {

            padding:
                0 !important;

            margin:
                0 !important;

            display:
                flex !important;

            align-items:
                center !important;

            justify-content:
                center !important;
        }


        /* ========================================================
           BRAND COLUMN
           ======================================================== */

        div[data-testid="stHorizontalBlock"]:has(
            [data-testid="stPageLink"]
        ) > div:first-child {

            justify-content:
                flex-start !important;
        }


        /* ========================================================
           BRAND
           ======================================================== */

        .sq-brand-native {

            display:
                flex;

            align-items:
                center;

            gap:
                8px;

            white-space:
                nowrap;

            font-size:
                15px;

            font-weight:
                800;

            color:
                #172554;
        }


        .sq-brand-native-icon {

            width:
                29px;

            height:
                29px;

            display:
                flex;

            align-items:
                center;

            justify-content:
                center;

            border-radius:
                8px;

            background:
                linear-gradient(
                    135deg,
                    #3155d9,
                    #0789c5
                );

            font-size:
                14px;

            box-shadow:
                0 3px 8px
                rgba(49, 85, 217, 0.18);
        }


        /* ========================================================
           PAGE LINKS
           ======================================================== */

        div[data-testid="stHorizontalBlock"]:has(
            [data-testid="stPageLink"]
        )
        [data-testid="stPageLink"] {

            width:
                auto !important;

            margin:
                0 !important;

            padding:
                0 !important;

            display:
                flex !important;

            align-items:
                center !important;

            justify-content:
                center !important;
        }


        div[data-testid="stHorizontalBlock"]:has(
            [data-testid="stPageLink"]
        )
        [data-testid="stPageLink"] a {

            min-height:
                34px !important;

            padding:
                5px 13px !important;

            border-radius:
                5px !important;

            color:
                #0f172a !important;

            font-size:
                12px !important;

            font-weight:
                600 !important;

            text-decoration:
                none !important;

            background:
                transparent !important;

            transition:
                background 0.15s ease,
                color 0.15s ease !important;
        }


        /* ========================================================
           LINK HOVER
           ======================================================== */

        div[data-testid="stHorizontalBlock"]:has(
            [data-testid="stPageLink"]
        )
        [data-testid="stPageLink"] a:hover {

            background:
                #f1f5f9 !important;

            color:
                #1d4ed8 !important;
        }


        /* ========================================================
           STATUS
           ======================================================== */

        .sq-status-native {

            display:
                flex;

            align-items:
                center;

            justify-content:
                center;

            gap:
                6px;

            white-space:
                nowrap;

            font-size:
                9px;

            font-weight:
                700;

            color:
                #15803d;

            padding:
                5px 10px;

            border:
                1px solid #bbf7d0;

            border-radius:
                999px;

            background:
                #f0fdf4;
        }


        .sq-status-native-dot {

            width:
                6px;

            height:
                6px;

            border-radius:
                50%;

            background:
                #22c55e;
        }


        /* ========================================================
           NAVBAR SPACE
           ======================================================== */

        .sq-navbar-space {

            width:
                100%;

            height:
                70px;
        }


        /* ========================================================
           TABLET
           ======================================================== */

        @media (max-width: 900px) {

            div[data-testid="stHorizontalBlock"]:has(
                [data-testid="stPageLink"]
            ) {

                height:
                    54px !important;

                padding:
                    0 20px !important;
            }


            .sq-brand-native {

                font-size:
                    14px;
            }


            .sq-brand-native-icon {

                width:
                    27px;

                height:
                    27px;

                font-size:
                    13px;
            }


            div[data-testid="stHorizontalBlock"]:has(
                [data-testid="stPageLink"]
            )
            [data-testid="stPageLink"] a {

                padding:
                    5px 8px !important;

                font-size:
                    10px !important;
            }


            .sq-status-native {

                padding:
                    4px 7px;

                font-size:
                    8px;
            }


            .sq-navbar-space {

                height:
                    64px;
            }
        }


        /* ========================================================
           MOBILE
           ======================================================== */

        @media (max-width: 600px) {

            div[data-testid="stHorizontalBlock"]:has(
                [data-testid="stPageLink"]
            ) {

                height:
                    50px !important;

                padding:
                    0 10px !important;
            }


            .sq-brand-native {

                font-size:
                    12px;
            }


            .sq-brand-native-icon {

                width:
                    25px;

                height:
                    25px;

                font-size:
                    12px;
            }


            div[data-testid="stHorizontalBlock"]:has(
                [data-testid="stPageLink"]
            )
            [data-testid="stPageLink"] a {

                padding:
                    4px 5px !important;

                font-size:
                    9px !important;
            }


            .sq-status-native {

                display:
                    none !important;
            }


            .sq-navbar-space {

                height:
                    60px;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


    # ============================================================
    # NAVBAR COLUMNS
    # ============================================================

    col_brand, col_home, col_analysis, col_about, col_status = st.columns(
        [3.2, 1.0, 1.35, 1.0, 1.25],
        gap="small",
        vertical_alignment="center",
    )


    # ============================================================
    # BRAND
    # ============================================================

    with col_brand:

        st.markdown(
            """
            <div class="sq-brand-native">
                <span class="sq-brand-native-icon">🛰️</span>
                <span>SatQuery AI</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ============================================================
    # HOME
    # ============================================================

    with col_home:

        st.page_link(
            "app.py",
            label="🏠 Home",
        )


    # ============================================================
    # AI ANALYSIS
    # ============================================================

    with col_analysis:

        st.page_link(
            "pages/1_AI_Analysis.py",
            label="🤖 AI Analysis",
        )


    # ============================================================
    # ABOUT
    # ============================================================

    with col_about:

        st.page_link(
            "pages/2_About_SatQuery.py",
            label="📖 About",
        )


    # ============================================================
    # SYSTEM STATUS
    # ============================================================

    with col_status:

        st.markdown(
            """
            <div class="sq-status-native">
                <span class="sq-status-native-dot"></span>
                <span>System Ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ============================================================
    # CONTENT OFFSET
    # ============================================================

    st.markdown(
        """
        <div class="sq-navbar-space"></div>
        """,
        unsafe_allow_html=True,
    )