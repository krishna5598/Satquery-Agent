def route_query(state):
    """
    SatQuery AI query router.

    Supported tasks:
        - optical_sar
        - change
        - caption
        - grounding
        - vqa
    """

    query = state["query"].lower().strip()

    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    image_paths = state.get(
        "image_paths",
        [],
    )

    image_count = len(image_paths)

    # ========================================================
    # 1. OPTICAL + SAR
    # ========================================================
    #
    # IMPORTANT:
    # This MUST come before change detection.
    #
    # Example:
    # "Compare optical and SAR images"
    # "Analyze SAR and optical data"
    # "What does the optical + SAR fusion show?"
    #
    # ========================================================

    optical_terms = [
        "optical",
        "optical image",
        "optical imagery",
        "multispectral",
        "rgb",
    ]

    sar_terms = [
        "sar",
        "sar image",
        "sar imagery",
        "synthetic aperture radar",
        "radar image",
    ]

    has_optical = any(
        term in query
        for term in optical_terms
    )

    has_sar = any(
        term in query
        for term in sar_terms
    )

    if has_optical and has_sar:

        task = "optical_sar"

    # ========================================================
    # 2. CHANGE DETECTION
    # ========================================================

    elif any(
        phrase in query
        for phrase in [
            "changed",
            "change",
            "difference",
            "differences",
            "before and after",
            "compare these images",
            "compare the images",
            "compare both images",
            "between these images",
            "between the images",
            "what changed",
            "what has changed",
        ]
    ):

        task = "change"

    # ========================================================
    # 3. CAPTIONING
    # ========================================================

    elif any(
        phrase in query
        for phrase in [
            "caption",
            "generate a caption",
            "give me a caption",
            "describe the image",
            "describe this image",
            "describe the satellite image",
            "describe this satellite image",
            "describe the scene",
            "describe these images",
            "describe the images",
        ]
    ):

        task = "caption"

    # ========================================================
    # 4. GROUNDING
    # ========================================================

    elif any(
        phrase in query
        for phrase in [
            "where",
            "locate",
            "location",
            "highlight",
            "identify the location",
            "show me where",
            "where is",
            "where are",
            "find the location",
            "find where",
            "point out",
            "mark the location",
        ]
    ):

        task = "grounding"

    # ========================================================
    # 5. GENERAL VQA
    # ========================================================

    else:

        task = "vqa"

    # ========================================================
    # TRACE
    # ========================================================

    trace_message = (
        f"route={task}"
    )

    if image_count:

        trace_message += (
            f"|images={image_count}"
        )

    # ========================================================
    # RETURN ROUTING RESULT
    # ========================================================

    return {

        "task": task,

        "selected_tools": [
            task
        ],

        "trace": [
            trace_message
        ],
    }