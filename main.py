from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from router import route_query

from retrieval.retriever import retrieve

from tools import (
    vqa_tool,
    caption_tool,
    grounding_tool,
    change_tool,
    optical_sar_tool,
)

from llm import generate_answer


# ============================================================
# STATE
# ============================================================

class State(TypedDict, total=False):

    # --------------------------------------------------------
    # User request
    # --------------------------------------------------------

    query: str

    # --------------------------------------------------------
    # Image inputs
    # --------------------------------------------------------

    # Backward-compatible primary image
    image_path: str

    # NEW: all uploaded images
    image_paths: list[str]

    # --------------------------------------------------------
    # Optical + SAR
    # --------------------------------------------------------

    optical_path: str
    sar_path: str

    # --------------------------------------------------------
    # Request artifacts
    # --------------------------------------------------------

    artifact_dir: str

    # --------------------------------------------------------
    # Agent routing
    # --------------------------------------------------------

    task: str
    selected_tools: list[str]

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved_patches: list

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    evidence: dict
    answer: str
    confidence: float

    # --------------------------------------------------------
    # Debug trace
    # --------------------------------------------------------

    trace: list[str]


# ============================================================
# RETRIEVER NODE
# ============================================================

def retrieve_node(state: State):

    query = state["query"]

    task = state.get(
        "task",
        "vqa",
    )

    image_paths = state.get(
        "image_paths",
        [],
    )

    # --------------------------------------------------------
    # Backward compatibility
    # --------------------------------------------------------

    if not image_paths:

        image_path = state.get(
            "image_path"
        )

        if image_path:
            image_paths = [
                image_path
            ]

    # --------------------------------------------------------
    # Validate images
    # --------------------------------------------------------

    if not image_paths:

        print(
            "\n[RETRIEVER]"
        )

        print(
            "No images available."
        )

        return {
            "retrieved_patches": [],
            "trace": state.get(
                "trace",
                []
            ) + [
                "retriever_skipped_no_images"
            ],
        }

    # ========================================================
    # OPTICAL + SAR
    # ========================================================
    #
    # IMPORTANT:
    # Optical + SAR has its own preprocessing/model.
    #
    # Do NOT send SAR through the existing RGB CLIP
    # retrieval pipeline.
    #
    # ========================================================

    if task == "optical_sar":

        print(
            "\n[RETRIEVER]"
        )

        print(
            "Skipped for Optical + SAR specialist."
        )

        return {
            "retrieved_patches": [],
            "trace": state.get(
                "trace",
                []
            ) + [
                "retriever_skipped_optical_sar"
            ],
        }

    # ========================================================
    # CHANGE ANALYSIS
    # ========================================================
    #
    # Change analysis works on multiple images.
    # Do not force the complete multi-image request through
    # the single-image CLIP retriever.
    #
    # The change specialist will receive all images directly.
    #
    # ========================================================

    if task == "change" and len(image_paths) >= 2:

        print(
            "\n[RETRIEVER]"
        )

        print(
            "Skipped for multi-image change analysis."
        )

        return {
            "retrieved_patches": [],
            "trace": state.get(
                "trace",
                []
            ) + [
                "retriever_skipped_change_analysis"
            ],
        }

    # ========================================================
    # STANDARD SINGLE-IMAGE RETRIEVAL
    # ========================================================

    image_path = image_paths[0]

    print(
        "\n[RETRIEVER]"
    )

    print(
        "Query:",
        query,
    )

    print(
        "Image:",
        image_path,
    )

    # --------------------------------------------------------
    # Existing CLIP + FAISS retrieval
    # --------------------------------------------------------

    _, patches = retrieve(
        image_path=image_path,
        query=query,
        top_k=4,
    )

    print(
        f"Retrieved {len(patches)} "
        "relevant patches"
    )

    for patch in patches:

        print(
            f"  Patch {patch['id']} "
            f"score={patch['score']:.4f} "
            f"bbox={patch['bbox']}"
        )

    return {
        "retrieved_patches": patches,

        "trace": state.get(
            "trace",
            []
        ) + [
            f"retrieved_top_{len(patches)}_patches"
        ],
    }


# ============================================================
# SPECIALIST NODE
# ============================================================

def run_specialist(state: State):

    task = state["task"]

    tools = {

        "vqa": vqa_tool,

        "caption": caption_tool,

        "grounding": grounding_tool,

        "change": change_tool,

        "optical_sar": optical_sar_tool,
    }

    # --------------------------------------------------------
    # Validate task
    # --------------------------------------------------------

    if task not in tools:

        raise ValueError(
            f"Unknown specialist task: {task}"
        )

    tool = tools[task]

    print(
        "\n[SPECIALIST]"
    )

    print(
        "Selected:",
        task,
    )

    # --------------------------------------------------------
    # Run specialist
    # --------------------------------------------------------

    result = tool(state)

    return {

        "evidence": result,

        "confidence": result.get(
            "confidence",
            0.0,
        ),

        "trace": state.get(
            "trace",
            []
        ) + [
            f"tool={task}"
        ],
    }


# ============================================================
# AGGREGATE EVIDENCE
# ============================================================

def aggregate_evidence(state: State):

    evidence = state["evidence"]

    aggregated = {

        "tool": evidence.get(
            "tool",
            state["task"],
        ),

        "status": evidence.get(
            "status",
            "completed",
        ),

        "answer": evidence.get(
            "answer",
            "",
        ),

        "confidence": evidence.get(
            "confidence",
            0.0,
        ),

        "evidence": evidence.get(
            "evidence",
            {},
        ),
    }

    return {

        "evidence": aggregated,

        "confidence": evidence.get(
            "confidence",
            0.0,
        ),

        "trace": state.get(
            "trace",
            []
        ) + [
            "evidence_aggregated"
        ],
    }


# ============================================================
# FINAL ANSWER
# ============================================================

def finish(state: State):

    answer = generate_answer(
        state["query"],
        state["evidence"],
    )

    return {

        "answer": answer,

        "trace": state.get(
            "trace",
            []
        ) + [
            "final_llm_answer_generated"
        ],
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

graph.add_node(
    "router",
    route_query,
)

graph.add_node(
    "retriever",
    retrieve_node,
)

graph.add_node(
    "specialist",
    run_specialist,
)

graph.add_node(
    "aggregate",
    aggregate_evidence,
)

graph.add_node(
    "finish",
    finish,
)


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

graph.add_edge(
    START,
    "router",
)

graph.add_edge(
    "router",
    "retriever",
)

graph.add_edge(
    "retriever",
    "specialist",
)

graph.add_edge(
    "specialist",
    "aggregate",
)

graph.add_edge(
    "aggregate",
    "finish",
)

graph.add_edge(
    "finish",
    END,
)


# ============================================================
# COMPILE
# ============================================================

# IMPORTANT:
# backend.main imports this variable.

app = graph.compile()


# ============================================================
# LOCAL TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example single image
    # --------------------------------------------------------

    image_path = (
        "data/images/satellite.jpg"
    )

    image_paths = [
        image_path
    ]

    # --------------------------------------------------------
    # User query
    # --------------------------------------------------------

    query = input(
        "\nEnter your satellite query: "
    ).strip()

    if not query:

        query = (
            "What is the category of "
            "this remote sensing image?"
        )

    # --------------------------------------------------------
    # Invoke graph
    # --------------------------------------------------------

    result = app.invoke(
        {

            "query": query,

            # Backward compatible
            "image_path": image_path,

            # NEW
            "image_paths": image_paths,

            "artifact_dir": (
                "backend/artifacts/test"
            ),

            "trace": [],
        }
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "QUERY:"
    )

    print(
        result["query"]
    )

    print(
        "\nTASK:"
    )

    print(
        result.get(
            "task",
            "unknown",
        )
    )

    print(
        "\nIMAGE PATHS:"
    )

    for path in result.get(
        "image_paths",
        [],
    ):

        print(
            " -",
            path,
        )

    print(
        "\nRETRIEVED PATCHES:"
    )

    for patch in result.get(
        "retrieved_patches",
        [],
    ):

        print(
            f"Patch {patch['id']} | "
            f"score={patch['score']:.4f} | "
            f"bbox={patch['bbox']}"
        )

    print(
        "\nFINAL ANSWER:"
    )

    print(
        result.get(
            "answer",
            "",
        )
    )

    print(
        "\nCONFIDENCE:"
    )

    print(
        result.get(
            "confidence",
            0.0,
        )
    )

    print(
        "\nTRACE:"
    )

    print(
        result.get(
            "trace",
            [],
        )
    )

    print(
        "\nEVIDENCE:"
    )

    print(
        result.get(
            "evidence",
            {},
        )
    )

    print(
        "=" * 60
    )