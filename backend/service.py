from pathlib import Path
import sys


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Import existing LangGraph application
# ---------------------------------------------------------

from main import graph_app


# ---------------------------------------------------------
# Analysis service
# ---------------------------------------------------------

def analyze_image(
    image_path: str,
    query: str,
    output_dir: str,
):
    """
    Runs the complete SatQuery AI pipeline.

    Frontend
        ↓
    FastAPI
        ↓
    LangGraph
        ↓
    Router
        ↓
    CLIP + FAISS
        ↓
    Specialist
        ↓
    Evidence
        ↓
    Ollama
    """

    state = {
        "query": query,
        "image_path": image_path,
        "output_dir": output_dir,
        "trace": [],
    }

    result = graph_app.invoke(state)

    return result