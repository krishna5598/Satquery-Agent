from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from main import app as graph_app


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "backend" / "artifacts"

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SatQuery AI API",
    version="1.0.0",
    description="Backend API for SatQuery AI satellite image analysis.",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


# ============================================================
# QUERY DETECTION
# ============================================================

def is_optical_sar_query(query: str) -> bool:
    """
    Detect Optical + SAR requests.
    """

    q = query.lower()

    optical_words = [
        "optical",
        "multispectral",
        "rgb",
    ]

    sar_words = [
        "sar",
        "synthetic aperture radar",
        "radar image",
    ]

    has_optical = any(
        word in q
        for word in optical_words
    )

    has_sar = any(
        word in q
        for word in sar_words
    )

    return has_optical and has_sar


def is_change_query(query: str) -> bool:
    """
    Detect change-analysis requests.
    """

    q = query.lower()

    change_words = [
        "change",
        "changed",
        "changes",
        "difference",
        "differences",
        "before and after",
        "before-after",
        "compare these images",
        "compare the images",
        "compare both images",
        "compare these",
        "what changed",
        "what has changed",
        "change detection",
        "change analysis",
        "detect change",
    ]

    return any(
        word in q
        for word in change_words
    )


# ============================================================
# OPTICAL + SAR IDENTIFICATION
# ============================================================

def identify_optical_sar(
    uploaded_files,
    saved_paths,
):
    """
    Identify Optical and SAR images.

    Priority:
        1. Filename
        2. Remaining image
        3. If exactly two unlabeled files:
           first = Optical
           second = SAR
    """

    optical_path = None
    sar_path = None

    # --------------------------------------------------------
    # Filename detection
    # --------------------------------------------------------

    for upload, path in zip(
        uploaded_files,
        saved_paths,
    ):

        filename = (
            upload.filename or ""
        ).lower()

        if any(
            word in filename
            for word in [
                "optical",
                "rgb",
                "multispectral",
            ]
        ):
            optical_path = str(path)

        elif any(
            word in filename
            for word in [
                "sar",
                "radar",
            ]
        ):
            sar_path = str(path)

    # --------------------------------------------------------
    # Two unlabeled images
    # --------------------------------------------------------

    if (
        optical_path is None
        and sar_path is None
        and len(saved_paths) == 2
    ):

        optical_path = str(
            saved_paths[0]
        )

        sar_path = str(
            saved_paths[1]
        )

    # --------------------------------------------------------
    # One side detected
    # --------------------------------------------------------

    if len(saved_paths) == 2:

        if optical_path and not sar_path:

            remaining = [
                str(path)
                for path in saved_paths
                if str(path) != optical_path
            ]

            if remaining:
                sar_path = remaining[0]

        elif sar_path and not optical_path:

            remaining = [
                str(path)
                for path in saved_paths
                if str(path) != sar_path
            ]

            if remaining:
                optical_path = remaining[0]

    return optical_path, sar_path


# ============================================================
# CHANGE IMAGE IDENTIFICATION
# ============================================================

def identify_change_images(
    uploaded_files,
    saved_paths,
):
    """
    Identify Before and After images.

    Priority:
        1. Filename containing before / pre / t1
        2. Filename containing after / post / t2
        3. Otherwise:
           image 1 = before
           image 2 = after
    """

    before_path = None
    after_path = None

    # --------------------------------------------------------
    # Filename detection
    # --------------------------------------------------------

    for upload, path in zip(
        uploaded_files,
        saved_paths,
    ):

        filename = (
            upload.filename or ""
        ).lower()

        if any(
            word in filename
            for word in [
                "before",
                "pre",
                "prechange",
                "t1",
                "time1",
                "earlier",
                "old",
            ]
        ):
            before_path = str(path)

        elif any(
            word in filename
            for word in [
                "after",
                "post",
                "postchange",
                "t2",
                "time2",
                "later",
                "new",
            ]
        ):
            after_path = str(path)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if (
        len(saved_paths) >= 2
        and before_path is None
        and after_path is None
    ):

        before_path = str(
            saved_paths[0]
        )

        after_path = str(
            saved_paths[1]
        )

    # --------------------------------------------------------
    # If only one side detected
    # --------------------------------------------------------

    if len(saved_paths) >= 2:

        if before_path and not after_path:

            remaining = [
                str(path)
                for path in saved_paths
                if str(path) != before_path
            ]

            if remaining:
                after_path = remaining[0]

        elif after_path and not before_path:

            remaining = [
                str(path)
                for path in saved_paths
                if str(path) != after_path
            ]

            if remaining:
                before_path = remaining[0]

    return before_path, after_path


# ============================================================
# SAVE UPLOAD
# ============================================================

def save_upload(
    upload: UploadFile,
    request_dir: Path,
    index: int,
):
    """
    Save one uploaded image.
    """

    if not upload.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Image {index + 1} "
                "has no filename."
            ),
        )

    extension = Path(
        upload.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type "
                f"for '{upload.filename}'. "
                "Supported: JPG, JPEG, PNG, "
                "TIF, TIFF."
            ),
        )

    input_path = (
        request_dir
        / f"input_{index + 1}{extension}"
    )

    try:

        with input_path.open("wb") as buffer:

            shutil.copyfileobj(
                upload.file,
                buffer,
            )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save "
                f"'{upload.filename}': {exc}"
            ),
        )

    return input_path


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "SatQuery AI",
        "status": "running",
        "version": "1.0.0",
        "multi_image": True,
        "optical_sar": True,
        "change_analysis": True,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "satquery-backend",
        "multi_image": True,
        "change_analysis": True,
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
async def analyze(

    images: list[UploadFile] | None = File(
        default=None
    ),

    # Backward compatibility
    image: UploadFile | None = File(
        default=None
    ),

    query: str = Form(...),

):

    # ========================================================
    # VALIDATE QUERY
    # ========================================================

    clean_query = query.strip()

    if not clean_query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    # ========================================================
    # NORMALIZE UPLOADS
    # ========================================================

    uploaded_files = []

    if images:
        uploaded_files.extend(images)

    if image is not None:
        uploaded_files.append(image)

    uploaded_files = [
        item
        for item in uploaded_files
        if item is not None
        and item.filename
    ]

    if not uploaded_files:

        raise HTTPException(
            status_code=400,
            detail="No image was uploaded.",
        )

    # ========================================================
    # CREATE REQUEST DIRECTORY
    # ========================================================

    request_id = uuid4().hex

    request_dir = (
        ARTIFACTS_DIR
        / request_id
    )

    request_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SAVE ALL IMAGES
    # ========================================================

    saved_paths = []

    for index, upload in enumerate(
        uploaded_files
    ):

        path = save_upload(
            upload,
            request_dir,
            index,
        )

        saved_paths.append(path)

    # ========================================================
    # PRIMARY IMAGE
    # ========================================================

    primary_image_path = (
        saved_paths[0]
    )

    # ========================================================
    # REQUEST TYPE
    # ========================================================

    optical_sar_request = (
        is_optical_sar_query(
            clean_query
        )
    )

    change_request = (
        is_change_query(
            clean_query
        )
    )

    # ========================================================
    # MODALITY PATHS
    # ========================================================

    optical_path = None
    sar_path = None

    before_path = None
    after_path = None

    # ========================================================
    # OPTICAL + SAR
    # ========================================================

    if optical_sar_request:

        if len(saved_paths) < 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Optical + SAR analysis "
                    "requires two images: "
                    "one Optical and one SAR."
                ),
            )

        optical_path, sar_path = (
            identify_optical_sar(
                uploaded_files,
                saved_paths,
            )
        )

        if (
            optical_path is None
            or sar_path is None
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not identify both "
                    "Optical and SAR images."
                ),
            )

    # ========================================================
    # CHANGE ANALYSIS
    # ========================================================

    elif change_request:

        if len(saved_paths) < 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Change analysis requires "
                    "at least two images: "
                    "a BEFORE image and an "
                    "AFTER image."
                ),
            )

        before_path, after_path = (
            identify_change_images(
                uploaded_files,
                saved_paths,
            )
        )

        if (
            before_path is None
            or after_path is None
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not identify the "
                    "before and after images."
                ),
            )

    # ========================================================
    # LOG REQUEST
    # ========================================================

    print("\n" + "=" * 60)
    print("SATQUERY AI BACKEND")
    print("=" * 60)

    print(
        "Request ID :",
        request_id,
    )

    print(
        "Images     :",
        len(saved_paths),
    )

    for index, path in enumerate(
        saved_paths,
        start=1,
    ):

        original_name = (
            uploaded_files[index - 1].filename
        )

        print(
            f"Image {index}   :",
            original_name,
            "->",
            path,
        )

    print(
        "Query      :",
        clean_query,
    )

    if optical_sar_request:

        print(
            "OPTICAL    :",
            optical_path,
        )

        print(
            "SAR        :",
            sar_path,
        )

    if change_request:

        print(
            "BEFORE     :",
            before_path,
        )

        print(
            "AFTER      :",
            after_path,
        )

    print("=" * 60)

    # ========================================================
    # LANGGRAPH INPUT
    # ========================================================

    graph_input = {

        "query": clean_query,

        # Backward compatibility
        "image_path": str(
            primary_image_path
        ),

        # All images
        "image_paths": [
            str(path)
            for path in saved_paths
        ],

        # Request artifacts
        "artifact_dir": str(
            request_dir
        ),

        # Optical + SAR
        "optical_path": optical_path,
        "sar_path": sar_path,

        # Change Analysis
        "before_path": before_path,
        "after_path": after_path,

        "trace": [],
    }

    # ========================================================
    # INVOKE LANGGRAPH
    # ========================================================

    try:

        result = graph_app.invoke(
            graph_input
        )

    except Exception as exc:

        print(
            "\n[LANGGRAPH ERROR]"
        )

        print(exc)

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: {exc}"
            ),
        )

    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = result.get(
        "evidence",
        {},
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    response = {

        "request_id": request_id,

        "query": result.get(
            "query",
            clean_query,
        ),

        "task": result.get(
            "task",
            "unknown",
        ),

        "answer": result.get(
            "answer",
            "No answer generated.",
        ),

        "confidence": result.get(
            "confidence",
            0.0,
        ),

        "image_count": len(
            saved_paths
        ),

        "images": [
            path.name
            for path in saved_paths
        ],

        # Optical + SAR
        "optical_image": (
            Path(optical_path).name
            if optical_path
            else None
        ),

        "sar_image": (
            Path(sar_path).name
            if sar_path
            else None
        ),

        # Change Analysis
        "before_image": (
            Path(before_path).name
            if before_path
            else None
        ),

        "after_image": (
            Path(after_path).name
            if after_path
            else None
        ),

        "evidence": evidence,

        "trace": result.get(
            "trace",
            [],
        ),

        "status": "completed",
    }

    # ========================================================
    # LOG RESULT
    # ========================================================

    print(
        "\n[ANALYSIS COMPLETE]"
    )

    print(
        "Task   :",
        response["task"],
    )

    print(
        "Images :",
        response["image_count"],
    )

    print(
        "Answer :",
        response["answer"],
    )

    print(
        "Trace  :",
        response["trace"],
    )

    print("=" * 60)

    return response


# ============================================================
# SERVE ARTIFACTS
# ============================================================

@app.get(
    "/artifacts/{request_id}/{filename}"
)
def get_artifact(
    request_id: str,
    filename: str,
):

    request_dir = (
        ARTIFACTS_DIR
        / request_id
    )

    if not request_dir.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Request artifact "
                "directory not found."
            ),
        )

    file_path = (
        request_dir
        / filename
    )

    # --------------------------------------------------------
    # SECURITY
    # --------------------------------------------------------

    try:

        file_path.resolve().relative_to(
            request_dir.resolve()
        )

    except ValueError:

        raise HTTPException(
            status_code=403,
            detail="Invalid artifact path.",
        )

    # --------------------------------------------------------
    # EXISTS
    # --------------------------------------------------------

    if (
        not file_path.exists()
        or not file_path.is_file()
    ):

        raise HTTPException(
            status_code=404,
            detail="Artifact not found.",
        )

    # --------------------------------------------------------
    # MEDIA TYPE
    # --------------------------------------------------------

    suffix = (
        file_path.suffix.lower()
    )

    if suffix in {
        ".jpg",
        ".jpeg",
    }:

        media_type = "image/jpeg"

    elif suffix == ".png":

        media_type = "image/png"

    elif suffix in {
        ".tif",
        ".tiff",
    }:

        media_type = "image/tiff"

    else:

        media_type = (
            "application/octet-stream"
        )

    return FileResponse(
        path=file_path,
        media_type=media_type,
    )