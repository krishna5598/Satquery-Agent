from pathlib import Path
import shutil
import sys

from retrieval.retriever import make_contact_sheet

from vqa_model import ask_vqa, generate_caption

from grounding import ground_patches


# ============================================================
# OPTICAL + SAR MODULE
# ============================================================

# Make the local optical_sar directory importable.
BASE_DIR = Path(__file__).resolve().parent

OPTICAL_SAR_DIR = BASE_DIR / "optical_sar"

if str(OPTICAL_SAR_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(OPTICAL_SAR_DIR),
    )


# These imports come from your friend's module.
try:

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.colors import ListedColormap

    import rasterio

    from model import DualBranchFusion
    from preprocessing import prepare_tensors
    from validator import validate_optical_sar_pair

    OPTICAL_SAR_AVAILABLE = True

except Exception as exc:

    OPTICAL_SAR_AVAILABLE = False

    OPTICAL_SAR_IMPORT_ERROR = str(exc)


# Cached model.
_FUSION_MODEL = None


# ============================================================
# ARTIFACT HELPERS
# ============================================================

def get_artifact_dir(state):
    """
    Get the artifact directory for the current request.
    """

    artifact_dir = state.get(
        "artifact_dir"
    )

    if artifact_dir:

        path = Path(
            artifact_dir
        )

    else:

        path = (
            BASE_DIR
            / "backend"
            / "artifacts"
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def patch_evidence(patches):
    """
    Convert retrieved patch information into
    JSON-serializable evidence.
    """

    return [

        {
            "id": patch["id"],
            "bbox": patch["bbox"],
            "score": patch["score"],
        }

        for patch in patches
    ]


def get_artifact_url(
    state,
    filename,
):
    """
    Generate API URL for a request artifact.
    """

    artifact_dir = Path(
        state["artifact_dir"]
    )

    request_id = artifact_dir.name

    return (
        f"/artifacts/"
        f"{request_id}/"
        f"{filename}"
    )


# ============================================================
# MULTI-IMAGE HELPERS
# ============================================================

def get_image_paths(state):
    """
    Return all uploaded image paths.

    Backward compatible with the old image_path field.
    """

    image_paths = state.get(
        "image_paths",
        [],
    )

    if image_paths:

        return [
            str(path)
            for path in image_paths
            if path
        ]

    image_path = state.get(
        "image_path"
    )

    if image_path:

        return [
            str(image_path)
        ]

    return []


def get_primary_image(state):
    """
    Return the first available image.
    """

    image_paths = get_image_paths(
        state
    )

    if not image_paths:

        return None

    return image_paths[0]


def make_multi_image_contact_sheet(
    image_paths,
    output_path,
):
    """
    Create a simple contact sheet for multiple uploaded images.

    This is used as frontend evidence when several images
    are uploaded.
    """

    from PIL import Image, ImageDraw

    if not image_paths:

        return None

    images = []

    for path in image_paths:

        try:

            image = (
                Image.open(path)
                .convert("RGB")
            )

            image.thumbnail(
                (512, 512)
            )

            images.append(
                (
                    Path(path).name,
                    image.copy(),
                )
            )

        except Exception as exc:

            print(
                "[MULTI IMAGE SHEET] "
                f"Could not open {path}: {exc}"
            )

    if not images:

        return None

    margin = 20
    label_height = 35
    columns = min(
        2,
        len(images),
    )

    rows = (
        len(images)
        + columns
        - 1
    ) // columns

    cell_width = 512
    cell_height = (
        512
        + label_height
    )

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width
            + (columns + 1) * margin,
            rows * cell_height
            + (rows + 1) * margin,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    for index, (
        filename,
        image,
    ) in enumerate(images):

        row = index // columns
        col = index % columns

        x = (
            margin
            + col * cell_width
        )

        y = (
            margin
            + row * cell_height
        )

        sheet.paste(
            image,
            (x, y),
        )

        draw.text(
            (
                x,
                y + 515,
            ),
            filename,
            fill="black",
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=95,
    )

    return str(
        output_path
    )


# ============================================================
# VQA TOOL
# ============================================================

def vqa_tool(state):
    """
    Visual Question Answering.

    Supports one or multiple uploaded images.

    For multiple images:
        each image is analyzed independently,
        then the individual answers are combined.
    """

    patches = state.get(
        "retrieved_patches",
        [],
    )

    image_paths = get_image_paths(
        state
    )

    if not image_paths:

        return {
            "tool": "vqa",
            "status": "failed",
            "answer": (
                "No image was uploaded."
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    artifact_dir = get_artifact_dir(
        state
    )

    # --------------------------------------------------------
    # Retrieved patch evidence
    # --------------------------------------------------------

    evidence = {}

    if patches:

        retrieved_path = (
            artifact_dir
            / "retrieved_patches.jpg"
        )

        make_contact_sheet(
            patches,
            output_path=retrieved_path,
        )

        evidence[
            "patches"
        ] = patch_evidence(
            patches
        )

        evidence[
            "retrieved_image"
        ] = get_artifact_url(
            state,
            "retrieved_patches.jpg",
        )

    # --------------------------------------------------------
    # Multiple image contact sheet
    # --------------------------------------------------------

    if len(image_paths) > 1:

        multi_path = (
            artifact_dir
            / "uploaded_images.jpg"
        )

        make_multi_image_contact_sheet(
            image_paths,
            multi_path,
        )

        evidence[
            "uploaded_images"
        ] = get_artifact_url(
            state,
            "uploaded_images.jpg",
        )

    # --------------------------------------------------------
    # Ask VQA model about every image
    # --------------------------------------------------------

    answers = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        try:

            answer = ask_vqa(
                image_path,
                state["query"],
            )

            answer = str(
                answer
            ).strip()

            if answer:

                if len(image_paths) > 1:

                    answers.append(
                        f"Image {index}: {answer}"
                    )

                else:

                    answers.append(
                        answer
                    )

        except Exception as exc:

            print(
                "[VQA ERROR]",
                image_path,
                exc,
            )

            answers.append(
                f"Image {index}: "
                "analysis failed."
            )

    if not answers:

        return {
            "tool": "vqa",
            "status": "failed",
            "answer": (
                "The uploaded images "
                "could not be analyzed."
            ),
            "confidence": 0.0,
            "evidence": evidence,
        }

    # --------------------------------------------------------
    # Final VQA answer
    # --------------------------------------------------------

    if len(answers) == 1:

        final_answer = answers[0]

    else:

        final_answer = (
            "Analysis of the uploaded images:\n"
            + "\n".join(answers)
        )

    return {
        "tool": "vqa",
        "status": "completed",
        "answer": final_answer,
        "confidence": 0.0,
        "evidence": evidence,
    }


# ============================================================
# CAPTION TOOL
# ============================================================

def caption_tool(state):
    """
    Image captioning.

    Supports one or multiple uploaded images.
    """

    patches = state.get(
        "retrieved_patches",
        [],
    )

    image_paths = get_image_paths(
        state
    )

    if not image_paths:

        return {
            "tool": "caption",
            "status": "failed",
            "answer": (
                "No image was uploaded."
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    artifact_dir = get_artifact_dir(
        state
    )

    evidence = {}

    # --------------------------------------------------------
    # Retrieved patches
    # --------------------------------------------------------

    if patches:

        retrieved_path = (
            artifact_dir
            / "retrieved_patches.jpg"
        )

        make_contact_sheet(
            patches,
            output_path=retrieved_path,
        )

        evidence[
            "patches"
        ] = patch_evidence(
            patches
        )

        evidence[
            "retrieved_image"
        ] = get_artifact_url(
            state,
            "retrieved_patches.jpg",
        )

    # --------------------------------------------------------
    # Multiple image sheet
    # --------------------------------------------------------

    if len(image_paths) > 1:

        multi_path = (
            artifact_dir
            / "uploaded_images.jpg"
        )

        make_multi_image_contact_sheet(
            image_paths,
            multi_path,
        )

        evidence[
            "uploaded_images"
        ] = get_artifact_url(
            state,
            "uploaded_images.jpg",
        )

    # --------------------------------------------------------
    # Generate caption for every image
    # --------------------------------------------------------

    captions = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        try:

            caption = generate_caption(
                image_path
            )

            caption = str(
                caption
            ).strip()

            if caption:

                if len(image_paths) > 1:

                    captions.append(
                        f"Image {index}: {caption}"
                    )

                else:

                    captions.append(
                        caption
                    )

        except Exception as exc:

            print(
                "[CAPTION ERROR]",
                image_path,
                exc,
            )

            captions.append(
                f"Image {index}: "
                "caption generation failed."
            )

    if not captions:

        return {
            "tool": "caption",
            "status": "failed",
            "answer": (
                "The uploaded images "
                "could not be captioned."
            ),
            "confidence": 0.0,
            "evidence": evidence,
        }

    if len(captions) == 1:

        final_answer = captions[0]

    else:

        final_answer = (
            "Captions for the uploaded images:\n"
            + "\n".join(captions)
        )

    return {
        "tool": "caption",
        "status": "completed",
        "answer": final_answer,
        "confidence": 0.0,
        "evidence": evidence,
    }


# ============================================================
# GROUNDING TOOL
# ============================================================

def grounding_tool(state):
    """
    Visual grounding.

    For multiple images, grounding is performed independently
    for each image.
    """

    patches = state.get(
        "retrieved_patches",
        [],
    )

    image_paths = get_image_paths(
        state
    )

    if not image_paths:

        return {
            "tool": "grounding",
            "status": "failed",
            "answer": (
                "No image was uploaded."
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    artifact_dir = get_artifact_dir(
        state
    )

    evidence = {}

    # --------------------------------------------------------
    # Retrieved patches
    # --------------------------------------------------------

    if patches:

        retrieved_path = (
            artifact_dir
            / "retrieved_patches.jpg"
        )

        make_contact_sheet(
            patches,
            output_path=retrieved_path,
        )

        evidence[
            "patches"
        ] = patch_evidence(
            patches
        )

        evidence[
            "retrieved_image"
        ] = get_artifact_url(
            state,
            "retrieved_patches.jpg",
        )

    # --------------------------------------------------------
    # Ground each uploaded image
    # --------------------------------------------------------

    grounded_images = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        output_name = (
            "grounded_result.jpg"
            if len(image_paths) == 1
            else f"grounded_result_{index}.jpg"
        )

        grounded_path = (
            artifact_dir
            / output_name
        )

        try:

            # For the first image use existing retrieved
            # patches. For additional images, retrieve
            # patches specifically for that image.
            current_patches = patches

            if index > 1:

                try:

                    from retrieval.retriever import (
                        retrieve
                    )

                    _, current_patches = retrieve(
                        image_path=image_path,
                        query=state["query"],
                        top_k=4,
                    )

                except Exception as exc:

                    print(
                        "[GROUNDING RETRIEVAL ERROR]",
                        exc,
                    )

                    current_patches = []

            if not current_patches:

                continue

            ground_patches(
                image_path,
                current_patches,
                output_path=grounded_path,
            )

            grounded_images.append(
                get_artifact_url(
                    state,
                    output_name,
                )
            )

        except Exception as exc:

            print(
                "[GROUNDING ERROR]",
                image_path,
                exc,
            )

    if not grounded_images:

        return {
            "tool": "grounding",
            "status": "failed",
            "answer": (
                "No relevant image regions "
                "could be grounded."
            ),
            "confidence": 0.0,
            "evidence": evidence,
        }

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence[
        "grounded_images"
    ] = grounded_images

    # Backward-compatible single image field
    if len(grounded_images) == 1:

        evidence[
            "grounded_image"
        ] = grounded_images[0]

    return {
        "tool": "grounding",
        "status": "completed",
        "answer": (
            "The relevant image regions "
            "have been identified."
        ),
        "confidence": 0.0,
        "evidence": evidence,
    }


# ============================================================
# CHANGE DETECTION TOOL
# ============================================================

# Real Change Analysis specialist from the supplied
# SatQuery AI Change Analysis package.
try:
    from change_analysis.tools.change_tool import (
        change_tool as run_change_analysis,
    )

    CHANGE_ANALYSIS_AVAILABLE = True
    CHANGE_ANALYSIS_IMPORT_ERROR = ""

except Exception as exc:
    CHANGE_ANALYSIS_AVAILABLE = False
    CHANGE_ANALYSIS_IMPORT_ERROR = str(exc)


_CHANGE_ANALYSIS_PIPELINE = None


def get_change_analysis_pipeline():
    """Create the supplied Change Analysis pipeline once and reuse it."""
    global _CHANGE_ANALYSIS_PIPELINE

    if not CHANGE_ANALYSIS_AVAILABLE:
        raise RuntimeError(
            "Change Analysis package could not be imported: "
            + CHANGE_ANALYSIS_IMPORT_ERROR
        )

    if _CHANGE_ANALYSIS_PIPELINE is None:
        from change_analysis.pipeline.change_pipeline import ChangePipeline

        _CHANGE_ANALYSIS_PIPELINE = ChangePipeline()

    return _CHANGE_ANALYSIS_PIPELINE


def identify_change_pair(image_paths):
    """
    Identify BEFORE and AFTER images.

    Priority:
      1. Filename containing before/pre/t1 -> BEFORE
      2. Filename containing after/post/t2 -> AFTER
      3. Otherwise first uploaded image = BEFORE,
         second uploaded image = AFTER.
    """
    paths = [Path(p) for p in image_paths if p]

    if len(paths) < 2:
        return None, None

    before = None
    after = None

    before_words = (
        "before",
        "pre",
        "t1",
        "earlier",
        "old",
    )
    after_words = (
        "after",
        "post",
        "t2",
        "later",
        "new",
    )

    for path in paths:
        name = path.stem.lower()

        if before is None and any(word in name for word in before_words):
            before = path

        if after is None and any(word in name for word in after_words):
            after = path

    # If both are explicitly identified, use them.
    if before is not None and after is not None and before != after:
        return before, after

    # If only one side was identified, use the remaining image for the other side.
    if before is not None and after is None:
        remaining = [p for p in paths if p != before]
        if remaining:
            return before, remaining[0]

    if after is not None and before is None:
        remaining = [p for p in paths if p != after]
        if remaining:
            return remaining[0], after

    # Default: upload order.
    return paths[0], paths[1]


def _copy_change_artifacts(state, pipeline_result):
    """Copy package artifacts into the current SatQuery request directory."""
    artifact_dir = get_artifact_dir(state)
    evidence = {}

    source_evidence = pipeline_result.get("evidence", [])

    # The supplied pipeline returns a list of typed artifact records.
    for item in source_evidence:
        if not isinstance(item, dict):
            continue

        source = item.get("path")
        artifact_type = item.get("type", "artifact")
        description = item.get("description", "")

        if not source:
            continue

        source_path = Path(source)

        if not source_path.exists():
            continue

        # Keep unique, frontend-friendly filenames inside the request folder.
        destination_name = source_path.name
        destination = artifact_dir / destination_name

        try:
            shutil.copy2(source_path, destination)
        except Exception as exc:
            print(
                "[CHANGE ARTIFACT COPY ERROR]",
                source_path,
                exc,
            )
            continue

        evidence[artifact_type] = {
            "url": get_artifact_url(state, destination_name),
            "filename": destination_name,
            "description": description,
        }

    return evidence


def change_tool(state):
    """
    Real bi-temporal Change Analysis specialist.

    Uses the supplied change_analysis package through its
    standardized change_tool()/ChangePipeline interface.

    Required state:
        state["image_paths"] -> at least two images
        state["query"]       -> natural-language question

    Produces:
        - change map
        - heatmap
        - overlay
        - comparison figure
        - changed-area metrics
        - changed-region information
        - evidence-grounded answer
    """

    image_paths = get_image_paths(state)
    query = state.get(
        "query",
        "What changed between these two images?",
    )

    if len(image_paths) < 2:
        return {
            "tool": "change",
            "status": "failed",
            "answer": (
                "Change analysis requires at least two images: "
                "a before image and an after image."
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    if not CHANGE_ANALYSIS_AVAILABLE:
        return {
            "tool": "change",
            "status": "failed",
            "answer": (
                "The Change Analysis specialist could not be loaded: "
                + CHANGE_ANALYSIS_IMPORT_ERROR
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    before_path, after_path = identify_change_pair(image_paths)

    if before_path is None or after_path is None:
        return {
            "tool": "change",
            "status": "failed",
            "answer": "Could not determine the before and after images.",
            "confidence": 0.0,
            "evidence": {},
        }

    if not before_path.exists() or not after_path.exists():
        return {
            "tool": "change",
            "status": "failed",
            "answer": "One or both change-analysis images could not be found.",
            "confidence": 0.0,
            "evidence": {
                "before": str(before_path),
                "after": str(after_path),
            },
        }

    print("\n[CHANGE ANALYSIS]")
    print("BEFORE:", before_path)
    print("AFTER :", after_path)
    print("QUERY :", query)

    try:
        # Use the package pipeline directly so the result can be redirected
        # into SatQuery's per-request artifact directory.
        pipeline = get_change_analysis_pipeline()

        result = pipeline.run(
            before_path=before_path,
            after_path=after_path,
            question=query,
            case_name=f"satquery_{Path(get_artifact_dir(state)).name}",
        )

    except Exception as exc:
        print("[CHANGE ANALYSIS ERROR]", exc)
        return {
            "tool": "change",
            "status": "failed",
            "answer": f"Change analysis failed: {exc}",
            "confidence": 0.0,
            "evidence": {
                "before": before_path.name,
                "after": after_path.name,
            },
        }

    # Package-level validation/model failure.
    model_name = result.get(
        "model",
        "deterministic_fallback",
    )

    answer = str(
        result.get(
            "answer",
            "Change analysis completed.",
        )
    ).strip()

    pipeline_evidence = _copy_change_artifacts(
        state,
        result,
    )

    # Structured SatQuery evidence. Keep the raw specialist result too,
    # because it is useful to the final LLM and for debugging.
    evidence = {
        "before_image": before_path.name,
        "after_image": after_path.name,
        "model": model_name,
        "metrics": result.get("metrics", {}),
        "warnings": result.get("warnings", []),
        "parameters": result.get("parameters", {}),
        "artifacts": pipeline_evidence,
        "specialist_result": result,
    }

    # Preserve the most useful metrics at the top level for downstream nodes.
    metrics = result.get("metrics", {})

    # The supplied fallback confidence is intentionally uncalibrated and
    # therefore returns None. SatQuery represents that as 0.0 for the UI.
    raw_confidence = result.get("confidence")
    confidence = (
        float(raw_confidence)
        if raw_confidence is not None
        else 0.0
    )

    # A validation failure is a completed specialist response but should be
    # surfaced clearly rather than silently presented as successful analysis.
    if model_name == "validation_failed":
        status = "failed"
    else:
        status = "completed"

    return {
        "tool": "change",
        "status": status,
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
        "metrics": metrics,
        "model": model_name,
    }


# ============================================================
# OPTICAL + SAR MODEL
# ============================================================

def get_optical_sar_model():

    global _FUSION_MODEL

    if not OPTICAL_SAR_AVAILABLE:

        raise RuntimeError(
            "Optical-SAR dependencies/imports "
            "are unavailable: "
            + OPTICAL_SAR_IMPORT_ERROR
        )

    if _FUSION_MODEL is None:

        print(
            "\n[OPTICAL-SAR]"
        )

        print(
            "Loading DualBranchFusion..."
        )

        _FUSION_MODEL = (
            DualBranchFusion(
                num_classes=2
            )
        )

        weights_path = (
            OPTICAL_SAR_DIR
            / "fusion_weights.pt"
        )

        if not weights_path.exists():

            raise FileNotFoundError(
                "Optical-SAR weights not found: "
                f"{weights_path}"
            )

        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )

        _FUSION_MODEL.load_state_dict(
            state_dict
        )

        _FUSION_MODEL.eval()

        print(
            "DualBranchFusion loaded."
        )

    return _FUSION_MODEL


# ============================================================
# OPTICAL + SAR TOOL
# ============================================================

def optical_sar_tool(state):
    """
    Real Optical + SAR fusion specialist.

    Required:
        state["optical_path"]
        state["sar_path"]

    Produces:
        - fused mask PNG
        - georeferenced GeoTIFF
        - built-up percentage
        - water percentage
        - confidence
        - modality evidence
    """

    optical_path = state.get(
        "optical_path"
    )

    sar_path = state.get(
        "sar_path"
    )

    query = state.get(
        "query",
        "",
    )

    artifact_dir = get_artifact_dir(
        state
    )

    # ========================================================
    # VALIDATE PATHS
    # ========================================================

    if not optical_path:

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                "No Optical image was provided."
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    if not sar_path:

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                "No SAR image was provided."
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    optical_path = Path(
        optical_path
    )

    sar_path = Path(
        sar_path
    )

    if not optical_path.exists():

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                f"Optical image not found: "
                f"{optical_path}"
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    if not sar_path.exists():

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                f"SAR image not found: "
                f"{sar_path}"
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    # ========================================================
    # VALIDATE OPTICAL + SAR
    # ========================================================

    try:

        validation = (
            validate_optical_sar_pair(
                str(optical_path),
                str(sar_path),
            )
        )

    except Exception as exc:

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                "Optical-SAR validation failed: "
                f"{exc}"
            ),
            "confidence": 0.0,
            "evidence": {},
        }

    if not validation.get(
        "valid",
        False,
    ):

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                "Optical-SAR analysis halted "
                "because the input files are "
                "not compatible: "
                + "; ".join(
                    validation.get(
                        "issues",
                        [],
                    )
                )
            ),
            "confidence": 0.0,
            "evidence": {
                "validation": validation,
            },
        }

    # ========================================================
    # PREPROCESS
    # ========================================================

    try:

        optical_tensor, sar_tensor = (
            prepare_tensors(
                str(optical_path),
                str(sar_path),
            )
        )

    except Exception as exc:

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                "Optical-SAR preprocessing failed: "
                f"{exc}"
            ),
            "confidence": 0.0,
            "evidence": {
                "validation": validation,
            },
        }

    # ========================================================
    # FIX SAR CHANNEL COUNT
    # ========================================================
    #
    # The supplied model expects exactly 2 SAR channels.
    #
    # If the input SAR raster has only one band,
    # duplicate it so the model receives 2 channels.
    #
    # ========================================================

    if sar_tensor.shape[0] == 1:

        sar_tensor = np.concatenate(
            [
                sar_tensor,
                sar_tensor,
            ],
            axis=0,
        )

    elif sar_tensor.shape[0] > 2:

        sar_tensor = (
            sar_tensor[:2]
        )

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    opt_tensor = (
        torch.from_numpy(
            optical_tensor
        ).unsqueeze(0)
    )

    sar_tensor = (
        torch.from_numpy(
            sar_tensor
        ).unsqueeze(0)
    )

    # ========================================================
    # MODEL
    # ========================================================

    try:

        model = (
            get_optical_sar_model()
        )

        with torch.no_grad():

            logits = model(
                opt_tensor,
                sar_tensor,
            )

            probs = torch.softmax(
                logits,
                dim=1,
            ).squeeze(0).numpy()

            pred_mask = np.argmax(
                probs,
                axis=0,
            ).astype(
                np.uint8
            )

    except Exception as exc:

        return {
            "tool": "optical_sar",
            "status": "failed",
            "answer": (
                "Optical-SAR model inference "
                f"failed: {exc}"
            ),
            "confidence": 0.0,
            "evidence": {
                "validation": validation,
            },
        }

    # ========================================================
    # METRICS
    # ========================================================

    water_pct = float(
        np.mean(
            pred_mask == 0
        )
        * 100
    )

    builtup_pct = float(
        np.mean(
            pred_mask == 1
        )
        * 100
    )

    avg_confidence = float(
        np.mean(
            np.max(
                probs,
                axis=0,
            )
        )
    )

    # ========================================================
    # SAVE PNG MASK
    # ========================================================

    output_png = (
        artifact_dir
        / "fused_mask_evidence.png"
    )

    try:

        cmap = ListedColormap(
            [
                "blue",
                "red",
            ]
        )

        plt.imsave(
            str(output_png),
            pred_mask,
            cmap=cmap,
            vmin=0,
            vmax=1,
        )

    except Exception as exc:

        print(
            "[OPTICAL-SAR PNG ERROR]",
            exc,
        )

    # ========================================================
    # SAVE GEOREFERENCED TIFF
    # ========================================================

    output_tif = (
        artifact_dir
        / "fused_mask_georeferenced.tif"
    )

    try:

        with rasterio.open(
            str(optical_path)
        ) as src:

            crs = src.crs
            transform = (
                src.transform
                * rasterio.Affine.scale(
                    src.width / 256,
                    src.height / 256,
                )
            )

        with rasterio.open(
            str(output_tif),
            "w",
            driver="GTiff",
            height=pred_mask.shape[0],
            width=pred_mask.shape[1],
            count=1,
            dtype=pred_mask.dtype,
            crs=crs,
            transform=transform,
        ) as dst:

            dst.write(
                pred_mask,
                1,
            )

    except Exception as exc:

        print(
            "[OPTICAL-SAR TIFF ERROR]",
            exc,
        )

    # ========================================================
    # ANSWER
    # ========================================================

    answer = (
        "Joint Optical-SAR analysis completed. "
        f"The fused analysis estimates "
        f"{builtup_pct:.1f}% built-up area and "
        f"{water_pct:.1f}% surface water "
        "within the analyzed region."
    )

    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = {

        "classification_summary": {
            "built_up_percentage": round(
                builtup_pct,
                2,
            ),
            "water_percentage": round(
                water_pct,
                2,
            ),
        },

        "modality_contribution": (
            "Optical imagery provides surface "
            "reflectance information while SAR "
            "provides radar backscatter and "
            "structural information."
        ),

        "optical_file": (
            optical_path.name
        ),

        "sar_file": (
            sar_path.name
        ),

        "validation": validation,

        "query": query,
    }

    # --------------------------------------------------------
    # PNG artifact
    # --------------------------------------------------------

    if output_png.exists():

        evidence[
            "visual_mask"
        ] = get_artifact_url(
            state,
            output_png.name,
        )

    # --------------------------------------------------------
    # GeoTIFF artifact
    # --------------------------------------------------------

    if output_tif.exists():

        evidence[
            "geospatial_mask"
        ] = get_artifact_url(
            state,
            output_tif.name,
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "tool": "optical_sar",

        "status": "completed",

        "answer": answer,

        "confidence": round(
            avg_confidence,
            2,
        ),

        "evidence": evidence,
    }