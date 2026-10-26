from pathlib import Path

from PIL import Image, ImageDraw
from .patcher import create_patches
from .clip_encoder import (
    encode_images,
    encode_text,
)
from .faiss_index import FAISSIndex


# Keeps the FAISS index in memory so we don't
# recreate all patches for every query.
CACHE = {}


# ---------------------------------------------------------
# BUILD IMAGE INDEX
# ---------------------------------------------------------

def build_index(image_path):

    image_path = str(Path(image_path))

    # Reuse existing index
    if image_path in CACHE:
        return CACHE[image_path]

    print("\nCreating satellite patches...")

    full_image, patches = create_patches(
        image_path
    )

    print(
        f"Created {len(patches)} patches"
    )

    # Extract PIL images
    images = [
        patch["image"]
        for patch in patches
    ]

    print(
        "Encoding patches with CLIP..."
    )

    # Batch encode all patches
    embeddings = encode_images(
        images
    )

    print(
        f"Patch embeddings shape: "
        f"{embeddings.shape}"
    )

    # Create FAISS index
    index = FAISSIndex(
        embeddings.shape[1]
    )

    metadata = []

    for patch in patches:

        metadata.append({
            "id": patch["id"],
            "bbox": patch["bbox"],
            "image": patch["image"],
        })

    index.add(
        embeddings,
        metadata
    )

    CACHE[image_path] = (
        full_image,
        index
    )

    print("FAISS index ready.")

    return full_image, index


# ---------------------------------------------------------
# QUERY RETRIEVAL
# ---------------------------------------------------------

def retrieve(
    image_path,
    query,
    top_k=4
):

    full_image, index = build_index(
        image_path
    )

    print(
        "Encoding query with CLIP..."
    )

    query_embedding = encode_text(
        query
    )

    results = index.search(
        query_embedding,
        top_k=top_k
    )

    return full_image, results


# ---------------------------------------------------------
# CREATE CONTACT SHEET FOR SmolVLM
# ---------------------------------------------------------

def make_contact_sheet(
    patches,
    output_path="retrieval/retrieved_patches.jpg"
):

    if not patches:

        raise RuntimeError(
            "No patches were retrieved."
        )

    patch_size = 224

    margin = 20

    label_height = 35

    columns = 2

    rows = (
        len(patches) + columns - 1
    ) // columns

    width = (
        columns * patch_size
        + (columns + 1) * margin
    )

    height = (
        rows * (
            patch_size
            + label_height
            + margin
        )
        + margin
    )

    sheet = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(
        sheet
    )

    for i, patch in enumerate(
        patches
    ):

        row = i // columns
        col = i % columns

        x = (
            margin
            + col * (
                patch_size
                + margin
            )
        )

        y = (
            margin
            + row * (
                patch_size
                + label_height
                + margin
            )
        )

        patch_image = patch[
            "image"
        ].resize(
            (
                patch_size,
                patch_size
            )
        )

        sheet.paste(
            patch_image,
            (x, y)
        )

        label = (
            f"P{patch['id']}  "
            f"score={patch['score']:.3f}"
        )

        draw.text(
            (
                x,
                y + patch_size + 5
            ),
            label,
            fill="black"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sheet.save(
        output_path,
        quality=95
    )

    print(
        f"Retrieved patch sheet: "
        f"{output_path}"
    )

    return str(output_path)