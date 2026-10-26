from pathlib import Path

from PIL import Image, ImageDraw


def ground_patches(
    original_image_path,
    retrieved_patches,
    output_path="grounded_result.jpg",
):

    image = Image.open(
        original_image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    for rank, patch in enumerate(
        retrieved_patches,
        start=1,
    ):

        x1, y1, x2, y2 = patch["bbox"]

        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=8,
            outline="red",
            width=5,
        )

        label = (
            f"#{rank} "
            f"P{patch['id']} "
            f"{patch['score']:.3f}"
        )

        draw.text(
            (x1 + 6, y1 + 6),
            label,
            fill="red",
        )

    output_path = Path(output_path)

    image.save(
        output_path,
        quality=95,
    )

    return str(output_path)