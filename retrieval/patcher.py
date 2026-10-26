    
from pathlib import Path
from PIL import Image


PATCH_SIZE = 224
STRIDE = 168


def create_patches(image_path):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    patches = []

    patch_id = 0

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):

        for x in range(0, width - PATCH_SIZE + 1, STRIDE):

            patch = image.crop(
                (
                    x,
                    y,
                    x + PATCH_SIZE,
                    y + PATCH_SIZE
                )
            )

            patches.append({
                "id": patch_id,
                "image": patch,
                "bbox": [
                    x,
                    y,
                    x + PATCH_SIZE,
                    y + PATCH_SIZE
                ]
            })

            patch_id += 1

    return image, patches