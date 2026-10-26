import torch
import numpy as np

from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Loading CLIP...")

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model = CLIPModel.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print("CLIP ready:", DEVICE)


# ---------------------------------------------------------
# IMAGE EMBEDDINGS
# ---------------------------------------------------------

def encode_images(images):

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.get_image_features(
            **inputs
        )

        # Newer Transformers versions can return
        # BaseModelOutputWithPooling instead of a tensor.
        if hasattr(outputs, "pooler_output"):

            features = outputs.pooler_output

        elif hasattr(outputs, "last_hidden_state"):

            features = outputs.last_hidden_state[:, 0]

        else:

            features = outputs

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features.cpu().numpy().astype(
        np.float32
    )


# ---------------------------------------------------------
# TEXT EMBEDDING
# ---------------------------------------------------------

def encode_text(query):

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.get_text_features(
            **inputs
        )

        # Same compatibility handling for text.
        if hasattr(outputs, "pooler_output"):

            features = outputs.pooler_output

        elif hasattr(outputs, "last_hidden_state"):

            features = outputs.last_hidden_state[:, 0]

        else:

            features = outputs

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features.cpu().numpy().astype(
        np.float32
    )