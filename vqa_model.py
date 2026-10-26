import torch

from transformers import pipeline


MODEL_NAME = (
    "HuggingFaceTB/SmolVLM-500M-Instruct"
)


DEVICE = (
    0
    if torch.cuda.is_available()
    else -1
)


print(
    "Loading SmolVLM..."
)

pipe = pipeline(
    "image-text-to-text",
    model=MODEL_NAME,
    device=DEVICE,
)

print(
    "SmolVLM loaded."
)


def ask_vqa(
    image_path: str,
    question: str,
) -> str:

    messages = [
        {
            "role": "user",

            "content": [

                {
                    "type": "image",
                    "url": image_path,
                },

                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    result = pipe(
        text=messages,
        max_new_tokens=60,
    )

    generated = result[0][
        "generated_text"
    ]

    # Hugging Face returns the conversation
    # after generation.

    if isinstance(generated, list):

        last_message = generated[-1]

        if isinstance(
            last_message,
            dict
        ):

            return last_message.get(
                "content",
                ""
            ).strip()

        return str(
            last_message
        ).strip()

    return str(
        generated
    ).strip()


def generate_caption(
    image_path: str
) -> str:

    return ask_vqa(

        image_path,

        (
    "Describe the visible geographic features "
    "in this satellite image. Explain the main "
    "land-cover or land-use characteristics and "
    "important visible features. Provide a clear "
    "and informative answer in 2 to 4 sentences. "
    "Base the answer only on what is visibly present "
    "in the image and do not invent details."
),
    )