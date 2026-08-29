"""WD14 danbooru-style image tagging.

A self-contained module: it imports nothing from the web app, and the web app
only needs `tag_image_file` and the two prompt-composition helpers. The ONNX
session and tag vocabulary are loaded lazily on first use and cached, so a
server that never ticks a tagging checkbox pays nothing for this module.

Model files live in `models/wd14/` and are not required for import. When they
are missing, tagging returns no tags and says why, rather than failing a
generation the user asked for.

The model is SmilingWolf's WD ViT tagger v3, which expects a square RGB image
padded with white, resized to the size its input tensor declares, in BGR
channel order and 0-255 range with no further normalization.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "wd14"
MODEL_FILE = MODEL_DIR / "model.onnx"
TAG_VOCABULARY_FILE = MODEL_DIR / "selected_tags.csv"

# Tagging a full-resolution reference costs time for detail the tagger cannot
# use, since it downsamples to a few hundred pixels square regardless.
MAX_TAGGED_PIXELS = 1024 * 1024

DEFAULT_GENERAL_TAG_THRESHOLD = 0.35
DEFAULT_CHARACTER_TAG_THRESHOLD = 0.85

GENERAL_TAG_CATEGORY = 0
CHARACTER_TAG_CATEGORY = 4

_loaded_tagger = None
_tagger_load_lock = threading.Lock()


class TaggerUnavailable(RuntimeError):
    """The tagger cannot run, with a reason worth showing the user."""


def downscale_image_to_pixel_budget(image: Image.Image,
                                    max_pixels: int = MAX_TAGGED_PIXELS) -> Image.Image:
    """Shrink an image to fit a total pixel budget, preserving aspect ratio."""
    pixel_count = image.width * image.height
    if pixel_count <= max_pixels:
        return image
    scale = (max_pixels / pixel_count) ** 0.5
    return image.resize((max(1, int(image.width * scale)),
                         max(1, int(image.height * scale))),
                        Image.LANCZOS)


def format_danbooru_tags_for_prompt(tags: list[str]) -> str:
    """Render tags the way a prompt wants them: spaces, single comma separated."""
    readable_tags = [tag.replace("_", " ").strip() for tag in tags]
    return ", ".join(tag for tag in readable_tags if tag)


def _load_tagger():
    """Build and cache the ONNX session and tag vocabulary."""
    global _loaded_tagger
    if _loaded_tagger is not None:
        return _loaded_tagger
    with _tagger_load_lock:
        if _loaded_tagger is not None:
            return _loaded_tagger
        for required_file in (MODEL_FILE, TAG_VOCABULARY_FILE):
            if not required_file.is_file():
                raise TaggerUnavailable(f"WD14 tagger file is missing: {required_file}")
        import onnxruntime

        # CPU only: the GPU is fully committed to the diffusion model, and
        # tagging one image is fast enough that competing for VRAM is not worth it.
        session = onnxruntime.InferenceSession(str(MODEL_FILE),
                                               providers=["CPUExecutionProvider"])
        with TAG_VOCABULARY_FILE.open(newline="", encoding="utf-8") as vocabulary:
            rows = list(csv.DictReader(vocabulary))
        tag_names = [row["name"] for row in rows]
        tag_categories = [int(row["category"]) for row in rows]
        _, height, _, _ = session.get_inputs()[0].shape
        _loaded_tagger = (session, tag_names, tag_categories, int(height))
        return _loaded_tagger


def _prepare_image_for_tagger(image: Image.Image, model_input_size: int) -> np.ndarray:
    """Pad to square on white, resize to the model's input, in BGR 0-255 float."""
    image = downscale_image_to_pixel_budget(image.convert("RGBA"))
    white_backing = Image.new("RGBA", image.size, (255, 255, 255, 255))
    flattened = Image.alpha_composite(white_backing, image).convert("RGB")

    square_edge = max(flattened.size)
    padded = Image.new("RGB", (square_edge, square_edge), (255, 255, 255))
    padded.paste(flattened, ((square_edge - flattened.width) // 2,
                             (square_edge - flattened.height) // 2))
    resized = padded.resize((model_input_size, model_input_size), Image.BICUBIC)

    pixels = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(pixels[:, :, ::-1], axis=0)


def tag_image_file(
    image_path: Path,
    general_threshold: float = DEFAULT_GENERAL_TAG_THRESHOLD,
    character_threshold: float = DEFAULT_CHARACTER_TAG_THRESHOLD,
) -> list[str]:
    """Danbooru-style tags for one image, most confident first.

    Character tags lead, since they name the subject, followed by general tags.
    Raises TaggerUnavailable when the model files are absent.
    """
    session, tag_names, tag_categories, model_input_size = _load_tagger()
    with Image.open(image_path) as opened_image:
        opened_image.load()
        model_input = _prepare_image_for_tagger(opened_image, model_input_size)

    confidences = session.run(None, {session.get_inputs()[0].name: model_input})[0][0]

    character_tags = []
    general_tags = []
    for name, category, confidence in zip(tag_names, tag_categories, confidences):
        if category == CHARACTER_TAG_CATEGORY and confidence >= character_threshold:
            character_tags.append((confidence, name))
        elif category == GENERAL_TAG_CATEGORY and confidence >= general_threshold:
            general_tags.append((confidence, name))

    character_tags.sort(reverse=True)
    general_tags.sort(reverse=True)
    return [name for _, name in character_tags] + [name for _, name in general_tags]
