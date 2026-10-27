"""Image embedders. All return L2-normalised float32 vectors so cosine similarity is a dot product.

- siglip  : google/siglip2-* (default; any google/siglip* checkpoint works). Strong semantic features and supports text search.
- dinov2  : facebook/dinov2-*. Strong fine-grained visual features (die varieties, perforations).
- simple  : dependency-free gradient/colour descriptor. For tests and machines without PyTorch;
            fine for tiny catalogs, not for real-world photos.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image, ImageChops, ImageOps


class Embedder(Protocol):
    key: str  # identifies model + preprocessing; part of the index cache path
    supports_text: bool

    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_image(path_or_file) -> Image.Image:
    img = Image.open(path_or_file)
    img = ImageOps.exif_transpose(img)  # phone photos
    return img.convert("RGB")


def autocrop(img: Image.Image, tolerance: int = 28, min_fraction: float = 0.08) -> Image.Image:
    """Trim a roughly uniform background (sampled from the corners) around the object, plus a small margin."""
    small = img.copy()
    small.thumbnail((256, 256))
    arr = np.asarray(small, dtype=np.int16)
    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    bg = np.median(corners, axis=0)
    if np.max(np.ptp(corners, axis=0)) > 3 * tolerance:
        return img  # busy background; cropping would be a guess
    diff = np.abs(arr - bg).max(axis=-1) > tolerance
    ys, xs = np.nonzero(diff)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if (x1 - x0) * (y1 - y0) < min_fraction * diff.size:
        return img
    sx, sy = img.width / small.width, img.height / small.height
    pad = int(0.04 * max(x1 - x0, y1 - y0))
    box = (max(0, int((x0 - pad) * sx)), max(0, int((y0 - pad) * sy)),
           min(img.width, int((x1 + pad) * sx)), min(img.height, int((y1 + pad) * sy)))
    return img.crop(box)


def square_pad(img: Image.Image) -> Image.Image:
    """Pad to square with the edge colour so resizing doesn't distort round objects."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    edge = img.getpixel((0, 0))
    canvas = Image.new("RGB", (side, side), edge)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess(img: Image.Image) -> Image.Image:
    return square_pad(autocrop(img))


class SimpleEmbedder:
    """Histogram-of-gradients + colour + thumbnail descriptor. No ML dependencies."""

    key = "simple-v1"
    supports_text = False

    def embed_images(self, images):
        return np.stack([self._one(preprocess(im)) for im in images])

    def embed_texts(self, texts):
        raise NotImplementedError("the simple embedder has no text encoder; use siglip")

    def _one(self, img: Image.Image) -> np.ndarray:
        size, cells, bins = 64, 4, 9
        gray = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins
        idx = np.minimum(ang.astype(int), bins - 1)
        cs = size // cells
        hog = np.zeros((cells, cells, bins), np.float32)
        for cy in range(cells):
            for cx in range(cells):
                sl = (slice(cy * cs, (cy + 1) * cs), slice(cx * cs, (cx + 1) * cs))
                hog[cy, cx] = np.bincount(idx[sl].ravel(), weights=mag[sl].ravel(), minlength=bins)
        hsv = np.asarray(img.convert("HSV").resize((32, 32)), dtype=np.float32) / 255.0
        color = np.concatenate([np.histogram(hsv[..., c], bins=8, range=(0, 1))[0] for c in range(3)]).astype(np.float32)
        thumb = np.asarray(img.convert("L").resize((12, 12), Image.BILINEAR), dtype=np.float32).ravel()
        thumb = thumb - thumb.mean()
        parts = [_normalize(hog.ravel()), _normalize(color) * 0.6, _normalize(thumb) * 0.8]
        return _normalize(np.concatenate(parts))


def _pooled(out):
    # transformers<5 returns a tensor; >=5 returns a model output with pooler_output.
    return out if hasattr(out, "float") else out.pooler_output


class _HFEmbedder:
    batch_size = 16

    def __init__(self, model_id: str, device: str | None = None):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "This embedder needs PyTorch and transformers: pip install 'curio[ml]'") from e
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed_images(self, images):
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = [preprocess(im) for im in images[i:i + self.batch_size]]
            with self.torch.inference_mode():
                out.append(self._image_features(batch))
        return _normalize(np.concatenate(out))


class SiglipEmbedder(_HFEmbedder):
    supports_text = True

    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str | None = None):
        super().__init__(model_id, device)
        self.key = "siglip-" + model_id.replace("/", "_")

    def _image_features(self, batch):
        inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
        return _pooled(self.model.get_image_features(**inputs)).float().cpu().numpy()

    def embed_texts(self, texts):
        with self.torch.inference_mode():
            inputs = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            return _normalize(_pooled(self.model.get_text_features(**inputs)).float().cpu().numpy())


class Dinov2Embedder(_HFEmbedder):
    supports_text = False

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str | None = None):
        super().__init__(model_id, device)
        self.key = "dinov2-" + model_id.replace("/", "_")

    def _image_features(self, batch):
        inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
        h = self.model(**inputs).last_hidden_state
        # CLS token (global) + mean of patch tokens (local detail) works better for retrieval than CLS alone.
        feats = self.torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1)
        return feats.float().cpu().numpy()

    def embed_texts(self, texts):
        raise NotImplementedError("DINOv2 has no text encoder; use siglip for text search")


DEFAULT_SIGLIP = "google/siglip2-base-patch16-224"
DEFAULT_MODELS = {"siglip": DEFAULT_SIGLIP, "dinov2": "facebook/dinov2-small"}
EMBEDDER_NAMES = ("siglip", "dinov2", "simple")

# Per-embedder defaults for turning cosine similarity into a confidence.
# temperature: softmax sharpness over candidates; min_similarity: below this we say "no confident match".
CALIBRATION = {
    "simple": {"temperature": 0.02, "min_similarity": 0.70},
    "siglip": {"temperature": 0.03, "min_similarity": 0.55},
    "dinov2": {"temperature": 0.03, "min_similarity": 0.45},
}


def create_embedder(name: str = "siglip", model_id: str | None = None, device: str | None = None) -> Embedder:
    name = name.lower()
    if name == "simple":
        return SimpleEmbedder()
    if name == "siglip":
        return SiglipEmbedder(model_id or DEFAULT_MODELS["siglip"], device)
    if name == "dinov2":
        return Dinov2Embedder(model_id or DEFAULT_MODELS["dinov2"], device)
    raise ValueError(f"unknown embedder '{name}' (choose siglip, dinov2 or simple)")


def embedder_key(name: str, model_id: str | None = None) -> str:
    """The key an embedder would have, without loading it (matches the classes' .key)."""
    name = name.lower()
    if name == "simple":
        return SimpleEmbedder.key
    if name not in DEFAULT_MODELS:
        raise ValueError(f"unknown embedder '{name}' (choose {', '.join(EMBEDDER_NAMES)})")
    return f"{name}-{(model_id or DEFAULT_MODELS[name]).replace('/', '_')}"


def family(embedder: Embedder) -> str:
    return embedder.key.split("-", 1)[0]
