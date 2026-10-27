"""Measure how well the current embedder identifies a pack, using simulated phone photos of its own
reference images (rotation, scale, off-centre framing, background, lighting, blur, JPEG).

This is a smoke test for pack maintainers, not a substitute for real held-out photos: if accuracy
is poor even here, the pack needs more or better reference images (or a different embedder).
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter

from .embed import load_image


def simulate_photo(img: Image.Image, rnd: random.Random, rotate: bool) -> Image.Image:
    img = img.convert("RGB")
    if rotate:
        img = img.rotate(rnd.uniform(0, 360), resample=Image.BICUBIC, expand=True, fillcolor=img.getpixel((0, 0)))
    else:
        img = img.rotate(rnd.uniform(-8, 8), resample=Image.BICUBIC, expand=True, fillcolor=img.getpixel((0, 0)))
    bg_color = tuple(rnd.randint(40, 220) for _ in range(3))
    scale = rnd.uniform(0.55, 0.85)
    side = int(max(img.size) / scale)
    canvas = Image.new("RGB", (side, int(side * rnd.uniform(1.0, 1.35))), bg_color)
    w, h = img.size
    canvas.paste(img, (rnd.randint(0, canvas.width - w), rnd.randint(0, canvas.height - h)))
    canvas = ImageEnhance.Brightness(canvas).enhance(rnd.uniform(0.7, 1.25))
    canvas = ImageEnhance.Contrast(canvas).enhance(rnd.uniform(0.75, 1.2))
    canvas = canvas.filter(ImageFilter.GaussianBlur(rnd.uniform(0, 1.5)))
    canvas = canvas.resize((max(1, canvas.width * 2 // 3), max(1, canvas.height * 2 // 3)))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=rnd.randint(55, 85))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def evaluate(engine, pack_id: str, per_image: int = 2, seed: int = 0) -> dict:
    pack = engine.packs[pack_id]
    rnd = random.Random(seed)
    total = top1 = top3 = 0
    misses = []
    for item in pack.items.values():
        for ref in item.images:
            for _ in range(per_image):
                q = simulate_photo(load_image(ref.path), rnd, pack.rotation_invariant)
                res = engine.identify(q, top_k=3, pack_ids=[pack_id])
                keys = [c["key"] for c in res["candidates"]]
                total += 1
                top1 += bool(keys) and keys[0] == item.key
                top3 += item.key in keys
                if not keys or keys[0] != item.key:
                    misses.append({"expected": item.key, "got": keys[:1], "image": str(ref.path)})
    return {"pack": pack_id, "embedder": engine.embedder.key, "queries": total,
            "top1": round(top1 / max(total, 1), 3), "top3": round(top3 / max(total, 1), 3), "misses": misses}
