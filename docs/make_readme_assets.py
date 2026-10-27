"""Regenerate the README's static images: the accuracy chart (light + dark) and the coin-pack montage.

    python docs/make_readme_assets.py

Accuracy numbers come from `curio eval` on packs/us-mint-coins (260 simulated photos); update RESULTS
when you re-measure. Screenshots and the demo GIF are captured from the running web UI separately.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

# (label, top-1 accuracy %) from `curio eval --per-image 2` on us-mint-coins.
RESULTS = [
    ("DINOv2 small", 99),
    ("DINOv2 base", 96),
    ("SigLIP 2 base, 384 px", 96),
    ("SigLIP 2 base, 224 px", 93),
]

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "bar": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33322f", "bar": "#3987e5"},
}


def accuracy_svg(theme: dict) -> str:
    W, label_w, right_pad, top, row, bar_h = 720, 190, 56, 64, 44, 22
    plot_w = W - label_w - right_pad
    H = top + row * len(RESULTS) + 40
    x = lambda pct: label_w + plot_w * pct / 100
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" role="img" '
        f'aria-label="Top-1 identification accuracy by embedder on the US Mint coin pack">',
        f'<rect width="{W}" height="{H}" fill="{theme["surface"]}" rx="8"/>',
        f'<text x="20" y="30" font-size="16" font-weight="600" fill="{theme["text"]}">Top-1 identification accuracy, US Mint coin pack</text>',
        f'<text x="20" y="50" font-size="12" fill="{theme["muted"]}">260 simulated phone photos of 86 coin types (rotation, background, lighting, blur, JPEG)</text>',
    ]
    axis_y = top + row * len(RESULTS)
    for pct in (0, 25, 50, 75, 100):
        parts.append(f'<line x1="{x(pct):.1f}" y1="{top - 6}" x2="{x(pct):.1f}" y2="{axis_y}" stroke="{theme["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x(pct):.1f}" y="{axis_y + 18}" font-size="11" fill="{theme["muted"]}" text-anchor="middle">{pct}%</text>')
    r = 4  # rounded data-end only; flat at the baseline
    for i, (label, pct) in enumerate(RESULTS):
        y = top + i * row + (row - bar_h) / 2
        x0, x1 = x(0), x(pct)
        parts.append(f'<text x="{label_w - 12}" y="{y + bar_h / 2 + 4:.1f}" font-size="13" fill="{theme["text"]}" text-anchor="end">{label}</text>')
        parts.append(f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + bar_h - r:.1f} '
                     f'Q{x1:.1f},{y + bar_h:.1f} {x1 - r:.1f},{y + bar_h:.1f} H{x0:.1f} Z" fill="{theme["bar"]}">'
                     f'<title>{label}: {pct}%</title></path>')
        parts.append(f'<text x="{x1 + 8:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="13" font-weight="600" fill="{theme["text"]}">{pct}%</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def pack_montage(cols: int = 12, rows: int = 4, cell: int = 120) -> Image.Image:
    pack = ROOT / "packs" / "us-mint-coins"
    items = [json.loads(l) for l in (pack / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # One image per item, spread across the catalog so every series shows up.
    picks = [it["images"][-1]["path"] for it in items]
    k = min(cols * rows, len(picks))
    picks = [picks[round(i * (len(picks) - 1) / max(k - 1, 1))] for i in range(k)]
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, rel in enumerate(picks):
        im = Image.open(pack / rel).convert("RGB")
        im.thumbnail((cell - 8, cell - 8))
        sheet.paste(im, ((i % cols) * cell + (cell - im.width) // 2, (i // cols) * cell + (cell - im.height) // 2))
    return sheet


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (OUT / f"accuracy-{name}.svg").write_text(accuracy_svg(theme), encoding="utf-8")
    pack_montage().save(OUT / "us-mint-pack.jpg", quality=85)
    print("wrote", sorted(p.name for p in OUT.iterdir()))
