"""Generate two small, entirely fictional demo packs (coins and stamps) so Curio runs out of the box.

The images are procedurally drawn; they exist to exercise the engine and show the pack format.

    python scripts/make_demo_packs.py [out_dir]   # default: ./packs
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

METALS = {"copper": (184, 115, 51), "silver": (192, 192, 196), "gold": (212, 175, 55), "nickel": (160, 165, 160)}


def font(size):
    return ImageFont.load_default(size=size)


def star(d, cx, cy, r, n, fill, rot=0.0):
    pts = []
    for i in range(2 * n):
        rr = r if i % 2 == 0 else r * 0.45
        a = rot + math.pi * i / n - math.pi / 2
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=fill)


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def coin_face(metal, motif, text, year, side, seed):
    rnd = random.Random(seed)
    S = 512
    img = Image.new("RGB", (S, S), (245, 245, 242))
    d = ImageDraw.Draw(img)
    base = METALS[metal]
    d.ellipse((16, 16, S - 16, S - 16), fill=shade(base, 0.7))
    d.ellipse((34, 34, S - 34, S - 34), fill=base)
    for i in range(0, 360, 4):  # reeded rim
        a = math.radians(i)
        d.line([(S / 2 + 228 * math.cos(a), S / 2 + 228 * math.sin(a)),
                (S / 2 + 238 * math.cos(a), S / 2 + 238 * math.sin(a))], fill=shade(base, 0.55), width=2)
    dark = shade(base, 0.5)
    c = S / 2
    if side == "reverse":
        d.text((c, c - 20), text, font=font(110), fill=dark, anchor="mm")
        d.text((c, c + 80), "EXAMPLIA", font=font(34), fill=dark, anchor="mm")
        for k in range(motif["wreath"]):
            a = math.pi * (0.65 + 0.7 * k / max(1, motif["wreath"] - 1))
            for sgn in (-1, 1):
                x, y = c + sgn * 170 * math.cos(a), c + 170 * math.sin(a)
                d.ellipse((x - 14, y - 7, x + 14, y + 7), fill=dark)
        return img
    kind = motif["kind"]
    if kind == "star":
        star(d, c, c - 10, 120, motif["n"], dark, rnd.uniform(0, 0.3))
    elif kind == "rings":
        for r in range(40, 150, motif["n"]):
            d.ellipse((c - r, c - 10 - r, c + r, c - 10 + r), outline=dark, width=6)
    elif kind == "polygon":
        pts = [(c + 120 * math.cos(2 * math.pi * i / motif["n"]), c - 10 + 120 * math.sin(2 * math.pi * i / motif["n"]))
               for i in range(motif["n"])]
        d.polygon(pts, outline=dark, width=10)
        d.ellipse((c - 30, c - 40, c + 30, c + 20), fill=dark)
    elif kind == "cross":
        d.rectangle((c - 22, c - 140, c + 22, c + 110), fill=dark)
        d.rectangle((c - 120, c - 60, c + 120, c - 16), fill=dark)
    elif kind == "head":  # stylised portrait
        d.ellipse((c - 80, c - 130, c + 70, c + 30), fill=dark)
        d.polygon([(c - 60, c + 10), (c + 50, c + 10), (c + 90, c + 120), (c - 100, c + 120)], fill=dark)
        d.ellipse((c + 40, c - 80, c + 95, c - 40), fill=dark)
    d.text((c, c + 170), str(year), font=font(48), fill=dark, anchor="mm")
    return img


def stamp(color, motif, value, seed):
    W, H = 400, 480
    img = Image.new("RGB", (W + 60, H + 60), (238, 236, 230))
    d = ImageDraw.Draw(img)
    x0, y0 = 30, 30
    d.rectangle((x0, y0, x0 + W, y0 + H), fill=(250, 248, 240))
    for x in range(x0, x0 + W + 1, 20):  # perforations
        for y in (y0, y0 + H):
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(238, 236, 230))
    for y in range(y0, y0 + H + 1, 20):
        for x in (x0, x0 + W):
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(238, 236, 230))
    d.rectangle((x0 + 30, y0 + 30, x0 + W - 30, y0 + H - 30), fill=color)
    inner = (250, 248, 240)
    cx, cy = x0 + W / 2, y0 + H / 2 - 30
    if motif == "ship":
        d.polygon([(cx - 120, cy + 40), (cx + 120, cy + 40), (cx + 80, cy + 90), (cx - 80, cy + 90)], fill=inner)
        d.polygon([(cx, cy - 120), (cx, cy + 30), (cx + 90, cy + 30)], fill=inner)
        d.polygon([(cx - 8, cy - 100), (cx - 8, cy + 30), (cx - 80, cy + 30)], fill=inner)
    elif motif == "bird":
        d.ellipse((cx - 70, cy - 40, cx + 50, cy + 40), fill=inner)
        d.ellipse((cx + 30, cy - 70, cx + 90, cy - 10), fill=inner)
        d.polygon([(cx - 60, cy - 10), (cx - 150, cy - 80), (cx - 20, cy - 30)], fill=inner)
        d.polygon([(cx + 85, cy - 45), (cx + 125, cy - 35), (cx + 85, cy - 25)], fill=inner)
    elif motif == "mountain":
        d.polygon([(cx - 140, cy + 90), (cx - 40, cy - 90), (cx + 30, cy + 20), (cx + 70, cy - 40), (cx + 140, cy + 90)], fill=inner)
        d.ellipse((cx + 60, cy - 130, cx + 110, cy - 80), fill=inner)
    elif motif == "flower":
        for k in range(6):
            a = 2 * math.pi * k / 6
            d.ellipse((cx + 60 * math.cos(a) - 40, cy + 60 * math.sin(a) - 40, cx + 60 * math.cos(a) + 40, cy + 60 * math.sin(a) + 40), fill=inner)
        d.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=color)
    elif motif == "crown":
        d.polygon([(cx - 110, cy + 60), (cx - 110, cy - 60), (cx - 55, cy), (cx, cy - 90), (cx + 55, cy), (cx + 110, cy - 60), (cx + 110, cy + 60)], fill=inner)
    d.text((cx, y0 + H - 70), value, font=font(54), fill=inner, anchor="mm")
    d.text((cx, y0 + 60), "EXAMPLIA", font=font(30), fill=inner, anchor="mm")
    return img


def make_coins(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    designs = [
        ("1-cent-1921", "1 Cent 1921", "copper", {"kind": "head", "n": 0, "wreath": 6}, "1", 1921),
        ("1-cent-1938", "1 Cent 1938", "copper", {"kind": "head", "n": 0, "wreath": 6}, "1", 1938),
        ("5-cent-1921", "5 Cents 1921", "nickel", {"kind": "star", "n": 5, "wreath": 4}, "5", 1921),
        ("5-cent-1950", "5 Cents 1950 (6-point star)", "nickel", {"kind": "star", "n": 6, "wreath": 4}, "5", 1950),
        ("10-cent-1905", "10 Cents 1905", "silver", {"kind": "rings", "n": 22, "wreath": 8}, "10", 1905),
        ("10-cent-1960", "10 Cents 1960", "silver", {"kind": "polygon", "n": 8, "wreath": 8}, "10", 1960),
        ("25-cent-1912", "25 Cents 1912", "silver", {"kind": "cross", "n": 0, "wreath": 10}, "25", 1912),
        ("50-cent-1899", "50 Cents 1899", "silver", {"kind": "polygon", "n": 5, "wreath": 12}, "50", 1899),
        ("1-crown-1888", "1 Crown 1888", "gold", {"kind": "head", "n": 0, "wreath": 14}, "1C", 1888),
        ("2-crown-1901", "2 Crowns 1901", "gold", {"kind": "star", "n": 8, "wreath": 14}, "2C", 1901),
    ]
    items = []
    for i, (cid, title, metal, motif, text, year) in enumerate(designs):
        imgs = []
        for side in ("obverse", "reverse"):
            p = f"images/{cid}-{side}.jpg"
            coin_face(metal, motif, text, year, side, seed=i).save(root / p, quality=92)
            imgs.append({"path": p, "side": side, "license": "CC0-1.0", "source": "procedurally generated"})
        items.append({"id": cid, "title": f"Examplia {title}",
                      "attributes": {"country": "Examplia (fictional)", "denomination": title.split(" (")[0].rsplit(" ", 1)[0],
                                     "year": year, "composition": metal},
                      "images": imgs})
    write_pack(root, {
        "id": "demo-coins", "name": "Demo: coins of Examplia", "domain": "coin", "version": "1.0.0",
        "license": "CC0-1.0", "rotation_invariant": True,
        "description": "Fictional coins, for trying Curio.",
        "maintainers": ["curio project"],
    }, items)


def make_stamps(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    designs = [
        ("ship-5c-blue", "5c Sailing Ship, blue", (40, 70, 140), "ship", "5c"),
        ("ship-5c-red", "5c Sailing Ship, red (color error)", (170, 40, 40), "ship", "5c"),
        ("bird-10c", "10c Kingfisher", (30, 120, 110), "bird", "10c"),
        ("mountain-20c", "20c Summit", (90, 60, 120), "mountain", "20c"),
        ("flower-1c", "1c Wildflower", (190, 120, 30), "flower", "1c"),
        ("crown-1d", "1d Coronation", (120, 30, 60), "crown", "1d"),
    ]
    items = []
    for i, (sid, title, color, motif, value) in enumerate(designs):
        p = f"images/{sid}.jpg"
        stamp(color, motif, value, i).save(root / p, quality=92)
        items.append({"id": sid, "title": f"Examplia {title}",
                      "attributes": {"country": "Examplia (fictional)", "face_value": value, "color": title.split(", ")[-1] if ", " in title else ""},
                      "images": [{"path": p, "side": "front", "license": "CC0-1.0", "source": "procedurally generated"}]})
    write_pack(root, {
        "id": "demo-stamps", "name": "Demo: stamps of Examplia", "domain": "stamp", "version": "1.0.0",
        "license": "CC0-1.0", "rotation_invariant": False,
        "description": "Fictional stamps, for trying Curio.",
        "maintainers": ["curio project"],
    }, items)


def write_pack(root: Path, meta: dict, items: list[dict]):
    (root / "pack.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (root / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "packs")
    make_coins(out / "demo-coins")
    make_stamps(out / "demo-stamps")
    print(f"wrote demo packs to {out}")
