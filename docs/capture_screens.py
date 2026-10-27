"""Capture the README screenshots and demo GIF from a running Curio web UI.

    curio --packs packs serve                     # in another terminal (default embedder: auto)
    pip install playwright
    python docs/capture_screens.py [--url http://127.0.0.1:8765] [--browser msedge|chrome|chromium]

The "phone photos" are simulated: a reference image from the pack, rotated and placed on a table-like
background. They show the UI flow; they are not evidence of real-world accuracy.
"""

from __future__ import annotations

import argparse
import io
import random
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
VIEW = {"width": 1200, "height": 760}


def phone_photo(ref: Path, angle: float, seed: int) -> Path:
    """A reference coin rotated on a warm table surface with a soft shadow and slight blur."""
    rnd = random.Random(seed)
    coin = Image.open(ref).convert("RGB")
    coin.thumbnail((520, 520))
    # Mask out the light background around the coin so only the coin lands on the table.
    mask = Image.new("L", coin.size, 0)
    ImageDraw.Draw(mask).ellipse((4, 4, coin.width - 4, coin.height - 4), fill=255)
    coin, mask = coin.rotate(angle, resample=Image.BICUBIC), mask.rotate(angle, resample=Image.BICUBIC)
    W, H = 900, 1100
    table = Image.new("RGB", (W, H))
    px = table.load()
    for y in range(H):
        for x in range(W):
            grain = 8 * ((x * 7 + y // 3) % 23 > 11)
            px[x, y] = (150 - y // 40 + grain + rnd.randint(-4, 4), 110 - y // 50 + grain, 78 - y // 60 + grain // 2)
    shadow = Image.new("L", coin.size, 0)
    shadow.paste(mask.point(lambda v: v * 0.55), (0, 0))
    pos = ((W - coin.width) // 2 + 20, (H - coin.height) // 2 - 30)
    table.paste((40, 28, 20), (pos[0] + 18, pos[1] + 22), shadow.filter(ImageFilter.GaussianBlur(14)))
    table.paste(coin, pos, mask)
    table = table.filter(ImageFilter.GaussianBlur(0.8))
    f = Path(tempfile.mkdtemp()) / f"photo-{seed}.jpg"
    table.save(f, quality=82)
    return f


def frame(page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def identify(page, photo: Path):
    page.set_input_files("#file", str(photo))
    page.wait_for_function("document.querySelector('#status').textContent !== 'identifying…' && !document.querySelector('#results').classList.contains('hidden')")
    page.wait_for_timeout(600)


def main():
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--browser", default="msedge", help="msedge, chrome, or chromium (bundled; needs `playwright install`)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gif: list[tuple[Image.Image, int]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=None if a.browser == "chromium" else a.browser)
        hi = browser.new_page(viewport=VIEW, device_scale_factor=2, color_scheme="light")
        lo = browser.new_page(viewport=VIEW, device_scale_factor=1, color_scheme="light")
        for page in (hi, lo):
            page.goto(a.url)
            page.wait_for_function("!document.querySelector('#status').textContent.includes('loading')")
        gif.append((frame(lo), 1600))

        # 1. A real US coin from the us-mint-coins pack.
        coin = phone_photo(ROOT / "packs/us-mint-coins/images/nickel-buffalo-obverse.jpg", 28, 1)
        for page in (hi, lo):
            identify(page, coin)
        hi.screenshot(path=str(OUT / "identify.png"))
        gif.append((frame(lo), 3200))
        lo.evaluate("document.querySelector('#detail h2').scrollIntoView({block: 'start'})")
        lo.wait_for_timeout(300)
        gif.append((frame(lo), 2600))
        for page in (hi, lo):
            page.evaluate("window.scrollTo(0, 0)")

        # 2. One specific design out of many similar quarters.
        quarter = phone_photo(ROOT / "packs/us-mint-coins/images/quarter-atb-yellowstone-reverse.jpg", -40, 2)
        identify(lo, quarter)
        gif.append((frame(lo), 3200))
        lo.evaluate("document.querySelector('#detail h2').scrollIntoView({block: 'start'})")
        lo.wait_for_timeout(300)
        gif.append((frame(lo), 2600))

        # 3. The installed packs.
        lo.evaluate("window.scrollTo(0, 0)")
        lo.click("nav button[data-tab=packs]")
        lo.wait_for_timeout(300)
        gif.append((frame(lo), 2600))
        browser.close()

    frames = [f.resize((900, round(900 * f.height / f.width)), Image.LANCZOS) for f, _ in gif]
    palette = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(OUT / "demo.gif", save_all=True, append_images=frames[1:],
                   duration=[d for _, d in gif], loop=0, optimize=True)
    for name in ("identify.png", "demo.gif"):
        print(name, (OUT / name).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
