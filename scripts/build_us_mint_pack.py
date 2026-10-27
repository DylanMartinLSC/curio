"""Build the `us-mint-coins` catalog pack from public-domain United States Mint images on Wikimedia Commons.

Every image is checked before it is used: Commons must mark the file as public domain AND its author,
credit or source must be the United States Mint (works of the US federal government are public domain,
17 U.S.C. 105). Files that fail the check are skipped with a reason, never silently included.

The pack's contents are defined by a reviewable source list (catalogs/us-mint-coins.sources.json):
coin types, their attributes, and the exact Commons file names for each side.

    # build (downloads are cached, so reruns only fetch what changed)
    python scripts/build_us_mint_pack.py build [--sources catalogs/us-mint-coins.sources.json] [--out packs/us-mint-coins]

    # find more verified images to add to the source list
    python scripts/build_us_mint_pack.py discover "Roosevelt dime reverse"
    python scripts/build_us_mint_pack.py discover --category "American Women quarters"

Only the Python standard library and Pillow are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CurioPackBuilder/0.1 (open-source collectible identifier; https://commons.wikimedia.org/wiki/User:Curio)"
THUMB_WIDTH = 640  # plenty for embedding models (224-384 px input) and keeps packs small
MINT_PROVENANCE = re.compile(r"united\s+states\s+mint|u\.\s?s\.\s+mint|\bus\s+mint\b|usmint\.gov", re.I)
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- Commons API

class Commons:
    def __init__(self, delay: float = 0.25):
        self.delay = delay
        self._last = 0.0

    def _get(self, url: str, binary: bool = False):
        for attempt in range(5):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                return data if binary else json.loads(data)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(float(e.headers.get("Retry-After") or 2 ** (attempt + 1)))
                    continue
                raise
            except urllib.error.URLError:
                if attempt < 4:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise

    def api(self, **params) -> dict:
        params |= {"format": "json", "formatversion": "2", "maxlag": "5"}
        return self._get(API + "?" + urllib.parse.urlencode(params))

    def file_info(self, titles: list[str]) -> dict[str, dict]:
        """Metadata for File: pages, keyed by the title as requested (redirects and normalisation resolved)."""
        out = {}
        for i in range(0, len(titles), 50):
            chunk = titles[i:i + 50]
            res = self.api(action="query", titles="|".join(chunk), redirects="1", prop="imageinfo",
                           iiprop="url|size|sha1|mime|extmetadata", iiurlwidth=str(THUMB_WIDTH),
                           iiextmetadatafilter="LicenseShortName|License|Artist|Credit|ImageDescription|DateTimeOriginal")
            q = res.get("query", {})
            alias = {}
            for n in q.get("normalized", []) + q.get("redirects", []):
                alias[n["to"]] = alias.get(n["from"], n["from"])
            for page in q.get("pages", []):
                requested = alias.get(page["title"], page["title"])
                out[requested] = page
        return out

    def search(self, query: str, limit: int = 30) -> list[str]:
        res = self.api(action="query", list="search", srsearch=query, srnamespace="6", srlimit=str(limit))
        return [s["title"] for s in res["query"]["search"]]

    def category(self, name: str, limit: int = 500) -> list[str]:
        res = self.api(action="query", list="categorymembers", cmtitle="Category:" + name.removeprefix("Category:"),
                       cmtype="file", cmlimit=str(limit))
        return [m["title"] for m in res["query"]["categorymembers"]]

    def download(self, url: str) -> bytes:
        return self._get(url, binary=True)


def _text(meta: dict, key: str) -> str:
    raw = meta.get(key, {}).get("value", "")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(raw))).strip()


def verify(page: dict) -> tuple[bool, str, dict]:
    """Is this file public domain and from the US Mint? Returns (ok, reason, attribution)."""
    if page.get("missing") or not page.get("imageinfo"):
        return False, "file not found on Commons", {}
    ii = page["imageinfo"][0]
    meta = ii.get("extmetadata", {})
    lic_short, lic = _text(meta, "LicenseShortName"), _text(meta, "License").lower()
    artist, credit = _text(meta, "Artist"), _text(meta, "Credit")
    attribution = {
        "title": page["title"], "page": ii.get("descriptionurl", ""), "license": lic_short or lic,
        "artist": artist, "credit": credit, "sha1": ii.get("sha1", ""),
    }
    if not (lic_short.lower() == "public domain" or lic.startswith("pd")):
        return False, f"not public domain (license: {lic_short or lic or 'unknown'})", attribution
    if not MINT_PROVENANCE.search(f"{artist} {credit}"):
        return False, f"public domain but not credited to the US Mint (artist: {artist or '-'}; credit: {credit[:60] or '-'})", attribution
    if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp", "image/tiff"):
        return False, f"unsupported type {ii.get('mime')}", attribution
    return True, "public domain, US Mint", attribution


# ---------------------------------------------------------------- build

def to_reference_jpeg(data: bytes, crop: str | None = None) -> bytes:
    """Optionally keep one half of a side-by-side image, flatten transparency onto a light neutral
    background, and re-encode as JPEG."""
    img = Image.open(io.BytesIO(data))
    img.load()
    if crop == "left":
        img = img.crop((0, 0, img.width // 2, img.height))
    elif crop == "right":
        img = img.crop((img.width // 2, 0, img.width, img.height))
    elif crop:
        raise ValueError(f"unknown crop '{crop}' (use left or right)")
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (236, 236, 232, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _entries(files: list) -> list[dict]:
    """Source-list file entries are either "File:..." or {"file": "File:...", "crop": "left"|"right"}."""
    return [f if isinstance(f, dict) else {"file": f} for f in files]


def build(sources_path: Path, out: Path, cache: Path, strict: bool) -> int:
    spec = json.loads(sources_path.read_text(encoding="utf-8"))
    commons = Commons()
    items = spec["items"]
    ids = [it["id"] for it in items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        print(f"duplicate item ids in source list: {sorted(dupes)}", file=sys.stderr)
        return 1
    wanted = sorted({e["file"] for it in items for files in it["files"].values() for e in _entries(files)})
    print(f"checking {len(wanted)} Commons files for {len(items)} coin types ...")
    info = commons.file_info(wanted)

    rejected: list[tuple[str, str]] = []
    ok_files: dict[str, dict] = {}
    for title in wanted:
        ok, reason, attr = verify(info.get(title, {"missing": True}))
        if ok:
            ok_files[title] = attr | {"thumb": info[title]["imageinfo"][0].get("thumburl") or info[title]["imageinfo"][0]["url"]}
        else:
            rejected.append((title, reason))
    for title, reason in rejected:
        print(f"  REJECTED {title}: {reason}", file=sys.stderr)
    if rejected and strict:
        print(f"{len(rejected)} file(s) failed verification; fix the source list or pass --allow-skips", file=sys.stderr)
        return 1

    (out / "images").mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    lock, lines, sources_md = {}, [], []
    used_images: set[str] = set()
    for it in items:
        images = []
        for side, files in it["files"].items():
            for n, entry in enumerate(_entries(files)):
                title, crop = entry["file"], entry.get("crop")
                if title not in ok_files:
                    continue
                attr = ok_files[title]
                cached = cache / f"{attr['sha1']}{'-' + crop if crop else ''}.jpg"
                if not cached.is_file():
                    print(f"  downloading {title}")
                    cached.write_bytes(to_reference_jpeg(commons.download(attr["thumb"]), crop))
                rel = f"images/{it['id']}-{side}{'' if n == 0 else f'-{n + 1}'}.jpg"
                dest = out / rel
                if not dest.is_file() or dest.read_bytes() != cached.read_bytes():
                    dest.write_bytes(cached.read_bytes())
                used_images.add(rel)
                images.append({"path": rel, "side": side, "license": "Public domain (US government work)",
                               "source": attr["page"]})
                lock[title] = {"sha1": attr["sha1"], "page": attr["page"], "artist": attr["artist"], "credit": attr["credit"]}
                sources_md.append(f"| `{rel}` | [{title.removeprefix('File:')}]({attr['page']}) | {attr['artist'] or '-'} |")
        if not images:
            print(f"  WARNING {it['id']}: no verified images, item left out", file=sys.stderr)
            continue
        lines.append(json.dumps({"id": it["id"], "title": it["title"], "attributes": it.get("attributes", {}),
                                 "images": images, "notes": it.get("notes", "")},
                                ensure_ascii=False))

    for stale in (out / "images").glob("*.jpg"):  # images from items/files that were removed
        if f"images/{stale.name}" not in used_images:
            stale.unlink()

    # No timestamps: rebuilding from the same sources must produce byte-identical output.
    meta = spec["pack"] | {"generator": "scripts/build_us_mint_pack.py"}
    (out / "pack.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "sources.lock.json").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "SOURCES.md").write_text(
        "# Image sources\n\nAll images are works of the United States Mint, a US government agency, and are in the "
        "public domain in the United States (17 U.S.C. 105). Each was verified on Wikimedia Commons as public "
        "domain and credited to the US Mint when this pack was built. Generated by "
        "`scripts/build_us_mint_pack.py`; do not edit by hand.\n\n| Image | Commons file | Author |\n|---|---|---|\n"
        + "\n".join(sources_md) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(lines)} items, {len(used_images)} images, {len(rejected)} file(s) rejected")
    return 0


# ---------------------------------------------------------------- discover

def discover(query: str | None, category: str | None, show_rejected: bool) -> int:
    commons = Commons()
    titles = commons.category(category) if category else commons.search(query)
    info = commons.file_info(titles)
    for t in titles:
        ok, reason, _ = verify(info.get(t, {"missing": True}))
        if ok or show_rejected:
            print(f"{'OK  ' if ok else 'NO  '} {json.dumps(t)}" + ("" if ok else f"   ({reason})"))
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Commons titles contain characters Windows consoles can't encode
        stream.reconfigure(errors="replace")
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", type=Path, default=ROOT / "catalogs" / "us-mint-coins.sources.json")
    b.add_argument("--out", type=Path, default=ROOT / "packs" / "us-mint-coins")
    b.add_argument("--cache", type=Path, default=ROOT / ".cache" / "commons")
    b.add_argument("--allow-skips", action="store_true", help="build even if some files fail verification")
    d = sub.add_parser("discover")
    d.add_argument("query", nargs="?")
    d.add_argument("--category")
    d.add_argument("--show-rejected", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return build(a.sources, a.out, a.cache, strict=not a.allow_skips)
    if not a.query and not a.category:
        ap.error("discover needs a search query or --category")
    return discover(a.query, a.category, a.show_rejected)


if __name__ == "__main__":
    sys.exit(main())
