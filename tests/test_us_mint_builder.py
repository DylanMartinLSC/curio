"""Offline tests for scripts/build_us_mint_pack.py (no network)."""

import importlib.util
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts" / "build_us_mint_pack.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def page(license_short="Public domain", license="pd", artist="", credit="", mime="image/png"):
    meta = {"LicenseShortName": {"value": license_short}, "License": {"value": license},
            "Artist": {"value": artist}, "Credit": {"value": credit}}
    return {"title": "File:X.png", "imageinfo": [{"extmetadata": meta, "mime": mime, "sha1": "abc",
                                                  "descriptionurl": "https://commons.wikimedia.org/wiki/File:X.png"}]}


def test_accepts_public_domain_us_mint():
    ok, reason, _ = builder.verify(page(artist="<a href='x'>United States Mint</a>"))
    assert ok, reason
    assert builder.verify(page(credit="https://www.usmint.gov/coins/foo"))[0]
    assert builder.verify(page(artist="US Mint (coin), National Numismatic Collection (photograph)"))[0]


def test_rejects_other_licenses_and_unknown_provenance():
    ok, reason, _ = builder.verify(page(license_short="CC BY-SA 4.0", license="cc-by-sa-4.0", artist="United States Mint"))
    assert not ok and "not public domain" in reason
    ok, reason, _ = builder.verify(page(artist="Some Photographer", credit="Own work"))
    assert not ok and "not credited" in reason
    assert not builder.verify({"missing": True})[0]
    assert not builder.verify(page(artist="United States Mint", mime="image/svg+xml"))[0]


def test_reference_jpeg_crop_and_transparency():
    img = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (0, 0, 100, 100))
    img.paste((0, 0, 255, 255), (100, 0, 200, 100))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    left = Image.open(io.BytesIO(builder.to_reference_jpeg(buf.getvalue(), "left")))
    right = Image.open(io.BytesIO(builder.to_reference_jpeg(buf.getvalue(), "right")))
    assert left.size == (100, 100) and left.getpixel((50, 50))[0] > 200
    assert right.getpixel((50, 50))[2] > 200
    whole = Image.open(io.BytesIO(builder.to_reference_jpeg(buf.getvalue())))
    assert whole.mode == "RGB" and whole.size == (200, 100)


def test_source_list_is_well_formed():
    src = json.loads((ROOT / "catalogs" / "us-mint-coins.sources.json").read_text(encoding="utf-8"))
    ids = [it["id"] for it in src["items"]]
    assert len(ids) == len(set(ids))
    for it in src["items"]:
        assert it["title"] and it["files"]
        for side, files in it["files"].items():
            assert side in ("obverse", "reverse")
            for f in builder._entries(files):
                assert f["file"].startswith("File:")
                assert f.get("crop") in (None, "left", "right")
