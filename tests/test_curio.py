import importlib.util
import io
import json
import random
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest
from PIL import Image

from curio.engine import Curio
from curio.evaluate import simulate_photo
from curio.pack import PackError, load_pack
from curio.valuation import estimate_value

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def packs_dir(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_demo", ROOT / "scripts" / "make_demo_packs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("packs")
    mod.make_coins(out / "demo-coins")
    mod.make_stamps(out / "demo-stamps")
    return out


@pytest.fixture
def engine(packs_dir, tmp_path):
    return Curio([packs_dir], "simple", home=tmp_path)


def test_packs_load(packs_dir):
    coins = load_pack(packs_dir / "demo-coins")
    assert coins.domain == "coin" and len(coins.items) == 10 and coins.rotation_invariant
    assert "VF-20" in coins.grade_multipliers


def test_pack_validation_errors(packs_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(packs_dir / "demo-stamps", bad)
    line = json.loads((bad / "items.jsonl").read_text().splitlines()[0])
    line["images"] = [{"path": "../../etc/passwd"}]
    (bad / "items.jsonl").write_text(json.dumps(line) + "\n")
    with pytest.raises(PackError, match="escapes"):
        load_pack(bad)
    line["images"] = [{"path": "images/flower-1c.jpg"}]
    line["prices"] = [{"value": 3, "source": "x", "grade": "Mint-ish"}]
    (bad / "items.jsonl").write_text(json.dumps(line) + "\n")
    with pytest.raises(PackError, match="unknown grade"):
        load_pack(bad)


def test_identifies_exact_reference(engine, packs_dir):
    res = engine.identify(packs_dir / "demo-stamps/images/bird-10c.jpg")
    assert res["verdict"] in ("match", "ambiguous")
    assert res["candidates"][0]["key"] == "demo-stamps:bird-10c"
    assert res["candidates"][0]["valuation"]["explanation"]


def test_identifies_simulated_photo_top3(engine, packs_dir):
    rnd = random.Random(1)
    hits = 0
    for name, key in [("ship-5c-red", "demo-stamps:ship-5c-red"), ("mountain-20c", "demo-stamps:mountain-20c"),
                      ("crown-1d", "demo-stamps:crown-1d")]:
        q = simulate_photo(Image.open(packs_dir / f"demo-stamps/images/{name}.jpg"), rnd, rotate=False)
        hits += key in [c["key"] for c in engine.identify(q, top_k=3, domain="stamp")["candidates"]]
    assert hits >= 2  # the simple embedder is weak; neural embedders are evaluated via `curio eval`


def test_domain_filter(engine, packs_dir):
    res = engine.identify(packs_dir / "demo-stamps/images/bird-10c.jpg", domain="coin")
    assert all(c["domain"] == "coin" for c in res["candidates"])


def test_rotated_coin_matches(engine, packs_dir):
    img = Image.open(packs_dir / "demo-coins/images/25-cent-1912-obverse.jpg").convert("RGB")
    rotated = img.rotate(90, fillcolor=img.getpixel((0, 0)))
    res = engine.identify(rotated, top_k=3)
    assert res["candidates"][0]["key"] == "demo-coins:25-cent-1912"


def test_index_cache_and_incremental_learning(engine, packs_dir, tmp_path):
    engine.build_all()
    assert (tmp_path / "index" / "demo-coins" / "simple-v1" / "vectors.npy").is_file()
    n = len(engine.index("demo-stamps"))
    buf = io.BytesIO()
    simulate_photo(Image.open(packs_dir / "demo-stamps/images/crown-1d.jpg"), random.Random(3), False).save(buf, "JPEG")
    engine.confirm(buf.getvalue(), "demo-stamps:crown-1d")
    assert len(engine.index("demo-stamps")) == n + 1
    # The learned photo now matches itself exactly.
    res = engine.identify(Image.open(io.BytesIO(buf.getvalue())), top_k=1, domain="stamp")
    assert res["candidates"][0]["key"] == "demo-stamps:crown-1d" and res["candidates"][0]["matched_learned"]


def test_valuation_is_explained_and_robust(packs_dir):
    pack = load_pack(packs_dir / "demo-coins")
    item = pack.items["1-cent-1938"]
    v = estimate_value(item, pack, "VF-20", as_of=date(2026, 10, 1))
    assert v.estimate is not None and v.low <= v.estimate <= v.high
    outlier = [c for c in v.comps if c.price == 95.0][0]
    assert not outlier.used and "outlier" in outlier.note
    assert any("weighted median" in line for line in v.explanation)
    assert all(c.grade_factor is not None for c in v.comps if c.used)


def test_grade_adjustment_math(packs_dir):
    pack = load_pack(packs_dir / "demo-coins")
    item = pack.items["5-cent-1921"]
    v = estimate_value(item, pack, "MS-65", as_of=date(2026, 10, 1))
    for c in v.comps:
        if c.grade and c.used:
            expected = c.price * pack.grade_multipliers["MS-65"] / pack.grade_multipliers[c.grade]
            assert c.adjusted_price == pytest.approx(expected)


def test_no_prices_gives_no_estimate(packs_dir):
    pack = load_pack(packs_dir / "demo-stamps")
    item = pack.items["bird-10c"]
    item.prices = []
    v = estimate_value(item, pack)
    assert v.estimate is None and v.confidence == "none"


def test_thin_data_is_low_confidence(packs_dir):
    pack = load_pack(packs_dir / "demo-coins")
    v = estimate_value(pack.items["2-crown-1901"], pack, "EF-40", as_of=date(2026, 10, 1))
    assert v.confidence in ("low", "medium")


def test_cli_contributor_workflow(tmp_path, packs_dir, capsys):
    from curio.cli import main
    pack = tmp_path / "my-pack"
    main(["new-pack", str(pack), "--id", "my-stamps", "--name", "My stamps", "--domain", "stamp"])
    main(["add-item", str(pack), "--id", "a1", "--title", "Test stamp",
          "--image", str(packs_dir / "demo-stamps/images/bird-10c.jpg"), "--attr", "country=Examplia"])
    main(["add-price", str(pack), "a1", "--value", "4.5", "--source", "my notes", "--date", "2026-01-02"])
    p = load_pack(pack)
    assert p.items["a1"].prices[0].value == 4.5
    with pytest.raises(SystemExit) as e:
        main(["validate", str(pack)])
    assert e.value.code == 0


def test_web_api(engine, packs_dir):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from curio.server import create_app
    client = TestClient(create_app(engine))
    assert client.get("/").status_code == 200
    assert {p["id"] for p in client.get("/api/packs").json()["packs"]} == {"demo-coins", "demo-stamps"}
    data = (packs_dir / "demo-stamps/images/flower-1c.jpg").read_bytes()
    res = client.post("/api/identify", files={"file": ("x.jpg", data, "image/jpeg")}).json()
    top = res["candidates"][0]
    assert top["key"] == "demo-stamps:flower-1c"
    assert client.get(top["matched_image_url"]).status_code == 200
    assert client.get("/img/not-a-real-token").status_code == 404
    v = client.get("/api/value", params={"item": top["key"], "grade": "Superb"}).json()
    assert v["target_grade"] == "Superb"
    assert client.post("/api/confirm", json={"upload_id": res["upload_id"], "item_key": top["key"]}).json()["ok"]
    client.post("/api/collection", json={"item_key": top["key"], "grade": "Fine", "upload_id": res["upload_id"]})
    coll = client.get("/api/collection").json()
    assert len(coll["items"]) == 1 and coll["items"][0]["photo_url"]
    assert client.post("/api/identify", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400


def _with_recommendation(src: Path, dest: Path, rec) -> Path:
    shutil.copytree(src, dest)
    meta = json.loads((dest / "pack.json").read_text())
    meta["recommended_embedder"] = rec
    (dest / "pack.json").write_text(json.dumps(meta))
    return dest


def test_auto_embedder_follows_pack_recommendation(packs_dir, tmp_path):
    d = _with_recommendation(packs_dir / "demo-stamps", tmp_path / "p" / "stamps", "simple")
    eng = Curio([d], home=tmp_path)  # default is auto
    assert eng.embedder.key == "simple-v1"
    assert [n["level"] for n in eng.embedder_notes] == ["info"]


def test_auto_without_recommendations_uses_siglip(packs_dir, tmp_path):
    from curio.engine import resolve_embedder
    packs = [load_pack(packs_dir / "demo-stamps")]
    assert resolve_embedder(packs, "auto") == ("siglip", None, [])


def test_explicit_mismatch_warns_but_is_honoured(packs_dir, tmp_path):
    d = _with_recommendation(packs_dir / "demo-coins", tmp_path / "p" / "coins",
                             {"name": "dinov2", "model": "facebook/dinov2-small"})
    eng = Curio([d], "simple", home=tmp_path)
    assert eng.embedder.key == "simple-v1"
    (warn,) = eng.embedder_notes
    assert warn["level"] == "warning" and "--embedder dinov2" in warn["message"]
    # Same model spelled with and without the default model id is not a mismatch.
    from curio.engine import resolve_embedder
    assert resolve_embedder([load_pack(d)], "dinov2")[2] == []


def test_conflicting_recommendations_pick_majority_and_warn(packs_dir, tmp_path):
    from curio.engine import resolve_embedder
    a = load_pack(_with_recommendation(packs_dir / "demo-coins", tmp_path / "a", "dinov2"))
    b = load_pack(_with_recommendation(packs_dir / "demo-stamps", tmp_path / "b", "dinov2"))
    c = load_pack(_with_recommendation(packs_dir / "demo-stamps", tmp_path / "c", "siglip"))
    c.id = "other"
    name, model, notes = resolve_embedder([a, b, c], "auto")
    assert (name, model) == ("dinov2", None)
    assert [n["level"] for n in notes] == ["info", "warning"] and "'other'" in notes[1]["message"]
    with pytest.raises(ValueError, match="--model"):
        resolve_embedder([a], "auto", "some/model")


def test_invalid_recommendation_rejected(packs_dir, tmp_path):
    d = _with_recommendation(packs_dir / "demo-stamps", tmp_path / "bad", "clip")
    with pytest.raises(PackError, match="recommended_embedder"):
        load_pack(d)
