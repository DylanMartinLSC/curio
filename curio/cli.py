"""Command line: identify, value, serve, and maintain packs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .engine import Curio, default_home
from .pack import PackError, discover_packs, load_pack


def _engine(a) -> Curio:
    def progress(pack_id, done, total):
        print(f"\r  indexing {pack_id}: {done}/{total} images", end="" if done < total else "\n", file=sys.stderr)
    try:
        eng = Curio([Path(p) for p in a.packs] if a.packs else None, a.embedder,
                    Path(a.home) if a.home else None, a.model, progress)
    except ValueError as e:
        sys.exit(f"curio: {e}")
    for note in eng.embedder_notes:
        print(f"{'warning: ' if note['level'] == 'warning' else ''}{note['message']}", file=sys.stderr)
    return eng


def _print_result(res: dict):
    print(f"Verdict: {res['verdict']}   (embedder {res['embedder']})")
    for line in res["explanation"]:
        print(f"  {line}")
    for i, c in enumerate(res["candidates"], 1):
        v = c["valuation"]
        price = (f"{v['estimate']:,.2f} {v['currency']} (range {v['low']:,.2f}-{v['high']:,.2f}, {v['confidence']} confidence)"
                 if v["estimate"] is not None else "no price data")
        print(f"\n{i}. {c['title']}  [{c['key']}]")
        print(f"   similarity {c['similarity']:.3f}  confidence {c['confidence']:.0%}")
        if c["attributes"]:
            print("   " + ", ".join(f"{k}: {val}" for k, val in c["attributes"].items()))
        print(f"   value: {price}")
    if res["candidates"] and res["candidates"][0]["valuation"]["estimate"] is not None:
        print("\nHow the top estimate was made:")
        for line in res["candidates"][0]["valuation"]["explanation"]:
            print(f"  - {line}")


def cmd_identify(a):
    eng = _engine(a)
    res = eng.identify(a.image, a.top, a.domain, a.pack_id, a.grade)
    print(json.dumps(res, indent=2, default=str)) if a.json else _print_result(res)


def cmd_search(a):
    res = _engine(a).search_text(a.text, a.top, a.domain)
    print(json.dumps(res, indent=2, default=str)) if a.json else _print_result(res)


def cmd_value(a):
    v = _engine(a).valuation(a.item, a.grade)
    if a.json:
        print(json.dumps(v, indent=2))
        return
    if v["estimate"] is None:
        print("No estimate.")
    else:
        print(f"{v['estimate']:,.2f} {v['currency']}  (range {v['low']:,.2f}-{v['high']:,.2f}, {v['confidence']} confidence)")
    for line in v["explanation"]:
        print(f"  - {line}")
    print("\n  used  weight  kind   date        grade   price      adj.price  source / note")
    for c in v["comps"]:
        print(f"  {'yes ' if c['used'] else 'no  '}  {c['weight']:.3f}   {c['kind']:<5}  {c['date'] or '-':<10}  "
              f"{c['grade'] or '-':<6}  {c['price']:>9,.2f}  {c['adjusted_price']:>9,.2f}  {c['source']}"
              + (f"  ({c['note']})" if c["note"] else ""))


def cmd_index(a):
    eng = _engine(a)
    eng.build_all()
    for pid in eng.packs:
        print(f"{pid}: {len(eng.index(pid))} vectors ({eng.embedder.key})")


def cmd_packs(a):
    dirs = [Path(p) for p in a.packs] if a.packs else [(Path(a.home) if a.home else default_home()) / "packs"]
    for p in discover_packs([d for d in dirs if d.exists()]):
        s = p.to_summary()
        rec = s["recommended_embedder"]
        rec = f"  [recommends {rec['name']}{' ' + rec['model'] if rec['model'] else ''}]" if rec else ""
        print(f"{s['id']:<24} {s['domain']:<10} {s['items']:>6} items {s['images']:>6} images  v{s['version']}  {s['license']}  {s['name']}{rec}")


def cmd_validate(a):
    ok = True
    for d in a.dirs:
        try:
            p = load_pack(d)
            no_price = sum(1 for i in p.items.values() if not i.prices)
            missing_mult = [g for g in p.grades if g not in p.grade_multipliers]
            print(f"OK   {p.id}: {len(p.items)} items, {no_price} without prices"
                  + (f"; grades without multipliers: {missing_mult}" if missing_mult else ""))
        except PackError as e:
            ok = False
            print(f"FAIL {d}: {e}")
    sys.exit(0 if ok else 1)


def cmd_install(a):
    home = Path(a.home) if a.home else default_home()
    p = load_pack(a.dir)
    dest = home / "packs" / p.id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(p.root, dest)
    print(f"installed {p.id} -> {dest}")


def cmd_new_pack(a):
    root = Path(a.dir)
    if (root / "pack.json").exists():
        sys.exit(f"{root} already has a pack.json")
    (root / "images").mkdir(parents=True, exist_ok=True)
    meta = {
        "id": a.id, "name": a.name, "domain": a.domain, "version": "0.1.0", "license": a.license,
        "currency": a.currency, "rotation_invariant": a.domain == "coin",
        "description": "", "maintainers": [],
        "grades": {"order": [], "multipliers": {}},
    }
    (root / "pack.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (root / "items.jsonl").write_text("", encoding="utf-8")
    print(f"created {root}. Add items with `curio add-item {root} ...`")


def cmd_add_item(a):
    pack = load_pack(a.pack)
    if a.id in pack.items:
        sys.exit(f"item {a.id} already exists")
    images = []
    for i, src in enumerate(a.image):
        src = Path(src)
        dest = pack.root / "images" / f"{a.id}-{i}{src.suffix.lower()}"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dest)
        images.append({"path": dest.relative_to(pack.root).as_posix(), "license": a.image_license, "source": a.image_source})
    attrs = dict(kv.split("=", 1) for kv in a.attr)
    item = {"id": a.id, "title": a.title, "attributes": attrs, "images": images, "prices": []}
    with open(pack.root / "items.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")
    load_pack(pack.root)
    print(f"added {pack.id}:{a.id}")


def cmd_add_price(a):
    pack = load_pack(a.pack)
    lines = (pack.root / "items.jsonl").read_text(encoding="utf-8").splitlines()
    for n, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#") and json.loads(line)["id"] == a.item:
            obj = json.loads(line)
            obj.setdefault("prices", []).append({k: v for k, v in {
                "value": a.value, "currency": a.currency or pack.currency, "grade": a.grade, "date": a.date,
                "source": a.source, "kind": a.kind, "url": a.url}.items() if v not in (None, "")})
            lines[n] = json.dumps(obj)
            break
    else:
        sys.exit(f"no item {a.item} in {pack.id}")
    (pack.root / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    load_pack(pack.root)
    print(f"added price to {pack.id}:{a.item}")


def cmd_eval(a):
    from .evaluate import evaluate
    eng = _engine(a)
    for pid in a.pack_id or list(eng.packs):
        r = evaluate(eng, pid, a.per_image)
        print(f"{pid:<24} {r['embedder']:<40} top-1 {r['top1']:.0%}  top-3 {r['top3']:.0%}  ({r['queries']} simulated photos)")
        for m in r["misses"][:a.show_misses]:
            print(f"    miss: expected {m['expected']}, got {m['got']}")


def cmd_serve(a):
    try:
        import uvicorn
    except ImportError:
        sys.exit("The web UI needs: pip install 'curio[web]'")
    from .server import create_app
    print(f"Curio running at http://{a.host}:{a.port}  (local only; nothing leaves this machine)")
    uvicorn.run(create_app(_engine(a)), host=a.host, port=a.port, log_level="warning")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="curio", description="Local, open-source collectible identifier.")
    ap.add_argument("--packs", action="append",
                    help="a pack, or a folder of packs; repeatable (default: $CURIO_HOME/packs)")
    ap.add_argument("--home", help="data directory (default: $CURIO_HOME or ~/.curio)")
    ap.add_argument("--embedder", default="auto", choices=["auto", "siglip", "dinov2", "simple"],
                    help="auto (default) uses the packs' recommended embedder, or siglip if none is given")
    ap.add_argument("--model", help="Hugging Face model id override (with --embedder), e.g. google/siglip2-base-patch16-384")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("identify", help="identify a photo")
    s.add_argument("image")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--domain", help="coin, stamp, ...")
    s.add_argument("--pack-id", nargs="*")
    s.add_argument("--grade", help="value at this grade (e.g. VF-20)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_identify)

    s = sub.add_parser("search", help="find items by text description (siglip only)")
    s.add_argument("text")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--domain")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("value", help="explain the value of a catalog item")
    s.add_argument("item", help="pack:item")
    s.add_argument("--grade")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_value)

    sub.add_parser("index", help="build/refresh the vector indexes").set_defaults(fn=cmd_index)

    s = sub.add_parser("eval", help="measure identification accuracy on simulated photos of each pack")
    s.add_argument("--pack-id", nargs="*")
    s.add_argument("--per-image", type=int, default=2)
    s.add_argument("--show-misses", type=int, default=5)
    s.set_defaults(fn=cmd_eval)
    sub.add_parser("packs", help="list installed packs").set_defaults(fn=cmd_packs)

    s = sub.add_parser("serve", help="start the local web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("validate", help="check pack directories for errors")
    s.add_argument("dirs", nargs="+")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("install", help="copy a pack into $CURIO_HOME/packs")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_install)

    s = sub.add_parser("new-pack", help="scaffold a new catalog pack")
    s.add_argument("dir")
    s.add_argument("--id", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--domain", required=True, help="coin, stamp, card, comic, antique, ...")
    s.add_argument("--license", default="CC-BY-SA-4.0")
    s.add_argument("--currency", default="USD")
    s.set_defaults(fn=cmd_new_pack)

    s = sub.add_parser("add-item", help="add an item with reference photos to a pack")
    s.add_argument("pack")
    s.add_argument("--id", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--image", nargs="+", required=True)
    s.add_argument("--attr", nargs="*", default=[], help="key=value pairs")
    s.add_argument("--image-license", default="")
    s.add_argument("--image-source", default="")
    s.set_defaults(fn=cmd_add_item)

    s = sub.add_parser("add-price", help="record a price observation for an item")
    s.add_argument("pack")
    s.add_argument("item")
    s.add_argument("--value", type=float, required=True)
    s.add_argument("--source", required=True, help="e.g. 'Heritage Auctions lot 1234'")
    s.add_argument("--kind", default="sale", choices=["sale", "guide", "ask"])
    s.add_argument("--grade")
    s.add_argument("--date", help="YYYY-MM-DD")
    s.add_argument("--currency")
    s.add_argument("--url", default="")
    s.set_defaults(fn=cmd_add_price)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
