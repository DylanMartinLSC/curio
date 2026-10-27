"""Catalog packs: the community-maintained data Curio identifies against.

A pack is a plain directory, easy to review in a pull request:

    my-pack/
      pack.json      metadata, grade scale, defaults
      items.jsonl    one catalog item per line (title, attributes, images, price observations)
      images/        reference photos (any path relative to the pack dir works)

Packs ship images and metadata, never model vectors, so any embedder can index them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRICE_KINDS = ("sale", "guide", "ask")


class PackError(ValueError):
    pass


@dataclass
class ImageRef:
    path: Path
    side: str = ""  # obverse / reverse / front / back / detail ...
    license: str = ""
    source: str = ""


@dataclass
class PriceObservation:
    value: float
    currency: str
    grade: str | None
    date: str | None  # ISO yyyy-mm-dd
    source: str
    kind: str = "sale"  # sale (realized), guide (catalog/guide value), ask (listing price)
    url: str = ""


@dataclass
class Item:
    id: str
    title: str
    attributes: dict[str, Any]
    images: list[ImageRef]
    prices: list[PriceObservation]
    notes: str = ""
    pack_id: str = ""

    @property
    def key(self) -> str:
        return f"{self.pack_id}:{self.id}"


@dataclass
class Pack:
    root: Path
    id: str
    name: str
    domain: str
    version: str
    license: str
    currency: str
    grades: list[str]
    grade_multipliers: dict[str, float]
    rotation_invariant: bool
    recommended_embedder: tuple[str, str | None] | None = None  # (name, model id or None for the default)
    description: str = ""
    maintainers: list[str] = field(default_factory=list)
    items: dict[str, Item] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Changes whenever items or reference images change, so stale indexes get rebuilt."""
        h = hashlib.sha256()
        h.update((self.root / "pack.json").read_bytes())
        h.update((self.root / "items.jsonl").read_bytes())
        for item in self.items.values():
            for img in item.images:
                st = img.path.stat()
                h.update(f"{img.path}:{st.st_size}:{st.st_mtime_ns}".encode())
        return h.hexdigest()[:16]

    def to_summary(self) -> dict:
        return {
            "id": self.id, "name": self.name, "domain": self.domain, "version": self.version,
            "license": self.license, "currency": self.currency, "items": len(self.items),
            "images": sum(len(i.images) for i in self.items.values()),
            "description": self.description,
            "recommended_embedder": ({"name": self.recommended_embedder[0], "model": self.recommended_embedder[1]}
                                     if self.recommended_embedder else None),
        }


def _require(obj: dict, key: str, where: str):
    if key not in obj or obj[key] in (None, ""):
        raise PackError(f"{where}: missing required field '{key}'")
    return obj[key]


def load_pack(root: str | Path) -> Pack:
    root = Path(root).resolve()
    meta_path = root / "pack.json"
    items_path = root / "items.jsonl"
    if not meta_path.is_file():
        raise PackError(f"{root}: no pack.json")
    if not items_path.is_file():
        raise PackError(f"{root}: no items.jsonl")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    where = str(meta_path)

    grades_cfg = meta.get("grades") or {}
    grades = list(grades_cfg.get("order", []))
    mults = {k: float(v) for k, v in (grades_cfg.get("multipliers") or {}).items()}
    for g in mults:
        if g not in grades:
            raise PackError(f"{where}: grade multiplier for unknown grade '{g}'")
    for m in mults.values():
        if m <= 0:
            raise PackError(f"{where}: grade multipliers must be positive")

    pack = Pack(
        root=root,
        id=_require(meta, "id", where),
        name=_require(meta, "name", where),
        domain=_require(meta, "domain", where),
        version=str(meta.get("version", "0.0.0")),
        license=_require(meta, "license", where),
        currency=meta.get("currency", "USD"),
        grades=grades,
        grade_multipliers=mults,
        rotation_invariant=bool(meta.get("rotation_invariant", False)),
        recommended_embedder=_parse_recommended(meta.get("recommended_embedder"), where),
        description=meta.get("description", ""),
        maintainers=list(meta.get("maintainers", [])),
    )
    if ":" in pack.id:
        raise PackError(f"{where}: pack id may not contain ':'")

    for lineno, line in enumerate(items_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        where = f"{items_path}:{lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise PackError(f"{where}: invalid JSON ({e})") from e
        item = _parse_item(raw, pack, where)
        if item.id in pack.items:
            raise PackError(f"{where}: duplicate item id '{item.id}'")
        pack.items[item.id] = item
    return pack


def _parse_recommended(raw, where: str) -> tuple[str, str | None] | None:
    """`"dinov2"` or `{"name": "dinov2", "model": "facebook/dinov2-small"}`; the model is optional."""
    from .embed import EMBEDDER_NAMES
    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        raw = {"name": raw}
    if not isinstance(raw, dict) or raw.get("name", "").lower() not in EMBEDDER_NAMES:
        raise PackError(f"{where}: recommended_embedder must name one of {EMBEDDER_NAMES}")
    return raw["name"].lower(), raw.get("model") or None


def _parse_item(raw: dict, pack: Pack, where: str) -> Item:
    images = []
    for img in raw.get("images", []):
        if isinstance(img, str):
            img = {"path": img}
        p = (pack.root / _require(img, "path", where)).resolve()
        if pack.root not in p.parents:
            raise PackError(f"{where}: image path escapes the pack directory: {img['path']}")
        if not p.is_file():
            raise PackError(f"{where}: image not found: {img['path']}")
        images.append(ImageRef(p, img.get("side", ""), img.get("license", ""), img.get("source", "")))
    if not images:
        raise PackError(f"{where}: item needs at least one image")

    prices = []
    for pr in raw.get("prices", []):
        kind = pr.get("kind", "sale")
        if kind not in PRICE_KINDS:
            raise PackError(f"{where}: price kind must be one of {PRICE_KINDS}")
        grade = pr.get("grade")
        if grade is not None and pack.grades and grade not in pack.grades:
            raise PackError(f"{where}: unknown grade '{grade}' (pack grades: {pack.grades})")
        value = float(_require(pr, "value", where))
        if value <= 0:
            raise PackError(f"{where}: price value must be positive")
        prices.append(PriceObservation(
            value=value,
            currency=pr.get("currency", pack.currency),
            grade=grade,
            date=pr.get("date"),
            source=_require(pr, "source", where),
            kind=kind,
            url=pr.get("url", ""),
        ))

    return Item(
        id=str(_require(raw, "id", where)),
        title=_require(raw, "title", where),
        attributes=dict(raw.get("attributes", {})),
        images=images,
        prices=prices,
        notes=raw.get("notes", ""),
        pack_id=pack.id,
    )


def discover_packs(dirs: list[Path]) -> list[Pack]:
    """Load every pack found directly in (or as) the given directories."""
    packs: dict[str, Pack] = {}
    for d in dirs:
        d = Path(d)
        candidates = [d] if (d / "pack.json").is_file() else sorted(p for p in d.glob("*") if (p / "pack.json").is_file())
        for c in candidates:
            pack = load_pack(c)
            if pack.id in packs:
                raise PackError(f"two packs share the id '{pack.id}': {packs[pack.id].root} and {pack.root}")
            packs[pack.id] = pack
    return list(packs.values())
