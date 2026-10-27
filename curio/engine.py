"""The identification engine: one embedder, many packs, one ranked answer."""

from __future__ import annotations

import io
import os
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from .embed import CALIBRATION, Embedder, create_embedder, embedder_key, family, load_image, preprocess
from .index import LearnedStore, VectorIndex, build_or_load
from .pack import Item, Pack, discover_packs
from .valuation import estimate_value, value_by_grade

ROTATIONS = (0, 45, 90, 135, 180, 225, 270, 315)
AUTO = "auto"
FALLBACK_EMBEDDER = "siglip"


def _describe(name: str, model: str | None) -> str:
    return f"{name} ({model})" if model else name


def resolve_embedder(packs: list[Pack], requested: str, model_id: str | None = None,
                     key: str | None = None) -> tuple[str, str | None, list[dict]]:
    """Pick the embedder to use and explain any mismatch with what the packs recommend.

    requested "auto" follows the packs' recommended_embedder (the most common one if they disagree,
    siglip if none say). An explicit choice is always honoured, with a warning for each pack that
    recommends something else. `key` is passed when an already-built embedder is in use.
    Returns (name, model_id, notes) where notes are {"level": "info"|"warning", "message": ...}.
    """
    recs = {p.id: p.recommended_embedder for p in packs if p.recommended_embedder}
    notes: list[dict] = []
    if requested == AUTO:
        if model_id:
            raise ValueError("--model needs an explicit --embedder (siglip or dinov2)")
        if recs:
            (name, model), _ = Counter(recs.values()).most_common(1)[0]
            backers = [pid for pid, r in recs.items() if r == (name, model)]
            notes.append({"level": "info", "message":
                          f"Using {_describe(name, model)}, recommended by {', '.join(backers)}."})
        else:
            name, model = FALLBACK_EMBEDDER, None
    else:
        name, model = requested, model_id
    chosen = key or embedder_key(name, model)
    for pid, (rname, rmodel) in recs.items():
        if embedder_key(rname, rmodel) != chosen:
            flag = f"--embedder {rname}" + (f" --model {rmodel}" if rmodel else "")
            notes.append({"level": "warning", "message":
                          f"Pack '{pid}' recommends {_describe(rname, rmodel)}, but {chosen} is in use; "
                          f"identification in that pack may be less accurate. Use {flag} for it."})
    return name, model, notes


def default_home() -> Path:
    return Path(os.environ.get("CURIO_HOME", Path.home() / ".curio"))


def rotated_views(img: Image.Image, angles=ROTATIONS) -> list[Image.Image]:
    base = preprocess(img)
    fill = base.getpixel((0, 0))
    return [base if a == 0 else base.rotate(a, resample=Image.BICUBIC, fillcolor=fill) for a in angles]


class Curio:
    def __init__(self, pack_dirs: list[Path] | None = None, embedder: Embedder | str = AUTO,
                 home: Path | None = None, model_id: str | None = None, progress=None):
        self.home = Path(home) if home else default_home()
        dirs = list(pack_dirs or []) or [self.home / "packs"]
        self.packs: dict[str, Pack] = {p.id: p for p in discover_packs([d for d in dirs if Path(d).exists()])}
        if isinstance(embedder, str):
            name, model, self.embedder_notes = resolve_embedder(list(self.packs.values()), embedder.lower(), model_id)
            self._embedder_spec = (name, model)
            self._embedder = None
        else:
            _, _, self.embedder_notes = resolve_embedder(list(self.packs.values()), "", key=embedder.key)
            self._embedder = embedder
        self.learned = LearnedStore(self.home)
        self.progress = progress
        self._indexes: dict[str, VectorIndex] = {}

    @property
    def embedder(self) -> Embedder:
        """Loaded on first use, so valuation-only commands don't pay for loading a model."""
        if self._embedder is None:
            self._embedder = create_embedder(*self._embedder_spec)
        return self._embedder

    @property
    def calibration(self) -> dict:
        return CALIBRATION[family(self.embedder)]

    # ---- lookup helpers -------------------------------------------------
    def item(self, key: str) -> tuple[Pack, Item]:
        pack_id, _, item_id = key.partition(":")
        pack = self.packs.get(pack_id)
        if not pack or item_id not in pack.items:
            raise KeyError(f"unknown item '{key}'")
        return pack, pack.items[item_id]

    def index(self, pack_id: str) -> VectorIndex:
        if pack_id not in self._indexes:
            prog = (lambda d, t: self.progress(pack_id, d, t)) if self.progress else None
            self._indexes[pack_id] = build_or_load(self.packs[pack_id], self.embedder, self.home, self.learned, prog)
        return self._indexes[pack_id]

    def build_all(self):
        for pid in self.packs:
            self.index(pid)

    def _select(self, domain: str | None, pack_ids: list[str] | None) -> list[Pack]:
        packs = [p for p in self.packs.values()
                 if (not domain or p.domain == domain) and (not pack_ids or p.id in pack_ids)]
        if not packs:
            raise ValueError("no packs match; install a pack or relax the domain/pack filter")
        return packs

    # ---- identification --------------------------------------------------
    def identify(self, image, top_k: int = 5, domain: str | None = None, pack_ids: list[str] | None = None,
                 grade: str | None = None) -> dict:
        img = image if isinstance(image, Image.Image) else load_image(image)
        packs = self._select(domain, pack_ids)

        upright = self.embedder.embed_images([img])
        rotated = None
        hits = []
        for pack in packs:
            if pack.rotation_invariant:
                if rotated is None:
                    rotated = self.embedder.embed_images(rotated_views(img))
                q = rotated
            else:
                q = upright
            for h in self.index(pack.id).search(q, top_k):
                hits.append(h)
        hits.sort(key=lambda h: -h["similarity"])
        hits = hits[:top_k]
        return self._package(hits, grade, query_kind="image")

    def search_text(self, text: str, top_k: int = 5, domain: str | None = None) -> dict:
        if not self.embedder.supports_text:
            raise ValueError(f"embedder {self.embedder.key} cannot search by text; use siglip")
        q = self.embedder.embed_texts([text])
        hits = []
        for pack in self._select(domain, None):
            hits += self.index(pack.id).search(q, top_k)
        hits.sort(key=lambda h: -h["similarity"])
        return self._package(hits[:top_k], None, query_kind="text")

    def _package(self, hits: list[dict], grade: str | None, query_kind: str) -> dict:
        cal = self.calibration
        sims = np.array([h["similarity"] for h in hits]) if hits else np.zeros(0)
        probs = np.exp((sims - sims.max()) / cal["temperature"]) if len(sims) else sims
        probs = probs / probs.sum() if len(sims) else probs

        candidates = []
        for h, p in zip(hits, probs):
            pack, item = self.item(h["item_key"])
            g = grade if (grade and grade in pack.grades) else None
            val = estimate_value(item, pack, g)
            candidates.append({
                "key": item.key, "title": item.title, "pack": pack.id, "pack_name": pack.name,
                "domain": pack.domain, "attributes": item.attributes, "notes": item.notes,
                "similarity": round(h["similarity"], 4), "confidence": round(float(p), 4),
                "matched_image": h["matched_image"], "matched_learned": h["matched_learned"],
                "grades": pack.grades, "valuation": val.to_dict(),
                "value_by_grade": value_by_grade(item, pack),
            })

        top = candidates[0] if candidates else None
        if query_kind == "text":
            verdict = "ranked" if top else "no_match"
        elif not top or top["similarity"] < cal["min_similarity"]:
            verdict = "no_match"
        elif top["confidence"] >= 0.6:
            verdict = "match"
        else:
            verdict = "ambiguous"
        return {"verdict": verdict, "embedder": self.embedder.key, "candidates": candidates,
                "explanation": self._explain(verdict, candidates)}

    def _explain(self, verdict: str, cands: list[dict]) -> list[str]:
        if verdict == "no_match":
            best = f" (best similarity {cands[0]['similarity']:.2f}, threshold {self.calibration['min_similarity']:.2f})" if cands else ""
            return [f"Nothing in the installed packs looks close enough{best}. It may not be catalogued yet; "
                    "try a straighter, evenly lit photo, or add it to a pack."]
        if verdict == "ranked":
            return ["Ranked by similarity between your text and the reference photos."]
        top = cands[0]
        lines = [f"Closest reference photo similarity {top['similarity']:.2f} "
                 f"({'one of your confirmed photos' if top['matched_learned'] else 'pack reference photo'})."]
        if len(cands) > 1:
            gap = top["similarity"] - cands[1]["similarity"]
            lines.append(f"Margin over the runner-up ({cands[1]['title']}): {gap:.3f}. "
                         "Confidence is a softmax over candidate similarities, so a small margin means a close call.")
        if verdict == "ambiguous":
            lines.append("Candidates are close: compare the reference photos (dates, mint marks, perforations) "
                         "and confirm the right one; confirming teaches Curio for next time.")
        return lines

    # ---- learning from the user -----------------------------------------
    def confirm(self, image_bytes: bytes, item_key: str) -> Path:
        """User says this photo is item_key: keep it locally as an extra reference."""
        pack, item = self.item(item_key)
        Image.open(io.BytesIO(image_bytes)).verify()
        path = self.learned.add(pack.id, item.id, image_bytes)
        self._indexes.pop(pack.id, None)  # rebuild on next use
        return path

    def valuation(self, item_key: str, grade: str | None = None, as_of: date | None = None) -> dict:
        pack, item = self.item(item_key)
        return estimate_value(item, pack, grade, as_of).to_dict()
