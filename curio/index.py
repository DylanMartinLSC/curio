"""Vector index over a pack's reference images (plus images the user confirmed locally).

Exact cosine search with numpy. A 100k-image catalog at 768 dims is ~300 MB and searches in
tens of milliseconds, which covers community catalogs comfortably without an ANN dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import Embedder, load_image
from .pack import Pack


@dataclass
class IndexEntry:
    item_key: str
    image: str
    learned: bool  # True for photos the user confirmed, not shipped by the pack


class LearnedStore:
    """Photos the user confirmed ("yes, this is X"). Stored per pack, used as extra references."""

    def __init__(self, home: Path):
        self.root = Path(home) / "learned"

    def _dir(self, pack_id: str) -> Path:
        return self.root / pack_id

    def entries(self, pack_id: str) -> list[dict]:
        f = self._dir(pack_id) / "learned.jsonl"
        if not f.is_file():
            return []
        return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]

    def add(self, pack_id: str, item_id: str, image_bytes: bytes, suffix: str = ".jpg") -> Path:
        d = self._dir(pack_id)
        d.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(image_bytes).hexdigest()[:20] + suffix
        (d / name).write_bytes(image_bytes)
        known = {(e["item"], e["image"]) for e in self.entries(pack_id)}
        if (item_id, name) not in known:
            with open(d / "learned.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"item": item_id, "image": name}) + "\n")
        return d / name

    def resolved(self, pack: Pack) -> list[tuple[str, Path]]:
        d = self._dir(pack.id)
        return [(e["item"], d / e["image"]) for e in self.entries(pack.id)
                if e["item"] in pack.items and (d / e["image"]).is_file()]

    def fingerprint(self, pack_id: str) -> str:
        f = self._dir(pack_id) / "learned.jsonl"
        return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.is_file() else "none"


class VectorIndex:
    def __init__(self, vectors: np.ndarray, entries: list[IndexEntry]):
        self.vectors = vectors.astype(np.float32)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def search(self, queries: np.ndarray, top_k: int = 5) -> list[dict]:
        """queries: (T, D) — several views of one object (e.g. rotations). Score = best view x best image per item."""
        if len(self) == 0:
            return []
        sims = (self.vectors @ queries.T).max(axis=1)
        best: dict[str, tuple[float, int]] = {}
        for i, s in enumerate(sims):
            k = self.entries[i].item_key
            if k not in best or s > best[k][0]:
                best[k] = (float(s), i)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top_k]
        return [{"item_key": k, "similarity": s, "matched_image": self.entries[i].image,
                 "matched_learned": self.entries[i].learned} for k, (s, i) in ranked]


def build_or_load(pack: Pack, embedder: Embedder, home: Path, learned: LearnedStore | None = None,
                  progress=None) -> VectorIndex:
    learned = learned or LearnedStore(home)
    cache = Path(home) / "index" / pack.id / embedder.key
    fp = f"{pack.fingerprint()}-{learned.fingerprint(pack.id)}"
    meta_path, vec_path = cache / "meta.json", cache / "vectors.npy"
    reusable: dict[str, np.ndarray] = {}
    if meta_path.is_file() and vec_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old = np.load(vec_path)
        entries = [IndexEntry(**{k: e[k] for k in ("item_key", "image", "learned")}) for e in meta["entries"]]
        if meta.get("fingerprint") == fp:
            return VectorIndex(old, entries)
        # Incremental rebuild: only embed images that are new or changed.
        reusable = {e["sig"]: old[i] for i, e in enumerate(meta["entries"]) if "sig" in e}

    work: list[tuple[str, Path, bool]] = []
    for item in pack.items.values():
        work += [(item.key, img.path, False) for img in item.images]
    work += [(f"{pack.id}:{item_id}", p, True) for item_id, p in learned.resolved(pack)]
    sigs = [_image_sig(p) for _, p, _ in work]

    todo = [i for i, s in enumerate(sigs) if s not in reusable]
    fresh: dict[int, np.ndarray] = {}
    batch = 32
    for b in range(0, len(todo), batch):
        idx = todo[b:b + batch]
        for i, v in zip(idx, embedder.embed_images([load_image(work[i][1]) for i in idx])):
            fresh[i] = v
        if progress:
            progress(min(b + batch, len(todo)), len(todo))
    vectors = (np.stack([fresh[i] if i in fresh else reusable[s] for i, s in enumerate(sigs)])
               if work else np.zeros((0, 1), np.float32))
    entries = [IndexEntry(k, str(p), l) for k, p, l in work]

    cache.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps({"fingerprint": fp, "embedder": embedder.key,
                                     "entries": [{**e.__dict__, "sig": s} for e, s in zip(entries, sigs)]}),
                         encoding="utf-8")
    return VectorIndex(vectors, entries)


def _image_sig(p: Path) -> str:
    st = p.stat()
    return f"{p}:{st.st_size}:{st.st_mtime_ns}"
