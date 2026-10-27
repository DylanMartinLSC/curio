"""Local web UI (FastAPI). Binds to 127.0.0.1 by default; photos never leave the machine."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

from .embed import load_image
from .engine import Curio

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 25 * 1024 * 1024


class ConfirmBody(BaseModel):
    upload_id: str
    item_key: str


class CollectBody(BaseModel):
    item_key: str
    upload_id: str | None = None
    note: str = ""


class Collection:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        # `grade` is no longer written; the column stays so collection files from older versions still open.
        self.db.execute("""create table if not exists items (
            id integer primary key, item_key text not null, grade text, photo blob,
            note text, added text not null)""")
        self.db.commit()

    def add(self, item_key, photo, note) -> int:
        with self.lock:
            cur = self.db.execute("insert into items(item_key, photo, note, added) values (?,?,?,?)",
                                  (item_key, photo, note, datetime.now().isoformat(timespec="seconds")))
            self.db.commit()
            return cur.lastrowid

    def all(self):
        with self.lock:
            return self.db.execute("select id, item_key, note, added, photo is not null from items order by id desc").fetchall()

    def photo(self, rid):
        with self.lock:
            row = self.db.execute("select photo from items where id=?", (rid,)).fetchone()
        return row[0] if row else None

    def delete(self, rid):
        with self.lock:
            self.db.execute("delete from items where id=?", (rid,))
            self.db.commit()


def create_app(engine: Curio) -> FastAPI:
    # Load the model and indexes up front: lazy loading from concurrent request threads is racy.
    engine.build_all()
    app = FastAPI(title="Curio", docs_url="/api/docs")
    uploads: OrderedDict[str, bytes] = OrderedDict()  # recent uploads, for confirm / add-to-collection
    collection = Collection(engine.home / "collection.sqlite3")
    lock = threading.Lock()  # the embedder is not guaranteed thread-safe

    def image_token(path: str) -> str:
        return hashlib.sha1(str(path).encode()).hexdigest()[:20]

    def allowed_images() -> dict[str, Path]:
        out = {}
        for pack in engine.packs.values():
            for item in pack.items.values():
                for img in item.images:
                    out[image_token(img.path)] = img.path
            for _, p in engine.learned.resolved(pack):
                out[image_token(p)] = p
        return out

    def decorate(res: dict) -> dict:
        for c in res["candidates"]:
            _, item = engine.item(c["key"])
            c["matched_image_url"] = f"/img/{image_token(c['matched_image'])}"
            c["images"] = [{"url": f"/img/{image_token(i.path)}", "side": i.side, "license": i.license,
                            "source": i.source} for i in item.images]
            del c["matched_image"]
        return res

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/img/{token}")
    def img(token: str):
        p = allowed_images().get(token)
        if not p:
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/packs")
    def packs():
        return {"embedder": engine.embedder.key, "supports_text": engine.embedder.supports_text,
                "notes": engine.embedder_notes,
                "packs": [p.to_summary() for p in engine.packs.values()]}

    @app.post("/api/identify")
    async def identify(file: UploadFile = File(...), domain: str = Form(""), top: int = Form(5)):
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "image too large")
        try:
            image = load_image(io.BytesIO(data))
        except Exception:
            raise HTTPException(400, "could not read that image")
        uid = hashlib.sha256(data).hexdigest()[:24]
        uploads[uid] = data
        while len(uploads) > 50:
            uploads.popitem(last=False)
        try:
            with lock:
                res = engine.identify(image, top_k=max(1, min(top, 20)), domain=domain or None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return decorate(res) | {"upload_id": uid}

    @app.get("/api/search")
    def search(q: str, domain: str = "", top: int = 8):
        try:
            with lock:
                return decorate(engine.search_text(q, max(1, min(top, 20)), domain or None))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/confirm")
    def confirm(body: ConfirmBody):
        data = uploads.get(body.upload_id)
        if data is None:
            raise HTTPException(404, "upload expired; identify the photo again")
        try:
            with lock:
                engine.confirm(data, body.item_key)
        except KeyError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.get("/api/collection")
    def list_collection():
        rows = []
        for rid, key, note, added, has_photo in collection.all():
            try:
                title = engine.item(key)[1].title
            except KeyError:
                title = f"{key} (pack not installed)"
            rows.append({"id": rid, "item_key": key, "title": title, "note": note, "added": added,
                         "photo_url": f"/api/collection/{rid}/photo" if has_photo else None})
        return {"items": rows}

    @app.post("/api/collection")
    def add_collection(body: CollectBody):
        try:
            engine.item(body.item_key)
        except KeyError as e:
            raise HTTPException(400, str(e))
        photo = None
        if body.upload_id and body.upload_id in uploads:
            im = Image.open(io.BytesIO(uploads[body.upload_id])).convert("RGB")
            im.thumbnail((800, 800))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            photo = buf.getvalue()
        return {"id": collection.add(body.item_key, photo, body.note)}

    @app.get("/api/collection/{rid}/photo")
    def collection_photo(rid: int):
        data = collection.photo(rid)
        if not data:
            raise HTTPException(404)
        return Response(data, media_type="image/jpeg")

    @app.delete("/api/collection/{rid}")
    def delete_collection(rid: int):
        collection.delete(rid)
        return {"ok": True}

    return app
