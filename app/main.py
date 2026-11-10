"""Web server: JSON API + the drag-and-drop front end.

Configuration (all optional, via environment variables):
  DATA_DIR        where datasets and trained models are stored   (default ./data)
  APP_USERNAME    } set both to protect the whole app with a
  APP_PASSWORD    } browser login prompt (recommended in the cloud)
  MAX_UPLOAD_MB   largest spreadsheet accepted                    (default 25)
  PORT            port used by `python -m app`                    (default 8000)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ml

HERE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", "data")).resolve()
DATASETS, MODELS = DATA_DIR / "datasets", DATA_DIR / "models"
SAMPLES = HERE / "samples"
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024
USERNAME, PASSWORD = os.environ.get("APP_USERNAME", ""), os.environ.get("APP_PASSWORD", "")
ID = re.compile(r"^[0-9a-f]{32}$")

SAMPLE_INFO = {
    "customer-churn": ("Customer churn", "Which customers are likely to cancel? Predict 'cancelled'."),
    "house-prices": ("House prices", "What is a home worth? Predict 'price'."),
    "iris-flowers": ("Iris flowers", "Tell flower species apart by petal size. Predict 'species'."),
}

DATASETS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Democratize ML", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)


@app.middleware("http")
async def login(request: Request, call_next):
    if PASSWORD and request.url.path != "/healthz":
        ok = False
        header = request.headers.get("authorization", "")
        if header.lower().startswith("basic "):
            try:
                user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(user.encode(), USERNAME.encode()) and secrets.compare_digest(
                    pw.encode(), PASSWORD.encode()
                )
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Democratize ML"'})
    return await call_next(request)


@app.exception_handler(ml.UserError)
async def user_error(_: Request, exc: ml.UserError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# ------------------------------------------------------------------ helpers


def _check(id_: str) -> str:
    if not ID.match(id_):
        raise HTTPException(404, "Not found.")
    return id_


def _read_table(data: bytes, filename: str) -> pd.DataFrame:
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(data))
        else:
            # sep=None lets pandas work out commas vs semicolons vs tabs.
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding_errors="replace")
    except Exception:
        raise ml.UserError(
            "That file could not be read as a table. Save your spreadsheet as CSV (or .xlsx) and try again."
        )
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ml.UserError("Two columns have the same name. Rename one of them and try again.")
    if df.shape[1] < 2 or len(df) < ml.MIN_ROWS:
        raise ml.UserError(
            f"The file needs at least 2 columns and {ml.MIN_ROWS} rows; this one has "
            f"{df.shape[1]} column(s) and {len(df)} row(s)."
        )
    if df.shape[1] > 300:
        raise ml.UserError("That file has more than 300 columns, which is more than this tool handles.")
    return df


def _save_dataset(df: pd.DataFrame, name: str) -> dict:
    id_ = uuid.uuid4().hex
    df.to_csv(DATASETS / f"{id_}.csv", index=False)
    (DATASETS / f"{id_}.json").write_text(json.dumps({"id": id_, "name": name}))
    return {"id": id_, "name": name, **ml.profile(df)}


def _load_dataset(id_: str) -> tuple[pd.DataFrame, str]:
    path = DATASETS / f"{_check(id_)}.csv"
    if not path.exists():
        raise HTTPException(404, "That data is no longer available. Please add it again.")
    return pd.read_csv(path), json.loads((DATASETS / f"{id_}.json").read_text())["name"]


def _model_meta(id_: str) -> dict:
    path = MODELS / _check(id_) / "meta.json"
    if not path.exists():
        raise HTTPException(404, "Model not found.")
    return json.loads(path.read_text())


def _model(id_: str):
    meta = _model_meta(id_)
    return joblib.load(MODELS / id_ / "model.joblib"), meta


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"


# ----------------------------------------------------------------- datasets


@app.get("/api/samples")
def samples():
    return [{"id": k, "name": v[0], "description": v[1]} for k, v in SAMPLE_INFO.items()]


@app.post("/api/datasets/sample/{sample}")
def use_sample(sample: str):
    if sample not in SAMPLE_INFO:
        raise HTTPException(404, "Unknown example.")
    return _save_dataset(pd.read_csv(SAMPLES / f"{sample}.csv"), SAMPLE_INFO[sample][0])


@app.post("/api/datasets")
async def upload(file: UploadFile):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise ml.UserError(f"That file is larger than {MAX_UPLOAD // 1024 // 1024} MB.")
    name = Path(file.filename or "data.csv").name
    return _save_dataset(_read_table(data, name), Path(name).stem[:80])


@app.get("/api/datasets/{id_}")
def dataset(id_: str):
    df, name = _load_dataset(id_)
    return {"id": id_, "name": name, **ml.profile(df)}


# ------------------------------------------------------------------- models


class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    ignore: list[str] = []
    steps: list[str] = []
    model: str = "auto"
    task: str = "auto"
    name: str = Field("", max_length=80)


@app.post("/api/models")
def train(req: TrainRequest):
    df, dataset_name = _load_dataset(req.dataset_id)
    pipe, meta = ml.train(df, req.target, req.ignore, req.steps, req.model, req.task)
    id_ = uuid.uuid4().hex
    meta.update(
        id=id_,
        name=req.name.strip() or f"{dataset_name}: predict {req.target}",
        dataset=dataset_name,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    folder = MODELS / id_
    folder.mkdir()
    joblib.dump(pipe, folder / "model.joblib")
    (folder / "meta.json").write_text(json.dumps(meta))
    return meta


@app.get("/api/models")
def models():
    out = []
    for path in MODELS.glob("*/meta.json"):
        m = json.loads(path.read_text())
        out.append({k: m[k] for k in ("id", "name", "task", "target", "model_label", "created", "metrics")})
        out[-1]["metrics"] = {k: v for k, v in m["metrics"].items() if k in ("accuracy", "r2", "mae")}
    return sorted(out, key=lambda m: m["created"], reverse=True)


@app.get("/api/models/{id_}")
def model(id_: str):
    return _model_meta(id_)


@app.delete("/api/models/{id_}")
def delete_model(id_: str):
    _model_meta(id_)
    shutil.rmtree(MODELS / id_)
    return {"deleted": id_}


class PredictRequest(BaseModel):
    rows: list[dict] = Field(..., max_length=10_000)


@app.post("/api/models/{id_}/predict")
def predict(id_: str, req: PredictRequest):
    pipe, meta = _model(id_)
    return {"target": meta["target"], "predictions": ml.predict(pipe, meta, pd.DataFrame(req.rows))}


@app.post("/api/models/{id_}/predict-file")
async def predict_file(id_: str, file: UploadFile):
    """Upload a spreadsheet, get it back with a prediction column added."""
    pipe, meta = _model(id_)
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise ml.UserError(f"That file is larger than {MAX_UPLOAD // 1024 // 1024} MB.")
    try:
        df = (
            pd.read_excel(io.BytesIO(data))
            if (file.filename or "").lower().endswith((".xlsx", ".xls"))
            else pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding_errors="replace")
        )
    except Exception:
        raise ml.UserError("That file could not be read as a table.")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [f["name"] for f in meta["features"] if f["name"] not in df.columns]
    if missing:
        raise ml.UserError("The file is missing these columns: " + ", ".join(missing) + ".")
    results = ml.predict(pipe, meta, df)
    df[f"predicted {meta['target']}"] = [r["prediction"] for r in results]
    if results and "confidence" in results[0]:
        df["confidence"] = [r["confidence"] for r in results]
    return Response(
        df.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{_slug(meta["name"])}-predictions.csv"'},
    )


@app.get("/api/models/{id_}/download")
def download(id_: str):
    meta = _model_meta(id_)
    return FileResponse(
        MODELS / id_ / "model.joblib", media_type="application/octet-stream", filename=f"{_slug(meta['name'])}.joblib"
    )


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
