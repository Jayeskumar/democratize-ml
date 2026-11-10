import io
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp()

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402

client = TestClient(main.app)


def sample(name):
    r = client.post(f"/api/datasets/sample/{name}")
    assert r.status_code == 200, r.text
    return r.json()


def train(**body):
    return client.post("/api/models", json=body)


def test_page_and_health():
    assert client.get("/healthz").json() == {"status": "ok"}
    assert "Democratize ML" in client.get("/").text
    assert len(client.get("/api/samples").json()) == 3


def test_profile_flags_ids_and_task():
    d = sample("customer-churn")
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["customer_id"]["id_like"]
    assert cols["cancelled"]["task"] == "classification"
    assert cols["support_calls"]["missing"] > 0
    assert sample("house-prices")["columns"][-1]["task"] == "regression"


def test_classification_end_to_end():
    d = sample("customer-churn")
    r = train(dataset_id=d["id"], target="cancelled", steps=["dedupe", "balance"])
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["task"] == "classification" and m["leaderboard"]
    assert m["metrics"]["accuracy"] > m["metrics"]["baseline"]
    assert "customer_id" not in [f["name"] for f in m["features"]]
    assert abs(sum(i["share"] for i in m["importance"]) - 1) < 0.01

    row = {f["name"]: f["default"] for f in m["features"]}
    p = client.post(f"/api/models/{m['id']}/predict", json={"rows": [row, {}]}).json()["predictions"]
    assert p[0]["prediction"] in ("Yes", "No") and 0.5 <= p[0]["confidence"] <= 1
    assert p[1]["prediction"] in ("Yes", "No")  # a completely blank row still gets an answer

    # The downloaded file works with nothing but scikit-learn.
    pipe = joblib.load(io.BytesIO(client.get(f"/api/models/{m['id']}/download").content))
    assert pipe.predict(pd.DataFrame([row]))[0] == p[0]["prediction"]

    assert any(x["id"] == m["id"] for x in client.get("/api/models").json())
    assert client.delete(f"/api/models/{m['id']}").status_code == 200
    assert client.get(f"/api/models/{m['id']}").status_code == 404


@pytest.mark.parametrize("model", ["linear", "tree", "forest", "boosting", "knn"])
def test_regression_every_learner(model):
    d = sample("house-prices")
    m = train(dataset_id=d["id"], target="price", model=model, steps=["scale", "drop_missing"]).json()
    assert m["task"] == "regression" and m["model"] == model
    assert m["metrics"]["mae"] < m["metrics"]["baseline_mae"]
    assert m["importance"][0]["name"] == "area_sqft"


def test_upload_and_batch_predict():
    csv = "size;colour;label\n" + "\n".join(f"{i};{'red' if i % 2 else 'blue'};{'big' if i > 30 else 'small'}" for i in range(60))
    d = client.post("/api/datasets", files={"file": ("my data.csv", csv)}).json()
    assert d["rows"] == 60 and [c["name"] for c in d["columns"]] == ["size", "colour", "label"]
    m = train(dataset_id=d["id"], target="label", model="tree").json()
    assert m["metrics"]["accuracy"] > 0.9

    r = client.post(f"/api/models/{m['id']}/predict-file", files={"file": ("new.csv", "size,colour\n5,red\n55,blue\n")})
    out = pd.read_csv(io.StringIO(r.text))
    assert list(out["predicted label"]) == ["small", "big"]
    r = client.post(f"/api/models/{m['id']}/predict-file", files={"file": ("new.csv", "size\n5\n")})
    assert r.status_code == 400 and "colour" in r.json()["detail"]


def test_friendly_errors():
    d = sample("iris-flowers")
    assert train(dataset_id=d["id"], target="nope").status_code == 400
    assert train(dataset_id=d["id"], target="species", task="regression").status_code == 400
    assert train(dataset_id="../../etc/passwd", target="x").status_code == 404
    assert client.post("/api/datasets", files={"file": ("a.csv", "a,b\n1,2\n")}).status_code == 400
    assert client.get("/api/models/" + "0" * 32).status_code == 404


def test_login(monkeypatch):
    monkeypatch.setattr(main, "USERNAME", "admin")
    monkeypatch.setattr(main, "PASSWORD", "s3cret")
    assert client.get("/api/samples").status_code == 401
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/samples", auth=("admin", "wrong")).status_code == 401
    assert client.get("/api/samples", auth=("admin", "s3cret")).status_code == 200
