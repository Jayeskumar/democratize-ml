# Democratize ML

Build a machine-learning model by dragging and dropping — no code, no jargon.

1. **Drop in a spreadsheet** (CSV or Excel) of past examples, or drag in one of the built-in examples.
2. **Drag the column you want predicted** into "Predict this". Everything else becomes a clue.
3. Optionally drag in **clean-up** blocks and a **learner** block (or leave it on *Best pick*).
4. Press **Train my model**.

You get a plain-language report ("92% correct, versus 61% for always guessing the most common
answer"), a chart of which columns matter most, a form to try single predictions, a way to fill in
predictions for a whole spreadsheet, a downloadable model file and a ready-made API endpoint.

The same application runs **on-premise** and **in the cloud**. It is a single container with no
external services, and the web page loads nothing from the internet, so it also works on
air-gapped networks.

## Run it on-premise

### With Docker (recommended)

```bash
docker compose up -d
```

Open <http://localhost:8080>. Uploaded data and trained models are stored in the `ml-data` Docker
volume on that machine and never leave it.

To share it on your office network, put a `.env` file next to `docker-compose.yml` first:

```ini
APP_USERNAME=team
APP_PASSWORD=choose-something-long
APP_PORT=8080
```

Colleagues then open `http://<server-name>:8080` and sign in. Put it behind your usual reverse
proxy (nginx, Caddy, IIS…) if you need HTTPS.

### Without Docker

Needs Python 3.10 or newer.

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app                                    # http://localhost:8000
```

## Run it in the cloud

Any service that can run a container works; the app listens on `$PORT` and needs one writable
folder (`/data`).

### Render (blueprint included)

1. In Render choose **New → Blueprint** and select this repository. `render.yaml` does the rest.
2. Under the service's **Environment** tab, copy the generated `APP_PASSWORD` (username `admin`).

The blueprint attaches a 1 GB disk so models survive restarts, which requires a paid instance
type. To try it free, delete the `disk:` block and change `plan` to `free` — it works the same,
but saved models are wiped whenever the service restarts.

### Google Cloud Run, Azure Container Apps, AWS App Runner, Fly.io, …

Build the `Dockerfile`, deploy the image, and set:

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_USERNAME`, `APP_PASSWORD` | Set both to require a login. **Always set these on the public internet.** | off |
| `DATA_DIR` | Folder for datasets and models. Mount persistent storage here. | `/data` in the image |
| `MAX_UPLOAD_MB` | Largest spreadsheet accepted. | `25` |
| `PORT` | Port to listen on (most platforms set this for you). | `8000` |

Example for Cloud Run:

```bash
gcloud run deploy democratize-ml --source . --memory 1Gi \
  --set-env-vars APP_USERNAME=admin,APP_PASSWORD=choose-something-long
```

Run a **single instance** (models are stored on local disk, not shared between copies), and give
it at least 1 GB of memory.

## Using a trained model elsewhere

Every model has a JSON endpoint (the model page shows a copy-paste `curl` command):

```bash
curl -X POST http://localhost:8080/api/models/<model-id>/predict \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"months_as_customer": 12, "plan": "Basic", "contract": "Monthly"}]}'
```

The **Download model file** button gives a standard scikit-learn pipeline:

```python
import joblib, pandas as pd
model = joblib.load("customer-churn-predict-cancelled.joblib")
model.predict(pd.DataFrame([{"months_as_customer": 12, "plan": "Basic", ...}]))
```

Interactive API reference: `/api/docs`.

## What it does for you

- Works out whether the answer is a **category** (yes/no, type A/B/C) or a **number** (price, days); you can override it.
- Fills in blanks, converts text columns to numbers, and leaves out columns that can't help (IDs, names, constants) — and tells you it did.
- Holds back 20% of the rows to test the model honestly, and compares the score against naive guessing.
- *Best pick* tries a simple formula, a random forest and gradient boosting, and keeps the winner.

Limits: tabular data only (no images or free text); up to 300 columns; training uses at most
50,000 rows (a random sample of larger files); categories to predict are capped at 50.

## Development

```bash
pip install -r requirements-dev.txt
pytest                              # API and training tests
python scripts/make_samples.py      # regenerate the example datasets
uvicorn app.main:app --reload
```

```
app/main.py      web server, storage, login
app/ml.py        profiling, training, prediction (scikit-learn)
app/static/      the drag-and-drop page (plain HTML/CSS/JS, no build step)
app/samples/     example datasets (synthetic)
```
