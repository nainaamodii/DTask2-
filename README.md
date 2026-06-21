# Event-Driven Traffic Congestion Forecasting — Prototype (Weeks 1–6)

This implements the full 6-week prototype plan, built against the real dataset you
provided (`Astram_event_data_anonymized...csv` — Bengaluru Traffic Police / ASTRAM
event log, 8,173 records, Nov 2023–Apr 2024).

## ⚡ Model update: now using your notebook-trained models

The ML layer has been swapped to use the three artifacts you supplied
(`xgboost_duration_model.pkl`, `xgboost_closure_model.pkl`,
`model_expected_columns.pkl`, reverse-engineered against
`Gridlock_traffic_data.ipynb`, now in `notebooks/`). This is a genuine upgrade,
not just a swap — these models predict a **congestion duration regression**
(closer to the original plan's `peak_delay_minutes` target) instead of the
Week 3 self-trained priority classifier. Details: `backend/app/notebook_features.py`
(the encoding pipeline, fully documented inline) and `backend/app/ml_engine.py`.

**Three things found while integrating these models, worth knowing about:**

1. **Duration target unit bug.** Your notebook computes
   `congestion_time_minutes = (resolved_datetime - start_datetime).dt.total_seconds()`
   — that's `.total_seconds()`, never divided by 60. Verified empirically: raw
   model output for realistic inputs comes out as 18,000–36,000 (5–10 hours), which
   only makes sense as seconds. The API divides the raw output by 60 before
   returning `predicted_duration_minutes`, and says so in the response's `notes`
   field so this isn't silently hidden.
2. **`veh_type` and `priority_High` aren't in the final fitted models**, despite
   being constructed in earlier notebook cells — `model.feature_names_in_` doesn't
   include them. Practical effect: vehicle type is captured for display only and
   cannot influence either prediction; "High priority" is represented implicitly
   (`priority_Low=0, priority_Unknown=0`).
3. **`direction` only covers 6 of 8 real directions** in the fitted model (no
   `north` or `south_east` columns). Selecting those is indistinguishable from the
   dropped baseline to the model — exposed honestly in the UI as "North / other
   (baseline)" rather than silently mapped to something specific.

**Framing change:** `priority` flips from being a model *output* (Week 3) to a
model *input* here — the officer's intake severity assessment, alongside cause,
location, time, and vehicle/cargo details. The models then predict what the
original plan actually asked for: how long the congestion will last, and whether
the road needs to close.

The Week 2 rule-based engine, Week 4 road graph/diversions, and Week 5 dashboard
are unchanged in spirit — `method=rule_based` still works as a comparison point,
and the impact-score/severity-band → resource-recommendation pipeline is reused
unchanged for the new models too (duration is converted to the same 0–10 scale).
The Week 3 ASTRAM-trained priority/closure classifiers are archived
(`backend/app/ml_engine_legacy.py`, `scripts/05_train_ml_model.py`) and no longer
used by default. The Week 6 retraining job has a new version,
`scripts/08_retrain_notebook_models.py`, adapted to this model's feature schema
(the old one is archived as `07_retrain_with_feedback_legacy.py`).

**No external API was needed for any of this** — everything required was in the
three files you uploaded plus the notebook for context.



## How the plan was adapted to the real data

The original plan assumed pre-event data (rallies/festivals with an expected
attendance figure, forecasted *before* the event happens). The actual dataset is a
**live incident log**: every row is an event (mostly unplanned — vehicle breakdowns,
accidents, tree falls, potholes, water-logging — plus some planned events like
processions, VIP movement, protests) with its location, cause, and how the traffic
police triaged it (`priority`, `requires_road_closure`).

That maps cleanly onto a still-very-useful real version of the same idea:

> When a new event is **reported** (a citizen calls in a breakdown, a corridor team
> logs an upcoming procession), predict how severe it will be and what response is
> needed — *before* an officer has manually triaged it.

So the two ML targets are:
- **`priority`** (High/Low) — the severity label the police currently assign
- **`requires_road_closure`** (True/False) — whether the road needs to be physically closed

Both are only modeled from features genuinely known at intake time (cause, location/
corridor/zone, time of day/week, vehicle type) — never from anything that's only known
after the event is resolved (status, duration, closed_datetime, etc.).

**Important data finding** (documented in `outputs/model_metrics.json` and surfaced by
`scripts/05_train_ml_model.py`): in the historical data, `priority` is almost perfectly
determined by whether the event sits on a named arterial corridor or not (Non-corridor →
~100% Low, On-corridor → ~99.6% High). That looks like an existing administrative
labeling rule in ASTRAM rather than an independently assessed judgement call. The
training script trains a second, **ablated** model that excludes corridor-identity
features, to show what's genuinely learnable from cause/time/zone/vehicle-type alone —
the realistic signal for a new report where corridor designation is itself ambiguous.
This kind of finding is exactly the sort of thing worth calling out in a hackathon demo
— it shows you understood the data, not just plugged it into a model.

> **Note:** the above describes the original (now legacy/archived) Week 3 models. The
> currently-active models flip `priority` to be an *input* rather than an output —
> see the "Model update" section at the top of this file for the current targets
> (congestion duration + road closure) and the reasoning behind the change.

## Project structure

```
task 2/
├── data/
│   ├── astram_event_data.csv       # your original dataset (copied in)
│   ├── processed_events.csv        # cleaned + feature-engineered (generated)
│   └── feedback.db                 # generated: SQLite post-event feedback log (Week 6)
├── notebooks/
│   └── Gridlock_traffic_data.ipynb # your supplied notebook (reference for the encoding pipeline)
├── scripts/
│   ├── 01_preprocessing.py         # Week 1: cleaning + feature engineering
│   ├── 02_eda.py                   # Week 1: EDA charts + summary
│   ├── 03_rule_based_baseline.py   # Week 2: weighted rule-based scoring engine
│   ├── 04_rule_based_eval.py       # Week 2: evaluate rule baseline vs historical priority
│   ├── 05_train_ml_model.py        # Week 3 (legacy/archived): train ASTRAM priority/closure models
│   ├── 06_build_road_graph.py      # Week 4: build the corridor road graph
│   ├── 07_retrain_with_feedback_legacy.py  # Week 6 (legacy): retraining for the old priority/closure schema
│   ├── 08_retrain_notebook_models.py       # Week 6 (current): retraining for the duration/closure models
│   └── requirements.txt
├── models/
│   ├── notebook/                   # ★ the models currently used by the API
│   │   ├── xgboost_duration_model.pkl
│   │   ├── xgboost_closure_model.pkl
│   │   ├── model_expected_columns.pkl
│   │   └── v2/, v3/, ...           # generated: versioned retraining outputs (Week 6)
│   ├── encoders.joblib, lookup_tables.joblib, priority_model.json, ...  # legacy Week 3 artifacts
│   └── road_graph.gpickle          # Week 4 road network
├── outputs/
│   ├── eda/                        # generated: charts + eda_summary.md
│   ├── model_metrics.json          # generated: legacy Week 3 eval numbers
│   └── rule_based_eval.csv         # generated: per-event rule-engine scores
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py                 # FastAPI app — /forecast, /feedback, /model-info
│       ├── schemas.py               # request/response models
│       ├── rule_engine.py            # Week 2 engine, re-used by the API
│       ├── notebook_features.py       # ★ encoding pipeline for the notebook models (read this first)
│       ├── ml_engine.py                # ★ current ML inference (duration + closure)
│       ├── ml_engine_legacy.py          # archived Week 3 inference (priority + closure)
│       ├── road_graph.py                 # Week 4 diversion routing + barricade placement
│       └── feedback_store.py              # Week 6 SQLite-backed feedback log
└── frontend/
    └── index.html                  # Week 5: event form + map dashboard + recommendation card
```

## Running it

From inside this folder:

```bash
pip install -r scripts/requirements.txt

# Week 1
python scripts/01_preprocessing.py
python scripts/02_eda.py

# Week 2
python scripts/03_rule_based_baseline.py     # quick demo print
python scripts/04_rule_based_eval.py         # eval vs historical labels

# Week 4 (road graph — needed for diversions/barricades in /forecast)
python scripts/06_build_road_graph.py
```

(Week 3's `05_train_ml_model.py` is optional/legacy now — the API uses your
notebook-trained models in `models/notebook/` directly, no training step needed.)

Then run the API:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger docs, or call it directly:

```bash
curl -X POST "http://127.0.0.1:8000/forecast?method=ml_model" \
  -H "Content-Type: application/json" \
  -d '{
        "event_type": "unplanned",
        "event_cause": "vehicle_breakdown",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "corridor": "Bellary Road 1",
        "authenticated": true,
        "at_junction": false,
        "priority": "High",
        "direction": "south",
        "cargo_macro": "empty_or_none",
        "breakdown_severity": "minor_issue",
        "event_datetime": "2026-06-21T13:30:00Z"
      }'
```

Swap `method=ml_model` for `method=rule_based` to compare the Week 2 baseline against
the trained duration/closure models side by side — useful for a "before/after ML" demo slide.

Then open the dashboard — with the backend running on port 8000, just open
`frontend/index.html` directly in a browser (double-click it, or `open frontend/index.html`
/ drag into a browser tab). No build step, no npm install — it's a single static file
that talks to `http://127.0.0.1:8000` via fetch.

## Results so far

These are **legacy Week 3 numbers** (the archived ASTRAM-trained priority/closure
models). They're kept for reference since the comparison to the rule-based baseline
is still a useful story.

| Model | Metric | Score |
|---|---|---|
| Rule-based baseline (Week 2) | Accuracy (best threshold) | ~0.66 |
| Rule-based baseline (Week 2) | ROC-AUC | ~0.667 |
| ML priority model — ablated, no corridor identity (Week 3, legacy) | ROC-AUC | ~0.69 |
| ML priority model — full features (Week 3, legacy) | ROC-AUC | ~1.00 (see data finding above — mostly re-learning the corridor rule) |
| ML road-closure model (Week 3, legacy) | ROC-AUC | ~0.79 |

Full numbers, per-class precision/recall, and feature importances are in
`outputs/model_metrics.json` after running the legacy training script.

**For the currently-active notebook models, we don't have held-out test metrics** —
only the fitted `.pkl` files were provided, not the original train/test split or the
notebook's evaluation cell output for those specific final models. `GET /model-info`
returns the integration-time findings (units bug, unused features, direction gaps)
instead of accuracy numbers. If you have the notebook's printed MAE/R²/accuracy/ROC-AUC
output from when these were trained, share it and I can add it here properly; otherwise,
the honest thing to do is not fabricate numbers we don't have.

## Week 4 — Road graph & diversion routing

**Adaptation note:** the original plan called for OSMnx pulling live OpenStreetMap data.
This sandbox's network is locked to a small allow-list of package-registry domains (no
openstreetmap.org / Overpass API), so a live OSM pull wasn't possible here. Instead,
`scripts/06_build_road_graph.py` builds a routable graph **from the dataset itself**:

- Each named corridor (Mysore Road, Bellary Road 1, ORR North 1, ...) becomes a chain of
  nodes placed along the real lat/lon centroid of events logged on that corridor (via a
  PCA-based ordering, so nodes run roughly along the road's actual direction).
- Corridors are joined at junction nodes wherever two corridors' point-clouds pass within
  ~1.3km of each other, and any disconnected components get bridged via their nearest
  node pair — so the result is a single connected, geographically grounded graph (104
  nodes, 133 edges from this dataset) rather than a toy abstraction.
- Each edge carries a distance, an assumed speed, and a derived travel time, so routing
  optimizes for time, not just hop count.

`backend/app/road_graph.py` uses this graph at request time to:
- Find the two corridor nodes that bracket an incoming incident's location.
- Heavily penalize (rather than hard-delete) the edges right at the incident, then run
  `networkx.shortest_simple_paths` to get up to 3 ranked alternate routes — this is the
  **diversion recommendation**.
- Generate evenly-spaced **barricade containment points** around the incident at a radius
  scaled to the recommended barricade count.

Both are now embedded directly in the `/forecast` response (`barricade_points`,
`diversion_routes`) and rendered live on the map in the Week 5 frontend — red dots for
barricades, colored lines for diversion routes.

If/when you have real OSM access (e.g. running this outside the sandbox), swap
`scripts/06_build_road_graph.py`'s graph construction for `osmnx.graph_from_place(...)`
and everything downstream (`road_graph.py`, the API, the frontend) keeps working
unchanged — they only depend on the graph having `lat`/`lon` node attributes and
`distance_km`/`travel_time_min` edge attributes.

## Week 5 — Frontend dashboard

`frontend/index.html` is the control-room dashboard: event intake form on the left, a
live dark-themed Leaflet map in the center, and a recommendation panel on the right
(impact score, severity band, manpower/barricades/diversions, diversion route list,
export-to-JSON, and a "log post-event feedback" action that round-trips into Week 6's
feedback loop).

**Adaptation note:** the original plan specified React + Leaflet.js with a build step.
This is built as a single self-contained HTML/JS/Leaflet file instead — same map
library, zero build tooling, so it runs by opening the file directly in any browser
with no `npm install` required. The design is intentionally a dark, dense "dispatch
software" aesthetic (amber/red/green severity colors mirroring real traffic-signal
colors, monospace for data, sans for labels) rather than a generic dashboard template,
since this is a tool a traffic control room operator would stare at all day, not a
marketing page.

Click anywhere on the map to set the incident location, pick a cause/corridor/zone,
and hit "Forecast Impact." Toggle "ML MODEL" vs "RULE-BASED" in the top bar to compare
methods on the same event.

## Week 6 — Post-event learning loop

The full loop from the original plan, now wired end-to-end for the duration/closure models:

```
Event occurs
    → Officer logs actual outcome via the dashboard's "Log post-event feedback"
      button (captures actual duration in minutes + actual closure), or directly: POST /feedback
    → System compares predicted vs actual: GET /feedback/stats
    → Feedback persisted to data/feedback.db (SQLite — survives restarts)
    → Monthly retraining batch: python scripts/08_retrain_notebook_models.py
    → New model version saved to models/notebook/v2/, v3/, ... (never overwrites
      the live model automatically — promotion is a manual, reviewable step)
```

**Two honest limitations, documented in code:**

1. (`scripts/08_retrain_notebook_models.py`) — same as before: the `/feedback`
   payload only carries the outcome and an `event_id`. The dashboard's feedback
   button passes the original intake payload through as JSON in `notes`, and the
   retraining script parses it back out; rows without that JSON are skipped with
   a logged count rather than silently fabricated.
2. **No access to the original training dataframe** (`data_shared.csv` from your
   notebook) — only the fitted `.pkl` files were provided. So this script can't do
   a true "old data + new feedback" full retrain. Instead it **warm-starts**:
   continues boosting the existing fitted models on the feedback batch alone
   (`xgb_model=...`, a real, supported XGBoost technique for incremental
   training). This is a reasonable prototype stand-in, but the before/after
   metrics it reports are evaluated **in-sample on the fine-tuning batch**, not a
   held-out set — treat them as a sanity check, not a generalization estimate.
   If you can supply `data_shared.csv` (or an equivalent cleaned dataframe), the
   script has a clearly marked spot (`ORIGINAL_TRAINING_DATA`) to wire up a proper
   full retrain instead.

Try it:

```bash
# 1. Start the backend, open the dashboard, run a forecast, click
#    "Log post-event feedback", fill in the actual outcome. Repeat
#    a few times (the script wants at least 5 rows to fine-tune on).
# 2. Then:
cd ..  # back to project root
python scripts/08_retrain_notebook_models.py
```

This prints a before/after metrics comparison and writes a new versioned model to
`models/notebook/v<N>/`, plus a `retrain_report.json` documenting exactly what changed
and why — so a bad batch of feedback can be inspected and rejected before promotion,
instead of silently overwriting the deployed model.

(The original Week 3 retraining script, `07_retrain_with_feedback_legacy.py`, still
works against the archived ASTRAM priority/closure models if you want to keep using
those for comparison.)

## What's not built (acknowledged scope cuts)

- **Real OSM road network** — using a dataset-derived synthetic graph instead (see Week 4
  note above); swapping in `osmnx` is a contained change if network access allows it.
- **Auto-promotion / scheduled retraining** — `scripts/07_retrain_with_feedback.py` is
  invoked manually here; production would put it behind a cron/Airflow schedule and a
  human approval gate before promoting `models/vN/` to live.
- **A persisted `events` table** — `/forecast` calls aren't currently logged server-side,
  which is why Week 6's feedback join is simulated via the `notes` field workaround
  above. Adding an `events` SQLite table mirroring `feedback` would close this cleanly.
