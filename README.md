# FORGETGRAPH

FORGETGRAPH is a small, runnable prototype of a dependency-aware deletion orchestrator for a managed machine-learning pipeline. It is built for the selected [PS09 problem statement](PS09.md).

## What it demonstrates

- A persisted artifact dependency graph for source records, engineered features, a prediction cache, and a Ridge regression model.
- Ordered deletion: remove a source record, recompute features, invalidate cached predictions, replace the model, and independently verify the replacement.
- A persistent SQLite deletion registry (tombstones) that blocks deleted IDs from managed ingestion, including after process restarts.
- Separate execution status and verification evidence, with coefficient, intercept, and prediction comparisons using declared tolerances.
- Idempotent repeated requests and retryable failed workflows; a failed execution is never presented as a completed deletion.
- A Streamlit dashboard, CLI, synthetic demo data, and pytest coverage.

The sample records are synthetic demo data, not real user information. This prototype controls only records ingested into its SQLite database; it does not claim to delete copies in external systems.

## Project structure

```text
.
├── data/
│   └── sample_records.csv
├── forgetgraph/
│   ├── __init__.py
│   ├── __main__.py
│   ├── core.py
│   └── dashboard.py
├── tests/
│   └── test_core.py
├── .gitignore
├── PS09.md
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

Runtime state and the serialized model are stored in the ignored `.forgetgraph/` directory by default.

## Setup

Python 3.10 or later is recommended.

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

With the environment activated, install dependencies and use the same commands on each platform:

```sh
python -m pip install -r requirements-dev.txt
```

## Run the dashboard

```sh
python -m streamlit run forgetgraph/dashboard.py
```

Open the local URL printed by Streamlit. The dashboard starts a demo pipeline on first run, visualizes lineage and artifact state, accepts deletion requests, shows execution and verification evidence, and provides a managed ingestion form.

## CLI

```sh
python -m forgetgraph init
python -m forgetgraph status
python -m forgetgraph delete user-003
```

To exercise recovery, inject a step failure, inspect the failed run, and retry the same ID:

```sh
python -m forgetgraph delete user-004 --fail-at feature_recomputation
python -m forgetgraph status
python -m forgetgraph delete user-004
```

The failure injection is for demonstrations and testing; omit `--fail-at` for normal operation. You can point the app or CLI at another SQLite file with `FORGETGRAPH_DB` or the CLI `--db` option. On a fresh database, the sample CSV is loaded once. Subsequent runs reuse persisted state and do not reseed deleted records.

Managed ingestion accepts CSV files with exactly these headers and in this order:

```text
record_id,age,tenure_months,monthly_usage,target
```

For example:

```sh
python -m forgetgraph ingest data/sample_records.csv
```

Ingestion rejects malformed records, duplicate primary keys, and tombstoned identifiers. Accepted records trigger feature, model, and prediction-cache refreshes.

## Model and verification contract

The demo derives `usage_per_tenure = monthly_usage / max(tenure_months, 1)` and fits a deterministic scikit-learn pipeline: `StandardScaler` followed by `Ridge(alpha=1.0)`. After a deletion, a separately constructed pipeline is fit on the retained feature rows. The saved replacement and independent reference are compared on Ridge coefficients, intercept, and predictions with `rtol=1e-10` and `atol=1e-12`. Verification metrics and status are stored in `verification_results`, separately from deletion-run execution status.

The default graph is:

```text
source_data ──> engineered_features ──> ridge_model ──> prediction_cache
                         └───────────────────────────> prediction_cache
```

Failure status and step details are persisted. A tombstone is written before deletion work starts and remains if a step fails, so a retry can finish safely and the identifier cannot be re-ingested. A repeated request for an already completed and verified deletion returns the prior run without repeating work.

## Test

```sh
python -m pytest
```

Tests cover graph/artifact creation, complete deletion, independent numerical verification, repeated requests, re-ingestion prevention, failure recovery, validation atomicity, and deletion of the final source record.
