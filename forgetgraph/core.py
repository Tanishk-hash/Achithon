from __future__ import annotations

import csv
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = (
    "age",
    "tenure_months",
    "monthly_usage",
    "usage_per_tenure",
)
RIDGE_ALPHA = 1.0
VERIFICATION_RTOL = 1e-10
VERIFICATION_ATOL = 1e-12

ARTIFACT_NODES = (
    ("source_data", "Source records"),
    ("engineered_features", "Engineered features"),
    ("prediction_cache", "Prediction cache"),
    ("ridge_model", "Ridge regression model"),
)
ARTIFACT_EDGES = (
    ("source_data", "engineered_features"),
    ("engineered_features", "prediction_cache"),
    ("engineered_features", "ridge_model"),
    ("ridge_model", "prediction_cache"),
)
DELETION_STEPS = (
    "source_removal",
    "feature_recomputation",
    "cache_invalidation",
    "model_replacement",
    "model_verification",
)


class TombstonedIdentifierError(ValueError):
    """Raised when managed ingestion attempts to restore a deleted identifier."""


class PipelineRefreshError(RuntimeError):
    """Raised when a committed ingestion cannot refresh its dependent artifacts."""


@dataclass(frozen=True)
class DeletionResult:
    record_id: str
    run_id: str
    execution_status: str
    verification_status: str | None
    idempotent: bool
    error: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


class ForgetGraphStore:
    """SQLite-backed source, lineage, deletion registry, and workflow history."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = self.database_path.with_name("ridge_model.joblib")
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_records (
                    record_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deletion_registry (
                    record_id TEXT PRIMARY KEY,
                    requested_at TEXT NOT NULL,
                    last_run_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deletion_runs (
                    run_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS deletion_steps (
                    run_id TEXT NOT NULL REFERENCES deletion_runs(run_id),
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (run_id, step_name)
                );
                CREATE TABLE IF NOT EXISTS verification_results (
                    run_id TEXT PRIMARY KEY REFERENCES deletion_runs(run_id),
                    status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS features (
                    record_id TEXT PRIMARY KEY,
                    values_json TEXT NOT NULL,
                    target REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    record_id TEXT PRIMARY KEY,
                    prediction REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_metadata (
                    name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
        required = {"record_id", *FEATURE_COLUMNS[:3], "target"}
        missing = required.difference(record)
        extra = set(record).difference(required)
        if missing or extra:
            raise ValueError(
                f"Record schema mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(extra)}"
            )

        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")

        normalized: dict[str, Any] = {"record_id": record_id.strip()}
        for field in ("age", "tenure_months", "monthly_usage", "target"):
            try:
                number = float(record[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} must be numeric") from error
            if not math.isfinite(number):
                raise ValueError(f"{field} must be finite")
            normalized[field] = number
        if normalized["age"] < 0 or normalized["tenure_months"] < 0:
            raise ValueError("age and tenure_months cannot be negative")
        return normalized

    def ingest_records(
        self, records: Sequence[dict[str, Any]], *, refresh_artifacts: bool = True
    ) -> int:
        normalized = [self._validate_record(record) for record in records]
        identifiers = [record["record_id"] for record in normalized]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("An ingestion batch cannot contain duplicate record_id values")

        with self._connect() as connection:
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                deleted = connection.execute(
                    f"SELECT record_id FROM deletion_registry "
                    f"WHERE record_id IN ({placeholders})",
                    identifiers,
                ).fetchall()
                if deleted:
                    blocked = sorted(row["record_id"] for row in deleted)
                    raise TombstonedIdentifierError(
                        f"Managed ingestion rejected deleted identifier(s): {blocked}"
                    )
            for record in normalized:
                connection.execute(
                    "INSERT INTO source_records(record_id, payload_json) VALUES (?, ?)",
                    (record["record_id"], _encode(record)),
                )
        if normalized and refresh_artifacts:
            try:
                with self._connect() as connection:
                    source_count = connection.execute(
                        "SELECT COUNT(*) AS count FROM source_records"
                    ).fetchone()["count"]
                self._set_artifact(
                    "source_data", "current", {"record_count": source_count}
                )
                self._recompute_features()
                self._invalidate_cache()
                self._replace_model_and_cache()
            except Exception as error:
                raise PipelineRefreshError(
                    "Source records were committed, but refreshing derived artifacts "
                    f"failed: {type(error).__name__}: {error}"
                ) from error
        return len(normalized)

    def ingest_csv(
        self, csv_path: str | Path, *, refresh_artifacts: bool = True
    ) -> int:
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            expected = ["record_id", "age", "tenure_months", "monthly_usage", "target"]
            if reader.fieldnames != expected:
                raise ValueError(f"CSV headers must be exactly {expected}")
            return self.ingest_records(
                list(reader), refresh_artifacts=refresh_artifacts
            )

    def bootstrap(self, sample_csv: str | Path) -> None:
        """Seed once on a fresh database and prepare initial derived artifacts."""
        with self._connect() as connection:
            setting = connection.execute(
                "SELECT value FROM settings WHERE key='bootstrap_complete'"
            ).fetchone()
            if setting:
                return
            source_count = connection.execute(
                "SELECT COUNT(*) AS count FROM source_records"
            ).fetchone()["count"]
            tombstone_count = connection.execute(
                "SELECT COUNT(*) AS count FROM deletion_registry"
            ).fetchone()["count"]

        if source_count == 0 and tombstone_count == 0:
            self.ingest_csv(sample_csv, refresh_artifacts=False)
        with self._connect() as connection:
            current_source_count = connection.execute(
                "SELECT COUNT(*) AS count FROM source_records"
            ).fetchone()["count"]
        self._set_artifact(
            "source_data", "current", {"record_count": current_source_count}
        )
        self._recompute_features()
        self._invalidate_cache()
        self._replace_model_and_cache()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES "
                "('bootstrap_complete', 'true')"
            )

    def _source_records(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM source_records ORDER BY record_id"
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    @staticmethod
    def _feature_values(record: dict[str, Any]) -> dict[str, float]:
        tenure = float(record["tenure_months"])
        return {
            "age": float(record["age"]),
            "tenure_months": tenure,
            "monthly_usage": float(record["monthly_usage"]),
            "usage_per_tenure": float(record["monthly_usage"]) / max(tenure, 1.0),
        }

    def _recompute_features(self) -> int:
        records = self._source_records()
        with self._connect() as connection:
            connection.execute("DELETE FROM features")
            connection.executemany(
                "INSERT INTO features(record_id, values_json, target) VALUES (?, ?, ?)",
                [
                    (
                        record["record_id"],
                        _encode(self._feature_values(record)),
                        record["target"],
                    )
                    for record in records
                ],
            )
        self._set_artifact(
            "engineered_features",
            "current",
            {"record_count": len(records), "columns": list(FEATURE_COLUMNS)},
        )
        return len(records)

    def _invalidate_cache(self) -> int:
        with self._connect() as connection:
            deleted = connection.execute("DELETE FROM prediction_cache").rowcount
        self._set_artifact(
            "prediction_cache", "invalidated", {"invalidated_count": deleted}
        )
        return deleted

    def _feature_rows(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT record_id, values_json, target FROM features ORDER BY record_id"
            ).fetchall()

    def _replace_model_and_cache(self) -> int:
        rows = self._feature_rows()
        with self._connect() as connection:
            connection.execute("DELETE FROM prediction_cache")

        if not rows:
            if self.model_path.exists():
                self.model_path.unlink()
            self._set_artifact("ridge_model", "empty", {"record_count": 0})
            self._set_artifact("prediction_cache", "current", {"record_count": 0})
            return 0

        x_values = [json.loads(row["values_json"]) for row in rows]
        x = np.asarray(
            [[values[column] for column in FEATURE_COLUMNS] for values in x_values],
            dtype=float,
        )
        y = np.asarray([row["target"] for row in rows], dtype=float)
        model = _model_pipeline()
        model.fit(x, y)
        joblib.dump(model, self.model_path)
        predictions = model.predict(x)

        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO prediction_cache(record_id, prediction) VALUES (?, ?)",
                [
                    (row["record_id"], float(prediction))
                    for row, prediction in zip(rows, predictions)
                ],
            )
        self._set_artifact(
            "ridge_model",
            "current",
            {
                "record_count": len(rows),
                "model": "sklearn.linear_model.Ridge",
                "alpha": RIDGE_ALPHA,
                "preprocessing": "StandardScaler",
                "feature_columns": list(FEATURE_COLUMNS),
                "path": str(self.model_path),
            },
        )
        self._set_artifact(
            "prediction_cache", "current", {"record_count": len(predictions)}
        )
        return len(rows)

    def _set_artifact(self, name: str, status: str, details: dict[str, Any]) -> None:
        with self._connect() as connection:
            previous = connection.execute(
                "SELECT version FROM artifact_metadata WHERE name=?", (name,)
            ).fetchone()
            version = 1 if previous is None else previous["version"] + 1
            connection.execute(
                """
                INSERT INTO artifact_metadata(name, status, version, updated_at, details_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    status=excluded.status,
                    version=excluded.version,
                    updated_at=excluded.updated_at,
                    details_json=excluded.details_json
                """,
                (name, status, version, _now(), _encode(details)),
            )

    def _create_run(self, record_id: str) -> str:
        run_id = str(uuid.uuid4())
        requested_at = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO deletion_registry(record_id, requested_at, last_run_id) "
                "VALUES (?, ?, ?)",
                (record_id, requested_at, run_id),
            )
            connection.execute(
                """
                INSERT INTO deletion_runs(run_id, record_id, status, requested_at)
                VALUES (?, ?, 'in_progress', ?)
                """,
                (run_id, record_id, requested_at),
            )
            connection.execute(
                "UPDATE deletion_registry SET last_run_id=? WHERE record_id=?",
                (run_id, record_id),
            )
            connection.executemany(
                "INSERT INTO deletion_steps(run_id, step_name, status) "
                "VALUES (?, ?, 'pending')",
                [(run_id, step) for step in DELETION_STEPS],
            )
        return run_id

    def _latest_run(self, record_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT run_id, status FROM deletion_runs
                WHERE record_id=? ORDER BY requested_at DESC LIMIT 1
                """,
                (record_id,),
            ).fetchone()

    def _verification_status(self, run_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM verification_results WHERE run_id=?", (run_id,)
            ).fetchone()
        return None if row is None else row["status"]

    def _set_step(
        self,
        run_id: str,
        step_name: str,
        status: str,
        *,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE deletion_steps SET status=?,
                    started_at=COALESCE(started_at, ?),
                    completed_at=CASE WHEN ? IN ('completed', 'failed') THEN ? ELSE completed_at END,
                    error=?, details_json=?
                WHERE run_id=? AND step_name=?
                """,
                (
                    status,
                    timestamp,
                    status,
                    timestamp,
                    error,
                    _encode(details or {}),
                    run_id,
                    step_name,
                ),
            )

    def _verify_model(self) -> dict[str, Any]:
        rows = self._feature_rows()
        if rows:
            value_dicts = [json.loads(row["values_json"]) for row in rows]
            x = np.asarray(
                [[values[column] for column in FEATURE_COLUMNS] for values in value_dicts],
                dtype=float,
            )
            y = np.asarray([row["target"] for row in rows], dtype=float)
            reference = _model_pipeline()
            reference.fit(x, y)
            replacement = joblib.load(self.model_path)
            replacement_ridge = replacement.named_steps["ridge"]
            reference_ridge = reference.named_steps["ridge"]
            replacement_predictions = replacement.predict(x)
            reference_predictions = reference.predict(x)
            coefficient_match = bool(
                np.allclose(
                    replacement_ridge.coef_,
                    reference_ridge.coef_,
                    rtol=VERIFICATION_RTOL,
                    atol=VERIFICATION_ATOL,
                )
            )
            intercept_match = bool(
                np.allclose(
                    replacement_ridge.intercept_,
                    reference_ridge.intercept_,
                    rtol=VERIFICATION_RTOL,
                    atol=VERIFICATION_ATOL,
                )
            )
            predictions_match = bool(
                np.allclose(
                    replacement_predictions,
                    reference_predictions,
                    rtol=VERIFICATION_RTOL,
                    atol=VERIFICATION_ATOL,
                )
            )
            prediction_difference = float(
                np.max(np.abs(replacement_predictions - reference_predictions))
            )
            coefficient_difference = float(
                np.max(np.abs(replacement_ridge.coef_ - reference_ridge.coef_))
            )
            intercept_difference = float(
                np.max(np.abs(replacement_ridge.intercept_ - reference_ridge.intercept_))
            )
        else:
            coefficient_match = intercept_match = predictions_match = True
            prediction_difference = coefficient_difference = intercept_difference = 0.0

        passed = coefficient_match and intercept_match and predictions_match
        return {
            "status": "passed" if passed else "failed",
            "record_count": len(rows),
            "rtol": VERIFICATION_RTOL,
            "atol": VERIFICATION_ATOL,
            "coefficient_match": coefficient_match,
            "intercept_match": intercept_match,
            "predictions_match": predictions_match,
            "max_coefficient_abs_difference": coefficient_difference,
            "max_intercept_abs_difference": intercept_difference,
            "max_prediction_abs_difference": prediction_difference,
        }

    def _save_verification(self, run_id: str, metrics: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO verification_results(run_id, status, metrics_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, metrics["status"], _encode(metrics), _now()),
            )

    def request_deletion(
        self, record_id: str, *, fail_at: str | None = None
    ) -> DeletionResult:
        record_id = record_id.strip()
        if not record_id:
            raise ValueError("record_id must be a non-empty string")
        if fail_at is not None and fail_at not in DELETION_STEPS:
            raise ValueError(f"fail_at must be one of {DELETION_STEPS}")

        latest = self._latest_run(record_id)
        if latest and latest["status"] == "completed":
            verification_status = self._verification_status(latest["run_id"])
            if verification_status == "passed":
                return DeletionResult(
                    record_id,
                    latest["run_id"],
                    "completed",
                    "passed",
                    True,
                )

        run_id = self._create_run(record_id)
        operations = {
            "source_removal": self._remove_source_record,
            "feature_recomputation": self._recompute_features,
            "cache_invalidation": self._invalidate_cache,
            "model_replacement": self._replace_model_and_cache,
            "model_verification": lambda: self._verify_model(),
        }
        for step_name in DELETION_STEPS:
            self._set_step(run_id, step_name, "in_progress")
            try:
                if fail_at == step_name:
                    raise RuntimeError(f"Simulated failure at step: {step_name}")
                if step_name == "source_removal":
                    result = operations[step_name](record_id)
                else:
                    result = operations[step_name]()
                if step_name == "model_verification":
                    self._save_verification(run_id, result)
                    self._set_step(
                        run_id,
                        step_name,
                        "completed",
                        details=result,
                    )
                else:
                    self._set_step(
                        run_id,
                        step_name,
                        "completed",
                        details={"result": result},
                    )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self._set_step(run_id, step_name, "failed", error=message)
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE deletion_runs SET status='failed', completed_at=?, error=? "
                        "WHERE run_id=?",
                        (_now(), message, run_id),
                    )
                return DeletionResult(
                    record_id,
                    run_id,
                    "failed",
                    self._verification_status(run_id),
                    False,
                    message,
                )

        with self._connect() as connection:
            connection.execute(
                "UPDATE deletion_runs SET status='completed', completed_at=? WHERE run_id=?",
                (_now(), run_id),
            )
        return DeletionResult(
            record_id,
            run_id,
            "completed",
            self._verification_status(run_id),
            False,
        )

    def _remove_source_record(self, record_id: str) -> int:
        with self._connect() as connection:
            removed = connection.execute(
                "DELETE FROM source_records WHERE record_id=?", (record_id,)
            ).rowcount
            remaining = connection.execute(
                "SELECT COUNT(*) AS count FROM source_records"
            ).fetchone()["count"]
        self._set_artifact(
            "source_data", "current", {"record_count": remaining, "removed_count": removed}
        )
        return removed

    def is_tombstoned(self, record_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM deletion_registry WHERE record_id=?", (record_id,)
                ).fetchone()
                is not None
            )

    def artifact_graph(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, status, version, updated_at, details_json FROM artifact_metadata"
            ).fetchall()
        by_name = {row["name"]: row for row in rows}
        nodes = []
        for name, label in ARTIFACT_NODES:
            row = by_name.get(name)
            nodes.append(
                {
                    "id": name,
                    "label": label,
                    "status": "not initialized" if row is None else row["status"],
                    "version": 0 if row is None else row["version"],
                    "updated_at": None if row is None else row["updated_at"],
                    "details": {} if row is None else json.loads(row["details_json"]),
                }
            )
        return {
            "nodes": nodes,
            "edges": [{"source": source, "target": target} for source, target in ARTIFACT_EDGES],
        }

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            source_count = connection.execute(
                "SELECT COUNT(*) AS count FROM source_records"
            ).fetchone()["count"]
            tombstones = connection.execute(
                "SELECT record_id, requested_at, last_run_id FROM deletion_registry "
                "ORDER BY requested_at DESC"
            ).fetchall()
            runs = connection.execute(
                "SELECT run_id, record_id, status, requested_at, completed_at, error "
                "FROM deletion_runs ORDER BY requested_at DESC"
            ).fetchall()
            steps = connection.execute(
                "SELECT run_id, step_name, status, started_at, completed_at, error, details_json "
                "FROM deletion_steps ORDER BY rowid"
            ).fetchall()
            verifications = connection.execute(
                "SELECT run_id, status, metrics_json, created_at FROM verification_results"
            ).fetchall()
            cache_count = connection.execute(
                "SELECT COUNT(*) AS count FROM prediction_cache"
            ).fetchone()["count"]
        return {
            "source_record_count": source_count,
            "prediction_cache_count": cache_count,
            "deletion_registry": [dict(row) for row in tombstones],
            "deletion_runs": [dict(row) for row in runs],
            "deletion_steps": [
                {**dict(row), "details": json.loads(row["details_json"])} for row in steps
            ],
            "verification_results": [
                {**dict(row), "metrics": json.loads(row["metrics_json"])}
                for row in verifications
            ],
            "artifact_graph": self.artifact_graph(),
        }

    def source_records(self) -> list[dict[str, Any]]:
        return self._source_records()

    def feature_records(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_id, values_json, target FROM features ORDER BY record_id"
            ).fetchall()
        return [
            {
                "record_id": row["record_id"],
                **json.loads(row["values_json"]),
                "target": row["target"],
            }
            for row in rows
        ]

    def prediction_records(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_id, prediction FROM prediction_cache ORDER BY record_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def deletion_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT step_name, status, started_at, completed_at, error, details_json "
                "FROM deletion_steps WHERE run_id=? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [
            {**dict(row), "details": json.loads(row["details_json"])} for row in rows
        ]
