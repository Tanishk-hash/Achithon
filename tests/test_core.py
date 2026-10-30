from __future__ import annotations

import csv

import pytest

from forgetgraph.core import (
    ARTIFACT_EDGES,
    ForgetGraphStore,
    PipelineRefreshError,
    TombstonedIdentifierError,
)

SAMPLE_CSV = "data/sample_records.csv"


@pytest.fixture
def store(tmp_path):
    pipeline = ForgetGraphStore(tmp_path / "forgetgraph.sqlite3")
    pipeline.bootstrap(SAMPLE_CSV)
    return pipeline


def test_bootstrap_builds_expected_graph_and_artifacts(store):
    status = store.status()

    assert status["source_record_count"] == 10
    assert status["prediction_cache_count"] == 10
    assert {
        (edge["source"], edge["target"])
        for edge in status["artifact_graph"]["edges"]
    } == set(ARTIFACT_EDGES)
    assert {node["id"] for node in status["artifact_graph"]["nodes"]} == {
        "source_data",
        "engineered_features",
        "prediction_cache",
        "ridge_model",
    }


def test_deletion_removes_identifier_from_all_derived_artifacts(store):
    result = store.request_deletion("user-003")

    assert result.execution_status == "completed"
    assert result.verification_status == "passed"
    assert not result.idempotent
    assert "user-003" not in {row["record_id"] for row in store.source_records()}
    assert "user-003" not in {row["record_id"] for row in store.feature_records()}
    assert "user-003" not in {row["record_id"] for row in store.prediction_records()}
    assert store.status()["source_record_count"] == 9


def test_verification_persists_numeric_evidence_separately_from_execution(store):
    result = store.request_deletion("user-004")
    state = store.status()
    run = next(run for run in state["deletion_runs"] if run["run_id"] == result.run_id)
    evidence = next(
        item for item in state["verification_results"] if item["run_id"] == result.run_id
    )

    assert run["status"] == "completed"
    assert evidence["status"] == "passed"
    assert evidence["metrics"]["coefficient_match"]
    assert evidence["metrics"]["intercept_match"]
    assert evidence["metrics"]["predictions_match"]
    assert evidence["metrics"]["rtol"] == 1e-10
    assert evidence["metrics"]["atol"] == 1e-12


def test_execution_completion_does_not_hide_failed_verification(store, monkeypatch):
    monkeypatch.setattr(
        store,
        "_verify_model",
        lambda: {
            "status": "failed",
            "coefficient_match": False,
            "intercept_match": True,
            "predictions_match": False,
            "rtol": 1e-10,
            "atol": 1e-12,
        },
    )

    result = store.request_deletion("user-008")
    state = store.status()
    run = next(run for run in state["deletion_runs"] if run["run_id"] == result.run_id)
    evidence = next(
        item for item in state["verification_results"] if item["run_id"] == result.run_id
    )

    assert result.execution_status == "completed"
    assert result.verification_status == "failed"
    assert run["status"] == "completed"
    assert evidence["status"] == "failed"


def test_repeat_request_is_idempotent_after_verified_deletion(store):
    first = store.request_deletion("user-005")
    second = store.request_deletion("user-005")

    assert first.run_id == second.run_id
    assert second.idempotent
    assert len(store.status()["deletion_runs"]) == 1


def test_managed_ingestion_cannot_restore_tombstoned_identifier(store):
    store.request_deletion("user-006")
    restarted_store = ForgetGraphStore(store.database_path)

    with pytest.raises(TombstonedIdentifierError, match="user-006"):
        restarted_store.ingest_records(
            [
                {
                    "record_id": "user-006",
                    "age": 20,
                    "tenure_months": 2,
                    "monthly_usage": 20,
                    "target": 5,
                }
            ]
        )


def test_failed_workflow_is_audited_and_can_be_retried(store):
    failed = store.request_deletion("user-007", fail_at="feature_recomputation")
    failed_state = store.status()
    failed_run = next(
        run for run in failed_state["deletion_runs"] if run["run_id"] == failed.run_id
    )
    failed_steps = store.deletion_steps(failed.run_id)

    assert failed.execution_status == "failed"
    assert failed_run["status"] == "failed"
    assert next(
        step for step in failed_steps if step["step_name"] == "feature_recomputation"
    )["status"] == "failed"
    assert "user-007" not in {row["record_id"] for row in store.source_records()}

    retried = store.request_deletion("user-007")
    assert retried.execution_status == "completed"
    assert retried.verification_status == "passed"
    assert len(store.status()["deletion_runs"]) == 2
    assert store.is_tombstoned("user-007")


def test_ingestion_batch_validation_is_atomic(store):
    initial_count = store.status()["source_record_count"]
    valid = {
        "record_id": "new-user",
        "age": 30,
        "tenure_months": 10,
        "monthly_usage": 150,
        "target": 42,
    }
    invalid = {**valid, "record_id": "bad-user", "age": float("nan")}

    with pytest.raises(ValueError, match="finite"):
        store.ingest_records([valid, invalid])

    assert store.status()["source_record_count"] == initial_count


def test_deletion_of_final_record_leaves_empty_artifacts(tmp_path):
    csv_path = tmp_path / "one.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["record_id", "age", "tenure_months", "monthly_usage", "target"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "only-user",
                "age": 35,
                "tenure_months": 10,
                "monthly_usage": 150,
                "target": 42,
            }
        )
    pipeline = ForgetGraphStore(tmp_path / "single.sqlite3")
    pipeline.bootstrap(csv_path)

    result = pipeline.request_deletion("only-user")

    assert result.execution_status == "completed"
    assert result.verification_status == "passed"
    assert pipeline.feature_records() == []
    assert pipeline.prediction_records() == []
    assert not pipeline.model_path.exists()


def test_managed_ingestion_refreshes_the_derived_pipeline(store):
    store.ingest_records(
        [
            {
                "record_id": "new-user",
                "age": 37,
                "tenure_months": 14,
                "monthly_usage": 225,
                "target": 63,
            }
        ]
    )

    assert len(store.source_records()) == 11
    assert len(store.feature_records()) == 11
    assert len(store.prediction_records()) == 11
    assert "new-user" in {row["record_id"] for row in store.prediction_records()}


def test_ingestion_refresh_failure_reports_that_source_was_committed(store, monkeypatch):
    def fail_refresh():
        raise OSError("model artifact is unavailable")

    monkeypatch.setattr(store, "_replace_model_and_cache", fail_refresh)
    record = {
        "record_id": "refresh-failure-user",
        "age": 37,
        "tenure_months": 14,
        "monthly_usage": 225,
        "target": 63,
    }

    with pytest.raises(PipelineRefreshError, match="Source records were committed"):
        store.ingest_records([record])

    assert "refresh-failure-user" in {
        row["record_id"] for row in store.source_records()
    }
