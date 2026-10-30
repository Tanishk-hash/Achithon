from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import streamlit as st

from forgetgraph.__main__ import DEFAULT_DATABASE, SAMPLE_CSV
from forgetgraph.core import (
    ForgetGraphStore,
    PipelineRefreshError,
    TombstonedIdentifierError,
)


def _database_path() -> Path:
    return Path(os.environ.get("FORGETGRAPH_DB", DEFAULT_DATABASE))


def _graph_dot(graph: dict) -> str:
    nodes = {
        node["id"]: f'{node["label"]}\\n{node["status"]} (v{node["version"]})'
        for node in graph["nodes"]
    }
    lines = [
        "digraph FORGETGRAPH {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fillcolor="#e8f0fe"];',
    ]
    for node_id, label in nodes.items():
        safe_label = label.replace('"', '\\"')
        lines.append(f'  "{node_id}" [label="{safe_label}"];')
    for edge in graph["edges"]:
        lines.append(f'  "{edge["source"]}" -> "{edge["target"]}";')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="FORGETGRAPH", layout="wide")
    st.title("FORGETGRAPH")
    st.caption("Dependency-aware deletion orchestrator for a controlled ML pipeline")

    store = ForgetGraphStore(_database_path())
    try:
        store.bootstrap(SAMPLE_CSV)
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not initialize the pipeline: {error}")
        st.stop()

    state = store.status()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Artifact dependency graph")
        st.graphviz_chart(_graph_dot(state["artifact_graph"]))
        st.caption(
            "Source records -> engineered features -> Ridge model -> prediction cache"
        )
    with right:
        st.subheader("Pipeline overview")
        st.metric("Source records", state["source_record_count"])
        st.metric("Prediction cache entries", state["prediction_cache_count"])
        st.metric("Deletion tombstones", len(state["deletion_registry"]))
        st.code(str(store.database_path), language="text")

    st.subheader("Request deletion")
    with st.form("deletion_request"):
        record_id = st.text_input("Record identifier", placeholder="user-001")
        confirmed = st.checkbox(
            "I confirm this deletion request should be executed in the demo pipeline."
        )
        submit = st.form_submit_button("Delete and rebuild")
    if submit:
        if not record_id.strip():
            st.error("Enter a non-empty record identifier.")
        elif not confirmed:
            st.warning("Confirm the deletion request before executing it.")
        else:
            result = store.request_deletion(record_id)
            if result.execution_status != "completed":
                st.error(f"Execution failed: {result.error}")
            elif result.verification_status != "passed":
                st.error(
                    "Deletion operations completed, but independent model verification "
                    f"is {result.verification_status or 'missing'}."
                )
            elif result.idempotent:
                st.info(
                    f"{record_id} was already deleted and verified. "
                    "No changes were repeated."
                )
            else:
                st.success(
                    f"Deletion run {result.run_id} completed and independently verified."
                )
            st.rerun()

    st.subheader("Managed ingestion")
    st.caption("Deleted identifiers are persistently blocked from this ingestion path.")
    with st.expander("Add one record"):
        with st.form("managed_ingestion"):
            new_id = st.text_input("New record identifier")
            age = st.number_input("Age", min_value=0.0, value=30.0)
            tenure = st.number_input("Tenure (months)", min_value=0.0, value=12.0)
            usage = st.number_input("Monthly usage", min_value=0.0, value=200.0)
            target = st.number_input("Training target", value=60.0)
            ingest_submit = st.form_submit_button("Ingest record")
        if ingest_submit:
            try:
                store.ingest_records(
                    [
                        {
                            "record_id": new_id,
                            "age": age,
                            "tenure_months": tenure,
                            "monthly_usage": usage,
                            "target": target,
                        }
                    ]
                )
            except PipelineRefreshError as error:
                st.error(str(error))
                st.warning(
                    "The new source record was committed. Review artifact status before "
                    "continuing; the derived pipeline refresh did not finish."
                )
            except TombstonedIdentifierError as error:
                st.error(f"Ingestion blocked: {error}")
            except (ValueError, sqlite3.Error) as error:
                st.error(f"Ingestion rejected: {error}")
            else:
                st.success(
                    "Record ingested; features, model, and prediction cache were refreshed."
                )
                st.rerun()

    st.subheader("Source and derived artifacts")
    tab_source, tab_features, tab_predictions = st.tabs(
        ["Source records", "Engineered features", "Prediction cache"]
    )
    with tab_source:
        st.dataframe(store.source_records(), width="stretch", hide_index=True)
    with tab_features:
        st.dataframe(store.feature_records(), width="stretch", hide_index=True)
    with tab_predictions:
        st.dataframe(store.prediction_records(), width="stretch", hide_index=True)

    st.subheader("Execution and verification audit")
    if state["deletion_runs"]:
        st.dataframe(state["deletion_runs"], width="stretch", hide_index=True)
    if state["deletion_steps"]:
        st.dataframe(state["deletion_steps"], width="stretch", hide_index=True)
    if state["verification_results"]:
        st.dataframe(
            [
                {
                    "run_id": result["run_id"],
                    "verification_status": result["status"],
                    **result["metrics"],
                }
                for result in state["verification_results"]
            ],
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Execution status and independent verification status are stored separately. "
        "A completed workflow is not considered verified unless its verification passed."
    )


if __name__ == "__main__":
    main()
