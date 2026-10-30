from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from .core import (
    DELETION_STEPS,
    ForgetGraphStore,
    PipelineRefreshError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE = PROJECT_ROOT / ".forgetgraph" / "forgetgraph.sqlite3"
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_records.csv"


def _store(path: str | None) -> ForgetGraphStore:
    store = ForgetGraphStore(path or os.environ.get("FORGETGRAPH_DB", DEFAULT_DATABASE))
    store.bootstrap(SAMPLE_CSV)
    return store


def main() -> int:
    parser = argparse.ArgumentParser(description="FORGETGRAPH deletion orchestrator")
    parser.add_argument("--db", help="SQLite database path (or set FORGETGRAPH_DB)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="initialize the demo pipeline")
    ingest = commands.add_parser("ingest", help="ingest records from a CSV file")
    ingest.add_argument("csv_path")
    delete = commands.add_parser("delete", help="delete a record and rebuild dependent artifacts")
    delete.add_argument("record_id")
    delete.add_argument(
        "--fail-at",
        choices=DELETION_STEPS,
        help="simulate an execution failure at a workflow step",
    )
    commands.add_parser("status", help="show current artifacts and deletion audit")

    args = parser.parse_args()
    store = _store(args.db)

    if args.command == "init":
        print(f"FORGETGRAPH initialized. SQLite database: {store.database_path}")
        return 0
    if args.command == "ingest":
        try:
            count = store.ingest_csv(args.csv_path)
        except (OSError, ValueError, sqlite3.Error, PipelineRefreshError) as error:
            print(f"Ingestion failed: {error}", file=sys.stderr)
            return 1
        print(f"Ingested {count} record(s) and refreshed derived artifacts.")
        return 0
    if args.command == "delete":
        result = store.request_deletion(args.record_id, fail_at=args.fail_at)
        print(json.dumps(result.__dict__, indent=2))
        return 0 if result.execution_status == "completed" else 1

    print(json.dumps(store.status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
