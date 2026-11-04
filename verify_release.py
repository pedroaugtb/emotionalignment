#!/usr/bin/env python3
"""Validate the aggregate-only anonymous supplementary release."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_APPENDIX_SHA256 = "3c604ebd1c51fc184a2bf3d9f3c8c09142a69fe218742b1ffb56609b917f8816"
EXPECTED_INPUT_TABLES = 37
EXPECTED_LEDGER_ROWS = 10064
ALLOWED_SUFFIXES = {".csv", ".md", ".py", ".tex"}
FORBIDDEN_HEADERS = {
    "participant_id", "report_id", "response_id", "worker_id", "subject_id",
    "email", "name", "author", "text", "narrative", "clause_text",
}


def decoded(hex_value: str) -> str:
    return bytes.fromhex(hex_value).decode("utf-8")


# Project-specific identity strings are encoded so the verifier does not itself
# disclose them in plain text. Generic metadata/path markers are listed normally.
FORBIDDEN_TEXT = (
    decoded("706564726f"),
    decoded("62656e746f"),
    decoded("706564726f6175677462"),
    decoded("75666d67"),
    decoded("756e69766572736964616465206665646572616c206465206d696e617320676572616973"),
    decoded("2f686f6d652f"),
    decoded("2f736372617463682f"),
    decoded("5c617574686f727b"),
    decoded("5c616666696c696174696f6e7b"),
    decoded("5c696e737469747574696f6e7b"),
    decoded("5c656d61696c7b"),
    decoded("5c73686f7274617574686f7273"),
)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def text_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            fail(f"symbolic link is not allowed: {relative}")
        if not path.is_file():
            continue
        if "__pycache__" in relative.parts:
            fail(f"excluded metadata/cache is present: {relative}")
        if path.name in {".gitattributes", ".gitignore"}:
            files.append(path)
        elif path.suffix.lower() not in ALLOWED_SUFFIXES:
            fail(f"unexpected file type: {relative}")
        else:
            files.append(path)
    return files


def check_text(files: list[Path]) -> None:
    email = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
    orcid = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", re.I)
    for path in files:
        content = path.read_text(encoding="utf-8")
        folded = content.casefold()
        for value in FORBIDDEN_TEXT:
            if value.casefold() in folded:
                fail(f"identity/absolute-path marker in {path.relative_to(ROOT)}")
        if email.search(content):
            fail(f"email-like string in {path.relative_to(ROOT)}")
        if orcid.search(content):
            fail(f"ORCID-like string in {path.relative_to(ROOT)}")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = {field.casefold() for field in (reader.fieldnames or [])}
        disclosed = fields & FORBIDDEN_HEADERS
        if disclosed:
            fail(f"identifier/free-text columns in {path.relative_to(ROOT)}: {sorted(disclosed)}")
        return list(reader)


def check_tables() -> None:
    source_tables = [p for p in ROOT.rglob("*.csv") if p.name != "RESULTS_MASTER.csv"]
    if len(source_tables) != EXPECTED_INPUT_TABLES:
        fail(f"expected {EXPECTED_INPUT_TABLES} aggregate inputs, found {len(source_tables)}")
    for path in source_tables:
        csv_rows(path)

    individual = ROOT / "outputs/individual_vs_population_alignment_v1/RESULTS.csv"
    individual_rows = csv_rows(individual)
    if len(individual_rows) != 192:
        fail(f"unexpected individual-vs-population aggregate row count: {len(individual_rows)}")
    if any(row.get("record_type") == "participant" for row in individual_rows):
        fail("participant-level individual-vs-population rows are present")

    mismatch = ROOT / "outputs/isear_compact_gpt_oss_v1/rsa/tables/rsa_participant_emotion_mismatch.csv"
    mismatch_rows = csv_rows(mismatch)
    if len(mismatch_rows) != 7 or any(row.get("scope") != "group_average" for row in mismatch_rows):
        fail("participant-level mismatch rows are present")


def rebuild_and_check() -> None:
    subprocess.run([sys.executable, str(ROOT / "build_appendix.py")], cwd=ROOT, check=True)
    appendix = ROOT / "APPENDIX_COMPLETE.tex"
    digest = hashlib.sha256(appendix.read_bytes()).hexdigest()
    if digest != EXPECTED_APPENDIX_SHA256:
        fail(f"appendix digest mismatch: {digest}")

    with (ROOT / "RESULTS_MASTER.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_LEDGER_ROWS:
        fail(f"expected {EXPECTED_LEDGER_ROWS} ledger rows, found {len(rows)}")

    labels = re.findall(r"\\label\{([^}]+)\}", appendix.read_text(encoding="utf-8"))
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        fail(f"duplicate LaTeX labels: {duplicates}")


def main() -> None:
    files = text_files()
    check_text(files)
    check_tables()
    rebuild_and_check()
    check_text(text_files())
    print(
        "PASS: anonymous aggregate release; "
        f"{EXPECTED_INPUT_TABLES} source tables, {EXPECTED_LEDGER_ROWS} ledger rows, "
        f"appendix sha256={EXPECTED_APPENDIX_SHA256}"
    )


if __name__ == "__main__":
    main()
