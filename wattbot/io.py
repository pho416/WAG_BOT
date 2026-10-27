from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

FLAGS = ("Quote", "Table", "Figure", "Math", "is_NA", "CrossPaper", "Reconcile")
FIELDS = ("id", "question", "answer", "answer_value", "answer_unit", "ref_id",
          "ref_url", "supporting_materials", "explanation", "Cohort")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def refs(value):
    value = str(value or "").strip()
    if value.lower() in ("", "is_blank", "na", "n/a"):
        return []
    if value.startswith("["):
        result = ast.literal_eval(value)
        if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
            raise ValueError(f"Invalid ref_id list: {value}")
        return sorted(set(result))
    return [value]


def questions(path):
    """Explicit allowlist: gold values, flags, references NEVER enter inference."""
    rows = read_csv(path)
    if not rows or not {"id", "question", "answer_unit"}.issubset(rows[0]):
        raise ValueError("Question CSV needs id, question, answer_unit")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError("Question IDs must be nonempty and unique")
    return [{k: r.get(k, "") for k in ("id", "question", "answer_unit", "Cohort")} for r in rows]
