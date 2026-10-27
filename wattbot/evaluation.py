from __future__ import annotations

import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .io import FLAGS, questions, read_csv, read_jsonl, refs, sha256, write_csv, write_json


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("wattbot_official_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(raw_dir, output):
    root = Path(raw_dir)
    train, test, metadata = [read_csv(root / f) for f in ("train_QA.csv", "test_Q.csv", "metadata.csv")]
    for name, rows in (("train", train), ("test", test), ("metadata", metadata)):
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs in {name}")
    if {r["id"] for r in train} & {r["id"] for r in test}:
        raise ValueError("Train/test ID overlap")
    known = {r["id"] for r in metadata}
    unknown = sorted({ref for r in train for ref in refs(r["ref_id"])} - known)
    if unknown:
        raise ValueError(f"Train references absent from metadata: {unknown}")
    scorer = load_scorer(root / "Score.py")
    truth = pd.read_csv(root / "train_QA.csv", keep_default_na=False, dtype=str)
    sanity = scorer.score(truth.copy(), truth.copy(), "id", verbose=False)
    if abs(sanity - 1.0) > 1e-12:
        raise ValueError(f"Official scorer self-check failed: {sanity}")
    result = {"train": len(train), "test": len(test), "documents": len(metadata),
              "flags": {f: sum(r[f] == "1" for r in train) for f in FLAGS},
              "cohorts": dict(Counter(r["Cohort"] for r in train)),
              "reference_counts": dict(Counter(len(refs(r["ref_id"])) for r in train)),
              "scorer_self_check": sanity,
              "hashes": {f: sha256(root / f) for f in ("train_QA.csv", "test_Q.csv", "metadata.csv", "Score.py")},
              "scorer_notes": [
                  "LAST_BREAKDOWN per-type scores use obsolete 0.75/0.15/0.10 weights; use our independent 2026 breakdown.",
                  "score() joins only answer_value/answer_unit/ref_id before row_bits, so ref_url/supporting_materials are not checked by the headline metric. Our validator checks them.",
                  "score() permits partial ID overlap; our evaluator rejects missing, extra and duplicate IDs.",
                  "is_blank in answer_unit denotes an unspecified/categorical unit, NOT an unanswerable question."]}
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def split(raw_dir, output_dir, seed=2026, fraction=0.2):
    # Greedy multi-label holdout. Flags are used only here/evaluation, never inference.
    rows = read_csv(Path(raw_dir) / "train_QA.csv")
    n = round(len(rows) * fraction)
    if not 0 < n < len(rows):
        raise ValueError("Validation fraction must leave rows in both splits")
    rng = random.Random(seed)
    remaining = rows.copy()
    rng.shuffle(remaining)
    labels = {r["id"]: {f for f in FLAGS if r[f] == "1"} | {"Cohort:" + r["Cohort"]} for r in rows}
    totals = Counter(t for r in rows for t in labels[r["id"]])
    targets = {t: max(1, round(c * fraction)) for t, c in totals.items()}
    current, validation = Counter(), []
    while len(validation) < n:
        def priority(row):
            return sum((targets[t] - current[t]) / totals[t] for t in labels[row["id"]])
        best = max(remaining, key=priority)
        remaining.remove(best)
        validation.append(best)
        current.update(labels[best["id"]])
    root = Path(output_dir)
    write_json(root / "split.json", {"seed": seed, "fraction": fraction,
                                    "development_ids": sorted(r["id"] for r in remaining),
                                    "validation_ids": sorted(r["id"] for r in validation),
                                    "validation_labels": dict(current),
                                    "note": "Multilabel question holdout; not document-disjoint. Gold is for evaluation only."})
    for name, subset in (("development", remaining), ("validation", validation)):
        subset.sort(key=lambda r: r["id"])
        write_csv(root / f"{name}_gold.csv", subset, fields=list(rows[0]))
        write_csv(root / f"{name}_questions.csv", subset, fields=("id", "question", "answer_unit", "Cohort"))


def validate(submission, question_file, metadata):
    rows = read_csv(submission)
    qs = {q["id"]: q for q in questions(question_file)}
    ids = [r.get("id", "") for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(qs):
        raise ValueError(f"Submission IDs must exactly match questions: missing={sorted(set(qs)-set(ids))[:10]}, extra={sorted(set(ids)-set(qs))[:10]}, duplicates={len(ids)-len(set(ids))}")
    known = {r["id"]: r["url"] for r in read_csv(metadata)}
    for row in rows:
        rid = row["id"]
        for col in ("answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials", "explanation"):
            if col not in row or not row[col].strip():
                raise ValueError(f"{rid}: missing or empty {col}")
        if row["answer_unit"] != qs[rid]["answer_unit"]:
            raise ValueError(f"{rid}: expected unit must be copied unchanged")
        if row["answer_value"].lower() in ("is_blank", "na", "n/a"):
            if any(row[c] != "is_blank" for c in ("answer_value", "ref_id", "ref_url", "supporting_materials")):
                raise ValueError(f"{rid}: abstention must blank all evidence fields")
        else:
            cited = refs(row["ref_id"])
            if not cited or set(cited) - set(known):
                raise ValueError(f"{rid}: empty or unknown citations {cited}")
            urls = refs(row["ref_url"])
            if set(urls) != {known[r] for r in cited}:
                raise ValueError(f"{rid}: ref_url does not match metadata citations")
    return rows


def evaluate(gold, submission, metadata, scorer_path, output_dir):
    rows = validate(submission, gold, metadata)
    scorer = load_scorer(scorer_path)
    sol = pd.read_csv(gold, dtype=str, keep_default_na=False)
    sub = pd.DataFrame(rows)
    official = scorer.score(sol.copy(), sub.copy(), "id", verbose=False)
    by_id = {r["id"]: r for r in rows}
    errors = []
    for truth in sol.to_dict("records"):
        predicted = by_id[truth["id"]]
        bits = scorer.row_bits(pd.Series(truth), pd.Series(predicted))
        precision, recall, f1 = scorer.ref_prf(truth["ref_id"], predicted["ref_id"])
        errors.append({"id": truth["id"], "question": truth["question"],
                       "gold_value": truth["answer_value"], "predicted_value": predicted["answer_value"],
                       "gold_refs": truth["ref_id"], "predicted_refs": predicted["ref_id"],
                       "value_match": bool(bits["val"]), "na_match": bool(bits["na"]),
                       "citation_precision": precision, "citation_recall": recall, "citation_f1": f1,
                       "score_2026": 0.75 * bits["val"] + 0.20 * f1 + 0.05 * bits["na"],
                       **{f: truth.get(f, "0") for f in FLAGS}})
    frame = pd.DataFrame(errors)
    groups = {"ALL": frame}
    groups.update({f: frame[frame[f] == "1"] for f in FLAGS})
    report = {"official_score": official, "strict_score_2026": float(frame.score_2026.mean()),
              "scorer_sha256": sha256(scorer_path), "prediction_sha256": sha256(submission),
              "by_type": {name: {"n": len(group), "answer_accuracy": float(group.value_match.mean()),
                                  "citation_f1": float(group.citation_f1.mean()), "score_2026": float(group.score_2026.mean())}
                          for name, group in groups.items() if len(group)}}
    root = Path(output_dir)
    write_csv(root / "rows.csv", errors, fields=list(errors[0]))
    write_json(root / "metrics.json", report)
    print(json.dumps(report, indent=2))
    return report


def retrieval_metrics(gold_path, retrieval_path, output, ks=(5, 10, 30, 50)):
    gold = {r["id"]: r for r in read_csv(gold_path)}
    retrieved = read_jsonl(retrieval_path)
    ids = [r["id"] for r in retrieved]
    if len(ids) != len(set(ids)) or set(ids) != set(gold):
        raise ValueError("Retrieval IDs must exactly match gold IDs")
    details = []
    for item in retrieved:
        row = gold[item["id"]]
        expected = set(refs(row["ref_id"]))
        if not expected:
            continue  # no gold evidence for NA questions, never count as perfect recall
        result = {"id": item["id"], "gold_refs": sorted(expected), **{f: row.get(f, "0") for f in FLAGS}}
        for k in ks:
            chunk_refs = {h["ref_id"] for h in item["hits"][:k]}
            document_refs = list(dict.fromkeys(h["ref_id"] for h in item["hits"]))[:k]
            result[f"chunk_recall@{k}"] = len(expected & chunk_refs) / len(expected)
            result[f"document_recall@{k}"] = len(expected & set(document_refs)) / len(expected)
            result[f"all_refs_in_chunks@{k}"] = int(expected <= chunk_refs)
        context_refs = {h["ref_id"] for h in item.get("context", [])}
        result["context_recall"] = len(expected & context_refs) / len(expected)
        details.append(result)
    if not details:
        raise ValueError("No answerable gold rows for retrieval evaluation")
    metrics = [k for k in details[0] if "recall" in k or k.startswith("all_refs")]
    report = {"n_answerable": len(details), "n_excluded_na": len(gold) - len(details),
              "retrieval_sha256": sha256(retrieval_path),
              "by_type": {name: {"n": len(group), **{k: sum(d[k] for d in group) / len(group) for k in metrics}}
                          for name in ("ALL", *FLAGS)
                          if (group := [d for d in details if name == "ALL" or d.get(name) == "1"])},
              "rows": details}
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report
