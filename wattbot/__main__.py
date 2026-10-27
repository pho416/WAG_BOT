from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from .io import questions, sha256, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Evidence-first WattBot baseline (local CPU / Kaggle / Colab)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("unpack")
    p.add_argument("--archive", default="WattBot2026.zip")
    p.add_argument("--raw-dir", default="data/raw")
    p = sub.add_parser("profile")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--output", default="artifacts/dataset_profile.json")
    p = sub.add_parser("split")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--output-dir", default="artifacts/splits")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--fraction", type=float, default=0.2)
    p = sub.add_parser("download")
    p.add_argument("--metadata", default="data/raw/metadata.csv")
    p.add_argument("--pdf-dir", default="data/pdfs")
    p.add_argument("--manifest", default="artifacts/download_manifest.json")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--only-failed", action="store_true")
    p = sub.add_parser("ingest")
    p.add_argument("--metadata", default="data/raw/metadata.csv")
    p.add_argument("--pdf-dir", default="data/pdfs")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--chunk-words", type=int, default=300)
    p.add_argument("--overlap", type=int, default=60)
    p = sub.add_parser("embed")
    p.add_argument("--chunks", default="data/processed/chunks.jsonl")
    p.add_argument("--index-dir", default="artifacts/dense")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p = sub.add_parser("retrieve")
    p.add_argument("--questions", default="data/raw/train_QA.csv")
    p.add_argument("--chunks", default="data/processed/chunks.jsonl")
    p.add_argument("--dense-dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reranker")
    p.add_argument("--reranker-revision")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--context-k", type=int, default=8)
    p.add_argument("--output", default="runs/bm25/train_retrieval.jsonl")
    p = sub.add_parser("retrieval-eval")
    p.add_argument("--gold", default="data/raw/train_QA.csv")
    p.add_argument("--retrieval", default="runs/bm25/train_retrieval.jsonl")
    p.add_argument("--output", default="runs/bm25/retrieval_metrics.json")
    p = sub.add_parser("generate")
    p.add_argument("--questions", default="data/raw/train_QA.csv")
    p.add_argument("--retrieval", default="runs/bm25/train_retrieval.jsonl")
    p.add_argument("--metadata", default="data/raw/metadata.csv")
    p.add_argument("--output-dir", default="runs/local_train")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--max-input-tokens", type=int, default=6500)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("evaluate")
    p.add_argument("--gold", default="data/raw/train_QA.csv")
    p.add_argument("--submission", required=True)
    p.add_argument("--metadata", default="data/raw/metadata.csv")
    p.add_argument("--scorer", default="data/raw/Score.py")
    p.add_argument("--output-dir", default="runs/evaluation")
    p = sub.add_parser("validate")
    p.add_argument("--submission", required=True)
    p.add_argument("--questions", default="data/raw/test_Q.csv")
    p.add_argument("--metadata", default="data/raw/metadata.csv")
    p = sub.add_parser("abstain-baseline")
    p.add_argument("--questions", default="data/raw/train_QA.csv")
    p.add_argument("--output", default="runs/abstain/train.csv")
    p = sub.add_parser("render")
    p.add_argument("--pdf-dir", default="data/pdfs")
    p.add_argument("--ref-id", required=True)
    p.add_argument("--page", type=int, required=True)
    p.add_argument("--output", required=True)
    sub.add_parser("doctor")
    args = parser.parse_args()
    if args.command == "unpack":
        import zipfile
        root = Path(args.raw_dir)
        required = {"metadata.csv", "train_QA.csv", "test_Q.csv", "Score.py"}
        allowed = required | {"CONTRIBUTE_QUESTIONS.md", "WRITEUP_TEMPLATE.md"}
        with zipfile.ZipFile(args.archive) as archive:
            names = set(archive.namelist())
            if not required <= names:
                raise ValueError(f"Dataset archive missing: {sorted(required - names)}")
            payloads = {}
            for name in sorted(allowed & names):
                if archive.getinfo(name).file_size > 10 * 1024 * 1024:
                    raise ValueError(f"Unexpectedly large dataset file: {name}")
                payloads[name] = archive.read(name)
                dest = root / name
                if dest.exists() and dest.read_bytes() != payloads[name]:
                    raise ValueError(f"Existing {dest} differs. Choose another --raw-dir to preserve the original.")
            root.mkdir(parents=True, exist_ok=True)
            for name, content in payloads.items():
                (root / name).write_bytes(content)
        print(f"Dataset unpacked to {root}")
    elif args.command == "profile":
        from .evaluation import profile
        profile(args.raw_dir, args.output)
    elif args.command == "split":
        from .evaluation import split
        split(args.raw_dir, args.output_dir, args.seed, args.fraction)
    elif args.command == "download":
        from .corpus import download
        results = download(args.metadata, args.pdf_dir, args.manifest, args.workers, args.delay, args.limit, args.only_failed)
        if any(r["status"] != "ok" for r in results.values()):
            parser.exit(2, "Some PDFs are unavailable. See download_manifest.json; retry with --only-failed.\n")
    elif args.command == "ingest":
        from .corpus import ingest
        ingest(args.metadata, args.pdf_dir, args.output_dir, args.chunk_words, args.overlap)
    elif args.command == "embed":
        from .retrieval import embed
        embed(args.chunks, args.index_dir, args.model, args.revision, args.batch_size, args.device)
    elif args.command == "retrieve":
        from .retrieval import Retriever, select_context
        if args.top_k < 1 or args.context_k < 1 or args.context_k > args.top_k:
            parser.error("Require 1 <= context-k <= top-k")
        retriever = Retriever(args.chunks, args.dense_dir, args.device, args.reranker, args.reranker_revision)
        records = []
        for q in questions(args.questions):
            hits = retriever.search(q["question"], top_k=args.top_k)
            records.append({**q, "hits": hits, "context": select_context(hits, count=args.context_k)})
        write_jsonl(args.output, records)
        write_json(str(args.output) + ".config.json", {**retriever.config,
                   "question_sha256": sha256(args.questions), "top_k": args.top_k, "context_k": args.context_k,
                   "retrieval_code_sha256": sha256(Path(__file__).with_name("retrieval.py"))})
        print(f"Retrieved evidence for {len(records)} questions -> {args.output}")
    elif args.command == "retrieval-eval":
        from .evaluation import retrieval_metrics
        retrieval_metrics(args.gold, args.retrieval, args.output)
    elif args.command == "generate":
        from .generation import generate
        generate(args.questions, args.retrieval, args.metadata, args.output_dir, args.model, args.revision,
                 quantize=not args.no_quantize, verify=not args.no_verify,
                 max_input_tokens=args.max_input_tokens, limit=args.limit)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args.gold, args.submission, args.metadata, args.scorer, args.output_dir)
    elif args.command == "validate":
        from .evaluation import validate
        rows = validate(args.submission, args.questions, args.metadata)
        print(f"Validated {len(rows)} rows")
    elif args.command == "abstain-baseline":
        from .io import write_csv
        rows = [{**q, "answer": "Unable to answer from the supplied evidence.", "answer_value": "is_blank",
                 "ref_id": "is_blank", "ref_url": "is_blank", "supporting_materials": "is_blank",
                 "explanation": "All-abstain diagnostic baseline; no answer generation was performed."}
                for q in questions(args.questions)]
        write_csv(args.output, rows)
        print(f"Wrote {len(rows)} diagnostic abstentions; this is a scoring floor, not a RAG result")
    elif args.command == "render":
        from .corpus import render
        render(args.pdf_dir, args.ref_id, args.page, args.output)
    elif args.command == "doctor":
        versions = {}
        for package in ("numpy", "pandas", "PyMuPDF", "requests", "torch", "transformers", "sentence-transformers", "bitsandbytes"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        print(json.dumps(versions, indent=2))


if __name__ == "__main__":
    main()
