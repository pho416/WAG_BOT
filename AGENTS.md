# Working on WattBot

Read `README.md` and `docs/BASELINE.md` for measured results, environment setup and remaining work.

- Goal: implement the user's evidence-first WattBot roadmap, starting with reproducible text RAG and official evaluation.
- Keep competition CSVs, original `Score.py`, PDFs, indexes, predictions and model files out of Git. Import the user's `WattBot2026.zip` with `python -m wattbot unpack`.
- Never include train answers, gold citations, supporting materials or evidence flags in inference inputs or retrieval indexes. Only `questions()` supplies inference rows.
- Preserve the official scorer byte-for-byte. Use strict ID/evidence validation and the independent 2026 breakdown to handle documented diagnostic inconsistencies.
- Separate retrieval recall from answer accuracy. Report which components actually ran and which are only implemented.
- CPU verification: `python -m unittest discover -s tests -v`, then the relevant CLI commands. GPU smoke/train commands are in `scripts/run_server.sh`.
- Linux GPU setup uses a separate `.venv`. Select one available GPU explicitly with `GPU_ID`; inspect free memory and never stop unrelated processes.
- Use a new run directory when changing model, prompt, retrieval or generation code. Preserve fingerprints, model revisions and per-question traces.
- Next work: GPU smoke and full train score, error analysis, then dense/reranking ablations and multimodal/multi-hop improvements justified by measured failures.
