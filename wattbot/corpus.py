from __future__ import annotations

import concurrent.futures
import json
import re
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests

from .io import read_csv, sha256, write_json, write_jsonl


def pdf_url(url):
    # Preserve the supplied version, never silently substitute a newer revision.
    return re.sub(r"(https?://(?:www\.)?arxiv.org)/abs/", r"\1/pdf/", url.strip())


def valid_pdf(path):
    import pymupdf
    with Path(path).open("rb") as f:
        if not f.read(1024).lstrip().startswith(b"%PDF-"):
            raise ValueError("Response is not a PDF")
    with pymupdf.open(path) as doc:
        if not doc.is_pdf or doc.page_count < 1 or doc.needs_pass:
            raise ValueError("Unreadable, empty or encrypted PDF")
        return doc.page_count


def download(metadata, pdf_dir, manifest, workers=3, delay=3.0, limit=None, only_failed=False):
    if workers < 1 or delay < 0 or (limit is not None and limit < 1):
        raise ValueError("Require workers >= 1, delay >= 0 and limit >= 1")
    rows = read_csv(metadata)
    root = Path(pdf_dir)
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads(Path(manifest).read_text(encoding="utf-8")) if Path(manifest).exists() else {}
    if only_failed:
        rows = [r for r in rows if previous.get(r["id"], {}).get("status") != "ok"]
    if limit:
        rows = rows[:limit]
    lock = threading.Lock()
    host_next = {}

    def one(row):
        rid = row["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
            raise ValueError(f"Unsafe metadata id: {rid}")
        dest = root / (rid + ".pdf")
        url = pdf_url(row["url"])
        record = {"id": rid, "metadata_url": row["url"], "download_url": url,
                  "version_pinned": bool(re.search(r"v\d+(?:\.pdf)?$", url)) if "arxiv.org/" in url else None}
        if dest.exists():
            try:
                pages = valid_pdf(dest)
                old = previous.get(rid, {})
                same = old.get("sha256") == sha256(dest)
                record.update(status="ok", pages=pages, sha256=sha256(dest), cached=True,
                              provenance="download" if same and old.get("provenance") == "download" else "local_unverified",
                              resolved_url=old.get("resolved_url") if same else None)
                return rid, record
            except Exception:
                pass
        for attempt in range(3):
            try:
                host = urlparse(url).netloc
                with lock:
                    wait = max(0, host_next.get(host, 0) - time.monotonic())
                    host_next[host] = time.monotonic() + wait + delay
                time.sleep(wait)
                user_agent = ("WattBotResearchBaseline/0.1 (academic PDF corpus retrieval)" if attempt == 0
                              else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36")
                with requests.get(url, stream=True, timeout=(15, 60), headers={
                    "User-Agent": user_agent, "Accept": "application/pdf,*/*;q=0.8"}) as response:
                    response.raise_for_status()
                    temp = dest.with_suffix(".pdf.part")
                    size = 0
                    with temp.open("wb") as f:
                        for block in response.iter_content(65536):
                            size += len(block)
                            if size > 100 * 1024 * 1024:
                                raise ValueError("PDF exceeded 100 MiB download limit")
                            f.write(block)
                    # PyMuPDF detects the format despite the temporary extension.
                    pages = valid_pdf(temp)
                    temp.replace(dest)
                    record.update(status="ok", pages=pages, bytes=size, sha256=sha256(dest),
                                  cached=False, provenance="download", resolved_url=response.url)
                    return rid, record
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}", attempts=attempt + 1)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return rid, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            rid, result = future.result()
            previous[rid] = result
            write_json(manifest, previous)
            print(f"{rid}: {result['status']}" + (f" - {result['error'][:180]}" if result['status'] != "ok" else ""), flush=True)
    summary = dict(Counter(x["status"] for x in previous.values()))
    print(json.dumps(summary))
    return previous


def ingest(metadata, pdf_dir, out_dir, chunk_words=300, overlap=60):
    import pymupdf
    if chunk_words < 20 or overlap < 0 or overlap >= chunk_words:
        raise ValueError("Require chunk_words >= 20 and 0 <= overlap < chunk_words")
    out = Path(out_dir)
    pages, chunks, docs, failures = [], [], [], []
    for row in read_csv(metadata):
        rid = row["id"]
        path = Path(pdf_dir) / (rid + ".pdf")
        if not path.exists():
            failures.append({"ref_id": rid, "reason": "missing_pdf"})
            continue
        try:
            with pymupdf.open(path) as doc:
                if not doc.is_pdf or doc.needs_pass:
                    raise ValueError("Invalid or encrypted PDF")
                doc_pages, doc_chunks = [], []
                for page_no, page in enumerate(doc, 1):
                    blocks = page.get_text("blocks", sort=True)
                    text_blocks = [{"bbox": list(b[:4]), "text": b[4]} for b in blocks if b[6] == 0 and b[4].strip()]
                    text = "\n".join(b["text"] for b in text_blocks)
                    common = {"ref_id": rid, "page": page_no, "title": row["title"],
                              "year": row["year"], "url": row["url"]}
                    doc_pages.append({**common, "text": text, "blocks": text_blocks,
                                      "width": page.rect.width, "height": page.rect.height,
                                      "needs_ocr": len(text.strip()) < 40})
                    # Page-bounded chunks, with source block boxes and immutable page provenance.
                    words = list(re.finditer(r"\S+", text))
                    for start in range(0, len(words), chunk_words - overlap):
                        end = min(start + chunk_words, len(words))
                        a, b = words[start].start(), words[end - 1].end()
                        excerpt = text[a:b]
                        if not excerpt.strip():
                            continue
                        cursor, boxes = 0, []
                        for block in text_blocks:
                            stop = cursor + len(block["text"])
                            if cursor < b and stop > a:
                                boxes.append(block["bbox"])
                            cursor = stop + 1
                        captions = re.findall(r"(?im)^\s*((?:fig(?:ure)?\.?|table)\s*\d+[^\n]*)", excerpt)
                        doc_chunks.append({**common, "chunk_id": f"{rid}:p{page_no}:w{start}",
                                           "text": excerpt, "char_start": a, "char_end": b,
                                           "bboxes": boxes, "section": None, "kind": "text",
                                           "captions": captions})
                        if end == len(words):
                            break
                docs.append({**row, "sha256": sha256(path), "pages": doc.page_count})
                pages.extend(doc_pages)
                chunks.extend(doc_chunks)
        except Exception as exc:
            failures.append({"ref_id": rid, "reason": str(exc)})
    if not chunks:
        raise ValueError("No PDF text found. Download PDFs or provide --pdf-dir first.")
    write_jsonl(out / "pages.jsonl", pages)
    write_jsonl(out / "chunks.jsonl", chunks)
    report = {"documents": len(docs), "pages": len(pages), "chunks": len(chunks),
              "pages_needing_ocr": sum(p["needs_ocr"] for p in pages), "failures": failures,
              "chunk_words": chunk_words, "overlap": overlap, "sources": docs}
    write_json(out / "ingestion.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, ensure_ascii=False))
    return report


def render(pdf_dir, ref_id, page, output, dpi=140):
    import pymupdf
    if not re.fullmatch(r"[A-Za-z0-9_-]+", ref_id):
        raise ValueError("Invalid ref_id")
    if page < 1 or dpi < 20 or dpi > 600:
        raise ValueError("Use a 1-based page number and DPI in [20,600]")
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(Path(pdf_dir) / f"{ref_id}.pdf") as doc:
        doc[page - 1].get_pixmap(dpi=dpi).save(dest)
