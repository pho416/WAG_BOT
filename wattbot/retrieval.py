from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .io import read_jsonl, sha256, write_json

# Small fixed list, independent of train answers. Keep negation and domain vocabulary.
STOP = set("a an the is are was were be been being of in on to for from with by as at and or that this these those what which who how according paper study report does do did it its than much many".split())


def tokenize(text):
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text.lower())
    return [t for t in re.findall(r"[\w]+(?:\.[0-9]+)?", text) if t not in STOP]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        tokens = [tokenize(t) for t in texts]
        self.n = len(tokens)
        self.lengths = np.array([len(t) for t in tokens], dtype=np.float32)
        avg = max(float(self.lengths.mean()), 1.0)
        norm = k1 * (1 - b + b * self.lengths / avg)
        postings = defaultdict(list)
        for i, toks in enumerate(tokens):
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        self.index = {}
        for term, occurrences in postings.items():
            ids = np.array([x[0] for x in occurrences], dtype=np.int32)
            tf = np.array([x[1] for x in occurrences], dtype=np.float32)
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            self.index[term] = ids, idf * tf * (k1 + 1) / (tf + norm[ids])

    def scores(self, query):
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term in self.index:
                ids, weights = self.index[term]
                scores[ids] += weights
        return scores


def top_indices(scores, k, positive=False):
    # Stable ordering also makes ties reproducible across runs.
    indices = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in indices if not positive or scores[i] > 0]


def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1 / (k + rank)
    return dict(scores)


def embed(chunks_path, index_dir, model_name, revision=None, batch_size=32, device="cpu"):
    from sentence_transformers import SentenceTransformer
    chunks = read_jsonl(chunks_path)
    model = SentenceTransformer(model_name, revision=revision, device=device, trust_remote_code=False)
    texts = [c["title"] + "\n" + c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    dest = Path(index_dir)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "embeddings.npy", vectors.astype(np.float32))
    resolved = getattr(model[0].auto_model.config, "_commit_hash", None)
    write_json(dest / "dense.json", {"model": model_name, "revision": resolved or revision,
                                    "requested_revision": revision,
                                    "chunks_sha256": sha256(chunks_path), "count": len(chunks),
                                    "dimensions": vectors.shape[1]})


class Retriever:
    def __init__(self, chunks_path, dense_dir=None, device="cpu", reranker=None, reranker_revision=None):
        self.chunks = read_jsonl(chunks_path)
        if not self.chunks:
            raise ValueError("Empty chunk corpus")
        self.bm25 = BM25([c["title"] + "\n" + c["text"] for c in self.chunks])
        self.model = self.vectors = self.reranker = None
        self.config = {"chunks_sha256": sha256(chunks_path), "method": "bm25",
                       "reranker": reranker, "reranker_revision": reranker_revision}
        if dense_dir:
            from sentence_transformers import SentenceTransformer
            info = json.loads((Path(dense_dir) / "dense.json").read_text(encoding="utf-8"))
            if info["chunks_sha256"] != sha256(chunks_path):
                raise ValueError("Dense index belongs to different chunks; rebuild it")
            self.vectors = np.load(Path(dense_dir) / "embeddings.npy", mmap_mode="r")
            if self.vectors.shape[0] != len(self.chunks):
                raise ValueError("Dense index row count mismatch")
            self.model = SentenceTransformer(info["model"], revision=info.get("revision"),
                                             device=device, trust_remote_code=False)
            self.config.update(method="bm25+dense_rrf", dense=info)
        if reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker, revision=reranker_revision, device=device, trust_remote_code=False)
            self.config["reranker_revision"] = getattr(self.reranker.model.config, "_commit_hash", None) or reranker_revision

    def search(self, query, top_k=50, candidates=80):
        candidates = max(candidates, top_k)
        lexical = self.bm25.scores(query)
        ranking = top_indices(lexical, candidates, positive=True)
        scores = {i: float(lexical[i]) for i in ranking}
        if self.model is not None:
            vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            semantic = self.vectors @ vector
            scores = rrf([ranking, top_indices(semantic, candidates)])
            ranking = sorted(scores, key=lambda i: (-scores[i], i))[:candidates]
        if self.reranker is not None and ranking:
            reranked = self.reranker.predict([(query, self.chunks[i]["title"] + "\n" + self.chunks[i]["text"]) for i in ranking])
            scores = {i: float(s) for i, s in zip(ranking, reranked)}
            ranking.sort(key=lambda i: (-scores[i], i))
        return [{**self.chunks[i], "retrieval_score": scores[i]} for i in ranking[:top_k]]


def select_context(hits, count=8, per_page=1):
    """Deduplicate overlapping chunks on a page while keeping document diversity."""
    selected, counts = [], Counter()
    for hit in hits:
        key = hit["ref_id"], hit["page"]
        if counts[key] >= per_page:
            continue
        selected.append(hit)
        counts[key] += 1
        if len(selected) >= count:
            break
    return selected
