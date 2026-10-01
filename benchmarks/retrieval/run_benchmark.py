from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_QUESTIONS = Path(__file__).with_name("questions.json")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results_hybrid_x3")
COLLECTION_NAME = "langchain"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
GENERATION_MODEL = "MiniMax-M2.7"
HYBRID_ALPHA = 0.4
TOP_K = 3
ANSWER_METHOD = "hybrid"
ANSWER_RUNS = 3


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small RAG retrieval benchmark.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate three MiniMax answers from the Hybrid Top-3 context.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a matching checkpoint.")
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique.")
    return questions


def load_vectorstore(db_dir: Path) -> tuple[Chroma, list[Chunk]]:
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )
    raw = db.get(include=["documents", "metadatas"])
    chunks = []
    for text, metadata in zip(raw["documents"], raw["metadatas"], strict=True):
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError("Every chunk must have metadata.chunk_id.")
        chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
    chunks.sort(key=lambda chunk: chunk.chunk_id)
    if len(chunks) != len({chunk.chunk_id for chunk in chunks}):
        raise ValueError("Chunk IDs must be unique.")
    return db, chunks


def validate_gold(questions: list[dict[str, Any]], chunks: list[Chunk]) -> None:
    known = {chunk.chunk_id for chunk in chunks}
    for question in questions:
        for group in question["evidence_groups"]:
            unknown = set(group) - known
            if unknown:
                raise ValueError(f"{question['id']} has unknown gold chunks: {sorted(unknown)}")


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def rank_bm25(
    query: str,
    chunks: list[Chunk],
    index: BM25Okapi,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    scores = index.get_scores(jieba.lcut(query))
    order = np.argsort(-scores, kind="stable")[:TOP_K]
    latency_ms = (time.perf_counter() - started) * 1000
    ranked = [
        {
            "rank": rank,
            "chunk_id": chunks[index_value].chunk_id,
            "score": float(scores[index_value]),
        }
        for rank, index_value in enumerate(order, start=1)
    ]
    return ranked, latency_ms


def vector_distances(db: Chroma, query: str, count: int) -> tuple[dict[str, float], float]:
    started = time.perf_counter()
    results = db.similarity_search_with_score(query, k=count)
    latency_ms = (time.perf_counter() - started) * 1000
    distances: dict[str, float] = {}
    for document, distance in results:
        chunk_id = document.metadata["chunk_id"]
        distances[chunk_id] = float(distance)
    return distances, latency_ms


def rank_vector(distances: dict[str, float]) -> list[dict[str, Any]]:
    ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))[:TOP_K]
    return [
        {"rank": rank, "chunk_id": chunk_id, "score": -distance, "distance": distance}
        for rank, (chunk_id, distance) in enumerate(ordered, start=1)
    ]


def rank_hybrid(
    chunks: list[Chunk],
    bm25_scores: np.ndarray,
    distances: dict[str, float],
) -> list[dict[str, Any]]:
    max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) else 0.0
    if max_bm25 > 0:
        normalized_bm25 = bm25_scores / (max_bm25 + 1e-8)
    else:
        normalized_bm25 = np.zeros_like(bm25_scores)

    combined = []
    for index, chunk in enumerate(chunks):
        vector_score = 1.0 - distances.get(chunk.chunk_id, 1.0)
        score = HYBRID_ALPHA * float(normalized_bm25[index]) + (1 - HYBRID_ALPHA) * vector_score
        combined.append((chunk.chunk_id, score))
    combined.sort(key=lambda item: (-item[1], item[0]))
    return [
        {"rank": rank, "chunk_id": chunk_id, "score": float(score)}
        for rank, (chunk_id, score) in enumerate(combined[:TOP_K], start=1)
    ]


def retrieval_metrics(question: dict[str, Any], ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
    groups = question["evidence_groups"]
    if not groups:
        return None
    ranked_ids = [item["chunk_id"] for item in ranked]
    covered_groups = sum(any(chunk_id in ranked_ids for chunk_id in group) for group in groups)
    relevant_ids = {chunk_id for group in groups for chunk_id in group}
    first_rank = next(
        (rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevant_ids),
        None,
    )
    return {
        "recall_at_3": covered_groups / len(groups),
        "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
        "first_relevant_rank": first_rank,
        "context_sufficient_at_3": int(covered_groups == len(groups)),
        "covered_groups": covered_groups,
        "total_groups": len(groups),
    }


def build_context(ranked: list[dict[str, Any]], chunks_by_id: dict[str, Chunk]) -> str:
    blocks = []
    for item in ranked:
        chunk = chunks_by_id[item["chunk_id"]]
        blocks.append(f"[rank={item['rank']} chunk_id={chunk.chunk_id}]\n{chunk.text}")
    return "\n\n".join(blocks)


def get_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is required for --generate.")
    return OpenAI(api_key=api_key, base_url="https://api.minimax.io/v1", timeout=90.0)


def call_model(client: OpenAI, system: str, user: str, max_tokens: int) -> str:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as error:  # Network/API errors are retried with bounded backoff.
            last_error = error
            if attempt == 3:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Model call failed after retries: {last_error}") from last_error


def strip_think_blocks(text: str) -> str:
    """Remove MiniMax reasoning blocks so only the user-facing answer remains."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_answer(client: OpenAI, question: str, context: str) -> tuple[str, float]:
    system = (
        "你是嚴格的文件問答助手。只能根據使用者提供的檢索內容回答，使用繁體中文。"
        "若檢索內容不足以回答，必須明確說明文件未提供或無法確認，不得使用外部知識猜測。"
        "回答應直接、完整且不加入無來源的建議。"
    )
    user = f"問題：{question}\n\n檢索內容：\n{context}\n\n請回答問題。"
    started = time.perf_counter()
    answer = strip_think_blocks(call_model(client, system, user, max_tokens=2_048))
    if not answer:
        raise ValueError("The model returned reasoning but no user-facing answer.")
    return answer, (time.perf_counter() - started) * 1000


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in ("bm25", "vector", "hybrid"):
        retrieval_rows = [
            item["retrievals"][method]["metrics"]
            for item in results
            if item["answerable"]
        ]
        method_summary = {
            "recall_at_3": float(np.mean([row["recall_at_3"] for row in retrieval_rows])),
            "mrr": float(np.mean([row["reciprocal_rank"] for row in retrieval_rows])),
            "context_sufficiency_at_3": float(
                np.mean([row["context_sufficient_at_3"] for row in retrieval_rows])
            ),
            "retrieval_latency_ms_mean": float(
                np.mean([item["retrievals"][method]["latency_ms"] for item in results])
            ),
        }
        summary[method] = method_summary
    return summary


def render_report(run: dict[str, Any]) -> str:
    lines = [
        "# Small RAG retrieval benchmark",
        "",
        f"- Run time: `{run['run_at']}`",
        f"- Corpus chunks: `{run['corpus']['chunk_count']}`",
        f"- Corpus SHA-256: `{run['corpus']['fingerprint']}`",
        f"- Embedding model: `{run['config']['embedding_model']}`",
        f"- Answer model: `{run['config']['generation_model']}`",
        f"- Answer retrieval: `{run['config']['answer_method']}` Top {run['config']['top_k']}",
        f"- Answer repetitions per question: `{run['config']['answer_runs']}`",
        "- Judge: `Codex manual source-grounded review`",
        f"- Top K: `{run['config']['top_k']}`",
        f"- Hybrid weights: BM25 `{HYBRID_ALPHA}`, Vector `{1 - HYBRID_ALPHA}`",
        "",
        "## Aggregate results",
        "",
        "| Method | Recall@3 | MRR | Context sufficient@3 |",
        "|---|---:|---:|---:|",
    ]
    for method, values in run["summary"].items():
        lines.append(
            "| {method} | {recall:.3f} | {mrr:.3f} | {context:.3f} |".format(
                method=method.upper(),
                recall=values["recall_at_3"],
                mrr=values["mrr"],
                context=values["context_sufficiency_at_3"],
            )
        )
    lines.extend(
        [
            "",
            "Retrieval averages use Q01–Q13. Q14–Q15 are answer-only controls.",
            "",
            "## Per-question results",
            "",
            "| ID | Method | Top 3 chunk IDs | Recall@3 | RR |",
            "|---|---|---|---:|---:|",
        ]
    )
    for item in run["results"]:
        for method in ("bm25", "vector", "hybrid"):
            method_result = item["retrievals"][method]
            metrics = method_result["metrics"]
            lines.append(
                "| {id} | {method} | {chunks} | {recall} | {rr} |".format(
                    id=item["id"],
                    method=method.upper(),
                    chunks=", ".join(row["chunk_id"] for row in method_result["ranked"]),
                    recall="—" if metrics is None else f"{metrics['recall_at_3']:.2f}",
                    rr="—" if metrics is None else f"{metrics['reciprocal_rank']:.2f}",
                )
            )
    lines.extend(["", "## MiniMax-M2.7 Hybrid answer repetitions", ""])
    for item in run["results"]:
        lines.extend([f"### {item['id']} — {item['query']}", ""])
        answer_evaluation = item.get("answer_evaluation")
        if not answer_evaluation:
            lines.extend(["_Answer generation was not requested._", ""])
            continue
        lines.append(
            f"Hybrid Top 3: {', '.join(row['chunk_id'] for row in item['retrievals']['hybrid']['ranked'])}"
        )
        lines.append("")
        for answer_run in answer_evaluation["runs"]:
            lines.append(f"**Run {answer_run['run']}**")
            lines.append("")
            lines.append(answer_run["answer"])
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    questions = load_questions(args.questions)
    db, chunks = load_vectorstore(args.db_dir)
    validate_gold(questions, chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    tokenized_chunks = [jieba.lcut(chunk.text) for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    client = get_client() if args.generate else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "benchmark_checkpoint.json"

    run_mode = {
        "generate": args.generate,
        "top_k": TOP_K,
        "answer_method": ANSWER_METHOD,
        "answer_runs": ANSWER_RUNS,
    }
    results: list[dict[str, Any]] = []
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("corpus_fingerprint") != corpus_fingerprint(chunks):
            raise ValueError("Checkpoint corpus does not match the current Chroma collection.")
        if checkpoint.get("run_mode") != run_mode:
            raise ValueError("Checkpoint mode does not match the current generation and top-k options.")
        results = checkpoint["results"]
    completed_ids = {item["id"] for item in results}

    for question_number, question in enumerate(questions, start=1):
        if question["id"] in completed_ids:
            print(f"[{question_number}/{len(questions)}] {question['id']} (checkpoint)", flush=True)
            continue
        print(f"[{question_number}/{len(questions)}] {question['id']}", flush=True)
        bm25_started = time.perf_counter()
        raw_bm25_scores = bm25_index.get_scores(jieba.lcut(question["query"]))
        bm25_ranked, _ = rank_bm25(question["query"], chunks, bm25_index)
        bm25_latency = (time.perf_counter() - bm25_started) * 1000
        distances, vector_latency = vector_distances(db, question["query"], len(chunks))
        vector_ranked = rank_vector(distances)
        hybrid_started = time.perf_counter()
        hybrid_ranked = rank_hybrid(chunks, raw_bm25_scores, distances)
        hybrid_latency = (time.perf_counter() - hybrid_started) * 1000

        item = {
            "id": question["id"],
            "category": question["category"],
            "query": question["query"],
            "answerable": question["answerable"],
            "evidence_groups": question["evidence_groups"],
            "gold_facts": question["gold_facts"],
            "retrievals": {},
        }
        method_rows = {
            "bm25": (bm25_ranked, bm25_latency),
            "vector": (vector_ranked, vector_latency),
            "hybrid": (hybrid_ranked, hybrid_latency),
        }
        for method, (ranked, latency_ms) in method_rows.items():
            method_result = {
                "ranked": ranked,
                "latency_ms": latency_ms,
                "metrics": retrieval_metrics(question, ranked),
            }
            item["retrievals"][method] = method_result

        if args.generate:
            context = build_context(item["retrievals"][ANSWER_METHOD]["ranked"], chunks_by_id)

            def generate_repetition(run_number: int) -> dict[str, Any]:
                answer, generation_latency = generate_answer(client, question["query"], context)
                return {
                    "run": run_number,
                    "answer": answer,
                    "generation_latency_ms": generation_latency,
                }

            with ThreadPoolExecutor(max_workers=ANSWER_RUNS) as executor:
                answer_runs = list(executor.map(generate_repetition, range(1, ANSWER_RUNS + 1)))
            item["answer_evaluation"] = {
                "retrieval_method": ANSWER_METHOD,
                "top_k": TOP_K,
                "runs": answer_runs,
            }
        results.append(item)
        checkpoint = {
            "completed_questions": len(results),
            "corpus_fingerprint": corpus_fingerprint(chunks),
            "run_mode": run_mode,
            "results": results,
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    run = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": TOP_K,
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "answer_method": ANSWER_METHOD,
            "answer_runs": ANSWER_RUNS,
            "hybrid_alpha_bm25": HYBRID_ALPHA,
            "hybrid_alpha_vector": 1 - HYBRID_ALPHA,
            "generation_temperature": 0,
        },
        "corpus": {
            "chunk_count": len(chunks),
            "fingerprint": corpus_fingerprint(chunks),
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
        },
        "results": results,
    }
    run["summary"] = aggregate(results)

    json_path = args.output_dir / "benchmark_results.json"
    report_path = args.output_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(run), encoding="utf-8")
    print(json.dumps(run["summary"], ensure_ascii=False, indent=2))
    print(f"Results: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
