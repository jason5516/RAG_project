from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results" / "benchmark_results.json"
REVIEW_PATH = HERE / "manual_review.json"
AUDITED_JSON_PATH = HERE / "results" / "manual_audit_results.json"
AUDITED_REPORT_PATH = HERE / "results" / "manual_audit_report.md"
METHODS = ("bm25", "vector", "hybrid")


def main() -> None:
    run = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    audited = deepcopy(run)

    overrides = {
        (item["question_id"], item["method"]): item
        for item in review["overrides"]
    }
    known_pairs = {
        (item["id"], method)
        for item in audited["results"]
        for method in METHODS
    }
    unknown = set(overrides) - known_pairs
    if unknown:
        raise ValueError(f"Manual review has unknown question/method pairs: {sorted(unknown)}")

    for item in audited["results"]:
        for method in METHODS:
            automatic = item["retrievals"][method]["judgement"]
            manual = {
                "helpfulness": automatic["helpfulness"],
                "faithfulness": automatic["faithfulness"],
                "reason": "Accepted automatic judgement after manual source review.",
                "overridden": False,
            }
            override = overrides.get((item["id"], method))
            if override:
                manual.update(
                    {
                        "helpfulness": override["helpfulness"],
                        "faithfulness": override["faithfulness"],
                        "reason": override["reason"],
                        "overridden": True,
                    }
                )
            manual["strict_pass"] = int(
                manual["helpfulness"] == 2 and manual["faithfulness"] == 1
            )
            item["retrievals"][method]["manual_judgement"] = manual

    audited_summary = {}
    for method in METHODS:
        retrieval_rows = [
            item["retrievals"][method]["metrics"]
            for item in audited["results"]
            if item["answerable"]
        ]
        answer_rows = [
            item["retrievals"][method]["manual_judgement"]
            for item in audited["results"]
        ]
        audited_summary[method] = {
            "recall_at_5": float(np.mean([row["recall_at_5"] for row in retrieval_rows])),
            "mrr": float(np.mean([row["reciprocal_rank"] for row in retrieval_rows])),
            "context_sufficiency_at_5": float(
                np.mean([row["context_sufficient_at_5"] for row in retrieval_rows])
            ),
            "helpfulness_mean": float(np.mean([row["helpfulness"] for row in answer_rows])),
            "faithfulness_pass_rate": float(np.mean([row["faithfulness"] for row in answer_rows])),
            "strict_pass_rate": float(np.mean([row["strict_pass"] for row in answer_rows])),
            "strict_pass_count": int(sum(row["strict_pass"] for row in answer_rows)),
            "answer_count": len(answer_rows),
        }

    audited["automatic_judge_summary"] = audited.pop("summary")
    audited["summary"] = audited_summary
    audited["manual_review"] = {
        "reviewer": review["reviewer"],
        "reviewed_at": review["reviewed_at"],
        "override_count": len(overrides),
        "policy": review["policy"],
    }
    AUDITED_JSON_PATH.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Manually audited RAG benchmark",
        "",
        f"- Retrieval questions: 13 answerable questions (Q01–Q13)",
        f"- Answer questions: 15 questions, including two unanswerable controls",
        f"- Corpus: {audited['corpus']['chunk_count']} chunks",
        f"- Manual overrides after source audit: {len(overrides)} of 45 answers",
        "",
        "## Final aggregate",
        "",
        "| Method | Recall@5 | MRR | Context sufficient@5 | Helpfulness / 2 | Faithfulness | Strict pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in audited_summary.items():
        lines.append(
            f"| {method.upper()} | {values['recall_at_5']:.3f} | {values['mrr']:.3f} | "
            f"{values['context_sufficiency_at_5']:.3f} | {values['helpfulness_mean']:.3f} | "
            f"{values['faithfulness_pass_rate']:.3f} | "
            f"{values['strict_pass_count']}/{values['answer_count']} "
            f"({values['strict_pass_rate']:.3f}) |"
        )

    lines.extend(
        [
            "",
            "## Manually adjudicated differences",
            "",
            "| Question | Method | Helpful | Faithful | Reason |",
            "|---|---|---:|---:|---|",
        ]
    )
    for override in review["overrides"]:
        reason = override["reason"].replace("|", "\\|")
        lines.append(
            f"| {override['question_id']} | {override['method'].upper()} | "
            f"{override['helpfulness']} | {override['faithfulness']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "The original model-judge scores remain unchanged in `benchmark_results.json`. "
            "The audited JSON stores both automatic and manual judgements for traceability.",
            "",
        ]
    )
    AUDITED_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audited_summary, ensure_ascii=False, indent=2))
    print(f"Audited results: {AUDITED_JSON_PATH}")
    print(f"Audited report: {AUDITED_REPORT_PATH}")


if __name__ == "__main__":
    main()
