from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results_top3" / "benchmark_results.json"
JUDGEMENTS_PATH = HERE / "codex_judgements_top3.json"
AUDITED_JSON_PATH = HERE / "results_top3" / "benchmark_results_codex_judged.json"
REPORT_PATH = HERE / "results_top3" / "final_report_two_tables.md"
METHODS = ("bm25", "vector", "hybrid")


def markdown_cell(value: str) -> str:
    value = value.replace("|", "\\|")
    value = re.sub(r"\n{2,}", "\n", value.strip())
    return value.replace("\n", "<br>")


def main() -> None:
    run = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    review = json.loads(JUDGEMENTS_PATH.read_text(encoding="utf-8"))
    audited = deepcopy(run)

    judgement_map = {
        (row["question_id"], row["method"]): row
        for row in review["judgements"]
    }
    expected = {
        (item["id"], method)
        for item in audited["results"]
        for method in METHODS
    }
    if set(judgement_map) != expected:
        missing = sorted(expected - set(judgement_map))
        extra = sorted(set(judgement_map) - expected)
        raise ValueError(f"Judgement coverage mismatch; missing={missing}, extra={extra}")

    for item in audited["results"]:
        for method in METHODS:
            judgement = dict(judgement_map[(item["id"], method)])
            judgement["strict_pass"] = int(
                judgement["helpfulness"] == 2 and judgement["faithfulness"] == 1
            )
            judgement["judge"] = review["judge"]
            item["retrievals"][method]["codex_judgement"] = judgement

    summary = {}
    for method in METHODS:
        retrieval_rows = [
            item["retrievals"][method]["metrics"]
            for item in audited["results"]
            if item["answerable"]
        ]
        answer_rows = [
            item["retrievals"][method]["codex_judgement"]
            for item in audited["results"]
        ]
        summary[method] = {
            "recall_at_3": float(np.mean([row["recall_at_3"] for row in retrieval_rows])),
            "mrr": float(np.mean([row["reciprocal_rank"] for row in retrieval_rows])),
            "context_sufficiency_at_3": float(
                np.mean([row["context_sufficient_at_3"] for row in retrieval_rows])
            ),
            "helpfulness_mean": float(np.mean([row["helpfulness"] for row in answer_rows])),
            "faithfulness_pass_rate": float(np.mean([row["faithfulness"] for row in answer_rows])),
            "strict_pass_rate": float(np.mean([row["strict_pass"] for row in answer_rows])),
            "strict_pass_count": int(sum(row["strict_pass"] for row in answer_rows)),
            "answer_count": len(answer_rows),
        }

    audited["summary"] = summary
    audited["evaluation"] = {
        "answer_model": review["answer_model"],
        "judge": review["judge"],
        "rubric": review["rubric"],
        "judgement_count": len(judgement_map),
    }
    AUDITED_JSON_PATH.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# RAG benchmark：Recall@3 與 Codex judge",
        "",
        f"- 回答模型：`{review['answer_model']}`",
        f"- Judge：`{review['judge']}`（逐題核對實際 Top 3 context 與 gold facts）",
        "- Retrieval 指標：Q01–Q13；回答指標：Q01–Q15",
        "- Strict Pass：Helpfulness = 2 且 Faithfulness = 1",
        "",
        "## 表一：三種檢索方式的整體指標",
        "",
        "| 檢索方式 | Recall@3 | MRR | Context Sufficiency@3 | Helpfulness 平均 / 2 | Faithfulness 通過率 | Strict Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary[method]
        lines.append(
            f"| {method.upper()} | {row['recall_at_3']:.3f} | {row['mrr']:.3f} | "
            f"{row['context_sufficiency_at_3']:.3f} | {row['helpfulness_mean']:.3f} | "
            f"{row['faithfulness_pass_rate']:.3f} | {row['strict_pass_count']}/{row['answer_count']} "
            f"({row['strict_pass_rate']:.3f}) |"
        )

    lines.extend(
        [
            "",
            "## 表二：逐題 MiniMax 回答、gold facts 與 Codex judge",
            "",
            "| 題號 | 問題 | 檢索方式 | Top 3 chunks | MiniMax-M2.7 回答 | Gold facts | Codex judge |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in audited["results"]:
        gold = "<br>".join(f"• {markdown_cell(fact)}" for fact in item["gold_facts"])
        for method in METHODS:
            result = item["retrievals"][method]
            judge = result["codex_judgement"]
            chunks = "<br>".join(row["chunk_id"] for row in result["ranked"])
            verdict = "PASS" if judge["strict_pass"] else "FAIL"
            judge_text = (
                f"Helpful={judge['helpfulness']}/2；Faithful={judge['faithfulness']}；"
                f"Strict={verdict}<br>{markdown_cell(judge['reason'])}"
            )
            lines.append(
                f"| {item['id']} | {markdown_cell(item['query'])} | {method.upper()} | "
                f"{markdown_cell(chunks)} | {markdown_cell(result['answer'])} | {gold} | {judge_text} |"
            )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Codex-judged results: {AUDITED_JSON_PATH}")
    print(f"Two-table report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
