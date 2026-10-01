from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results_hybrid_x3" / "benchmark_results.json"
JUDGEMENTS_PATH = HERE / "codex_judgements_hybrid_x3.json"
AUDITED_JSON_PATH = HERE / "results_hybrid_x3" / "benchmark_results_codex_judged.json"
REPORT_PATH = HERE / "results_hybrid_x3" / "final_report_hybrid_x3.md"
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
        (row["question_id"], row["run"]): row
        for row in review["judgements"]
    }
    expected = {
        (item["id"], answer_run["run"])
        for item in audited["results"]
        for answer_run in item["answer_evaluation"]["runs"]
    }
    if set(judgement_map) != expected:
        missing = sorted(expected - set(judgement_map))
        extra = sorted(set(judgement_map) - expected)
        raise ValueError(f"Judgement coverage mismatch; missing={missing}, extra={extra}")

    answer_rows = []
    for item in audited["results"]:
        for answer_run in item["answer_evaluation"]["runs"]:
            judgement = dict(judgement_map[(item["id"], answer_run["run"])])
            judgement["strict_pass"] = int(
                judgement["helpfulness"] == 2 and judgement["faithfulness"] == 1
            )
            judgement["judge"] = review["judge"]
            answer_run["codex_judgement"] = judgement
            answer_rows.append(judgement)

    retrieval_summary = {}
    for method in METHODS:
        metrics = [
            item["retrievals"][method]["metrics"]
            for item in audited["results"]
            if item["answerable"]
        ]
        retrieval_summary[method] = {
            "recall_at_3": float(np.mean([row["recall_at_3"] for row in metrics])),
            "mrr": float(np.mean([row["reciprocal_rank"] for row in metrics])),
            "context_sufficiency_at_3": float(
                np.mean([row["context_sufficient_at_3"] for row in metrics])
            ),
        }

    by_run = {}
    for run_number in range(1, review["answer_runs"] + 1):
        rows = [row for row in answer_rows if row["run"] == run_number]
        by_run[str(run_number)] = {
            "helpfulness_mean": float(np.mean([row["helpfulness"] for row in rows])),
            "faithfulness_pass_rate": float(np.mean([row["faithfulness"] for row in rows])),
            "strict_pass_count": int(sum(row["strict_pass"] for row in rows)),
            "strict_pass_rate": float(np.mean([row["strict_pass"] for row in rows])),
            "answer_count": len(rows),
        }

    all_three_pass = 0
    for item in audited["results"]:
        if all(run["codex_judgement"]["strict_pass"] for run in item["answer_evaluation"]["runs"]):
            all_three_pass += 1

    answer_summary = {
        "retrieval_method": review["retrieval_method"],
        "answer_runs_per_question": review["answer_runs"],
        "answer_count": len(answer_rows),
        "helpfulness_mean": float(np.mean([row["helpfulness"] for row in answer_rows])),
        "faithfulness_pass_rate": float(np.mean([row["faithfulness"] for row in answer_rows])),
        "strict_pass_count": int(sum(row["strict_pass"] for row in answer_rows)),
        "strict_pass_rate": float(np.mean([row["strict_pass"] for row in answer_rows])),
        "all_three_runs_pass_question_count": all_three_pass,
        "question_count": len(audited["results"]),
        "all_three_runs_pass_question_rate": all_three_pass / len(audited["results"]),
        "by_run": by_run,
    }

    audited["retrieval_summary"] = retrieval_summary
    audited["answer_summary"] = answer_summary
    audited["evaluation"] = {
        "answer_model": review["answer_model"],
        "answer_retrieval": f"{review['retrieval_method']}@{review['top_k']}",
        "answer_runs": review["answer_runs"],
        "judge": review["judge"],
        "rubric": review["rubric"],
    }
    AUDITED_JSON_PATH.write_text(
        json.dumps(audited, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# RAG benchmark：三種 retrieval + Hybrid 三次回答",
        "",
        "- Retrieval：BM25、Vector、Hybrid 比較 Recall@3、MRR、Context Sufficiency@3。",
        f"- Answer：固定 `{review['retrieval_method'].upper()} Top {review['top_k']}`，"
        f"每題由 `{review['answer_model']}` 回答 {review['answer_runs']} 次。",
        f"- Judge：`{review['judge']}`，逐次核對相同的 Hybrid Top 3 context 與 gold facts。",
        "- Strict Pass：Helpfulness = 2 且 Faithfulness = 1。",
        "",
        "## 表一：三種檢索方式的 Retrieval 指標",
        "",
        "| 檢索方式 | Recall@3 | MRR | Context Sufficiency@3 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = retrieval_summary[method]
        lines.append(
            f"| {method.upper()} | {row['recall_at_3']:.3f} | {row['mrr']:.3f} | "
            f"{row['context_sufficiency_at_3']:.3f} |"
        )

    run_parts = [
        f"Run {number}: {values['strict_pass_count']}/{values['answer_count']}"
        for number, values in by_run.items()
    ]
    lines.extend(
        [
            "",
            "## Hybrid 三次回答整體結果",
            "",
            f"- Helpfulness 平均：`{answer_summary['helpfulness_mean']:.3f} / 2`",
            f"- Faithfulness 通過率：`{answer_summary['faithfulness_pass_rate']:.3f}`",
            f"- Strict Pass：`{answer_summary['strict_pass_count']}/{answer_summary['answer_count']}` "
            f"(`{answer_summary['strict_pass_rate']:.3f}`)",
            f"- 每題三次全部 Strict Pass：`{all_three_pass}/{len(audited['results'])}` "
            f"(`{answer_summary['all_three_runs_pass_question_rate']:.3f}`)",
            f"- 分次 Strict Pass：`{'；'.join(run_parts)}`",
            "",
            "## 表二：每題三次 Hybrid 回答、gold facts 與 Codex judge",
            "",
            "| 題號 | 問題 | Run | Hybrid Top 3 chunks | MiniMax-M2.7 回答 | Gold facts | Codex judge |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for item in audited["results"]:
        gold = "<br>".join(f"• {markdown_cell(fact)}" for fact in item["gold_facts"])
        chunks = "<br>".join(row["chunk_id"] for row in item["retrievals"]["hybrid"]["ranked"])
        for answer_run in item["answer_evaluation"]["runs"]:
            judge = answer_run["codex_judgement"]
            verdict = "PASS" if judge["strict_pass"] else "FAIL"
            judge_text = (
                f"Helpful={judge['helpfulness']}/2；Faithful={judge['faithfulness']}；"
                f"Strict={verdict}<br>{markdown_cell(judge['reason'])}"
            )
            lines.append(
                f"| {item['id']} | {markdown_cell(item['query'])} | {answer_run['run']} | "
                f"{markdown_cell(chunks)} | {markdown_cell(answer_run['answer'])} | {gold} | {judge_text} |"
            )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"retrieval": retrieval_summary, "answers": answer_summary}, ensure_ascii=False, indent=2))
    print(f"Codex-judged results: {AUDITED_JSON_PATH}")
    print(f"Final report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
