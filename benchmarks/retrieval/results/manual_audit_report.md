# Manually audited RAG benchmark

- Retrieval questions: 13 answerable questions (Q01–Q13)
- Answer questions: 15 questions, including two unanswerable controls
- Corpus: 14 chunks
- Manual overrides after source audit: 12 of 45 answers

## Final aggregate

| Method | Recall@5 | MRR | Context sufficient@5 | Helpfulness / 2 | Faithfulness | Strict pass |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 1.000 | 0.885 | 1.000 | 1.800 | 0.800 | 12/15 (0.800) |
| VECTOR | 1.000 | 0.840 | 1.000 | 1.733 | 0.667 | 10/15 (0.667) |
| HYBRID | 1.000 | 0.885 | 1.000 | 1.867 | 0.733 | 11/15 (0.733) |

## Manually adjudicated differences

| Question | Method | Helpful | Faithful | Reason |
|---|---|---:|---:|---|
| Q04 | VECTOR | 2 | 0 | The substantive answer is correct, but it explicitly attributes the facts to chunk 0004; the application steps are in chunk 0003. |
| Q07 | HYBRID | 2 | 0 | The required procedure is correct, but the answer adds unsupported implementation details: signing on paper, PDF or photographed files, and sending the files specifically to the student. |
| Q08 | VECTOR | 2 | 1 | Automatic-judge false negative: the answer does include the red hardcover library copy for doctoral students and all other required copies. |
| Q09 | VECTOR | 2 | 0 | The eligibility and interest result are correct, but the retrieved chunks only say interest-free; they do not state that the government pays the interest. |
| Q09 | HYBRID | 2 | 0 | The eligibility and interest result are correct, but the retrieved chunks only say interest-free; they do not state that the government pays the interest. |
| Q12 | BM25 | 1 | 0 | The answer implies ordinary students may also borrow living expenses and merely lack a special document requirement, contradicting the corpus restriction to low- and middle-low-income households. |
| Q12 | HYBRID | 2 | 0 | The core eligibility is correct, but the answer adds unsupported details such as per-semester limits, bank-determined amounts, example proof documents, and paying the housing deposit in cash or by another method. |
| Q13 | BM25 | 2 | 0 | The comparison is useful, but its table attaches the new-student student-card exemption to the previous-borrower column. |
| Q13 | VECTOR | 1 | 0 | The answer incorrectly says a first-time applicant cannot use online signing and must personally visit a branch; the retrieved context allows branch or online signing and only says the guarantor must attend under the new-applicant rules. |
| Q14 | BM25 | 0 | 0 | The answer acknowledges insufficient evidence but speculates that a laptop may fall under the book allowance, violating the unanswerable-item rule. |
| Q14 | VECTOR | 0 | 0 | The answer ultimately abstains but first speculates that laptops are usually categorized as book expenses, which is absent from the retrieved chunks. |
| Q14 | HYBRID | 0 | 0 | The answer infers from omission that laptops cannot be financed and then suggests other funding routes; neither claim is established by the retrieved chunks. |

The original model-judge scores remain unchanged in `benchmark_results.json`. The audited JSON stores both automatic and manual judgements for traceability.
