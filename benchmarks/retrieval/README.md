# Small retrieval benchmark

This benchmark compares BM25, vector, and the project's current hybrid scoring on the same frozen Chroma collection.

## Evaluation split

- MiniMax-M2.7 generates three independent answers from the Hybrid top-three chunks for every question.
- Codex performs the source-grounded answer review separately; MiniMax never grades its own output.
- Retrieval metrics use answerable questions Q01–Q13.
- Recall@3 measures the fraction of required evidence groups covered by the top three chunks.
- MRR uses the rank of the first chunk belonging to any gold evidence group.
- Context sufficiency is one only when all required evidence groups appear in the top three.
- Answer metrics use all three Hybrid-grounded runs for Q01–Q15. Q14–Q15 are unanswerable controls and should produce an explicit abstention.

Overlapping chunks that contain the same evidence are placed in one evidence group; retrieving any member covers that group. Q10 has two independent evidence groups because a complete answer requires two chunks.

## Run retrieval and MiniMax answers

Use the project's `airag` environment:

```bash
/home/jason/miniconda3/envs/airag/bin/python benchmarks/retrieval/run_benchmark.py --generate
```

If an API run is interrupted after a completed question, resume it with the same options plus `--resume`.

Raw retrieval results and the three Hybrid-grounded MiniMax answers are written under `benchmarks/retrieval/results_hybrid_x3/`. Codex judgements and the final report are added only after generation completes.
