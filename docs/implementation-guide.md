# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| Embeddings | Azure OpenAI embedding deployment | Cache by content hash |
| Retrieval | Azure AI Search: vector, keyword and hybrid with RRF | Same ablation, now on real embeddings |
| Rerank | Azure AI Search semantic ranker | Compare against no rerank in the ablation |
| Evaluation | Azure AI Foundry evaluations plus this ablation in CI |  |

## Steps

1. Build the gold set from real user questions with the documents that answer them (100+).
2. Run the ablation on Azure AI Search with real embeddings; keep only changes whose interval excludes zero.
3. Track tokens per answer next to recall, and set a budget.

## Guardrails to keep when going live

- One change at a time from a fixed baseline, so each row isolates a technique.
- Every delta has a 95% interval; rows whose interval includes zero are marked.
- Token cost is reported next to quality, so a gain that doubles context is visible.

## What to measure before calling it production-ready

- Recall@k and precision@k on the gold set, with intervals.
- Tokens per answer and latency per stage.
- Answer faithfulness once generation is added.
