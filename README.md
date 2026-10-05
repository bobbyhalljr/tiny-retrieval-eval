# tiny-retrieval-eval

A tiny retrieval eval for RAG, in one TypeScript file. Run it before you switch embedding models.
It compares two embedding models on labeled queries (answer recall, evidence recall, MRR), applies a ship gate, and tests whether a query model can search an index built by another model.
The "embedding models" are mocked with hashed bag-of-words vectors. The corpus, queries, labels and tolerance are example inputs. No model, no API key.

## Why it matters

Three embedding launches landed in one week:

- Sep 24, 2026: TopK [introduced topk-embed-v1](https://www.topk.io/blog/topk-embed-v1), multi-vector embedding models for text and visual documents.
- Sep 30, 2026: Cohere [released Embed 5](https://cohere.com/blog/embed-5) in Pro and Fast tiers that share one embedding space, so you can index with Pro and query with either.
- Sep 30, 2026: Perplexity [introduced pplx-embed-v2-context-9b-preview](https://www.perplexity.ai/hub/blog/contextual-embedding-beyond-the-gold-passage), trained to retrieve answer chunks and supporting context, not a single gold chunk. Its [model card](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b-preview) warns that preview embeddings should not be mixed with embeddings from a future release.

Every launch has a chart. None of them were drawn on your documents.

## Run it

You need Node.js 18 or newer.

```bash
npm install
npx tsx eval.ts
```

## Example output

This is real output from `npx tsx eval.ts`:

```text
12 chunks, 4 labeled queries (example data), k=2

== switch test ==
chunk-only  answer@2 0.75  evidence@2 0.50  MRR 0.58
contextual  answer@2 1.00  evidence@2 0.75  MRR 1.00
  rent at 48 Elm     answer rank 2 -> 1
  rent at 12 Oak     answer rank 3 -> 1
  late fee at Elm    answer rank 2 -> 1
  payments rollback  answer rank 1 -> 1
gate: SHIP (drops: none; worse queries: none)

== mixed-index test (index built with contextual) ==
query with contextual  answer@2 1.00  evidence@2 0.75  MRR 1.00
query with ctx-fast    answer@2 1.00  evidence@2 0.75  MRR 1.00
query with ctx-v2      answer@2 0.00  evidence@2 0.00  MRR 0.21
```

- `chunk-only` ranked the Elm Avenue rent above the Oak Street rent for an Oak Street question. The contextual model put the answer first on every query. The gate says SHIP.
- `ctx-fast` shares the contextual model's space and scores the same against its index.
- `ctx-v2` has the same recipe and the same dimensions but a new seed. Against the old index, recall drops to zero. Re-embed the index when the space changes.

## How it works

```text
New embedding model
  ↓
Labeled queries ──→ answer@k, evidence@k, MRR
  ↓
Gate ──→ no metric drops, no query gets worse
  ↓
Mixed-index test ──→ can the old vectors stay?
  ↓
Ship, or re-embed first
```

| File | What it does |
| --- | --- |
| `eval.ts` | Labeled corpus, mock embedders, index and rank, metrics, ship gate, mixed-index test |
| `output.txt` | Captured stdout from one run |

To use a real model, replace `embed()` with your provider's embedding call and keep the rest.

## Write-up

Full tutorial: SUBSTACK_URL

Also: DEV_URL

## License

MIT
