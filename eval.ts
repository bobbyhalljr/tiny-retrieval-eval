// eval.ts: a tiny retrieval eval to run before you switch embedding models.
// Everything is mocked: the "embedding models" are hashed bag-of-words vectors, not real models.
// The corpus, queries, labels and the ship tolerance are example inputs. No API key, no network.

// Step 1: a tiny labeled corpus
type Chunk = { id: string; title: string; text: string };
type Query = { id: string; text: string; answer: string; evidence: string[] };

const DOCS: Record<string, [string, string[]]> = {
  oak: ["Lease for 12 Oak Street", [
    "This lease covers 12 Oak Street, unit 3.",
    "Monthly rent is 2,100 dollars.",
    "Late fee: 50 dollars after five days.",
    "Dogs and cats are allowed with a pet deposit.",
  ]],
  elm: ["Lease for 48 Elm Avenue", [
    "This lease covers 48 Elm Avenue, unit 9.",
    "Monthly rent is 2,650 dollars.",
    "Late fee: 75 dollars after three days.",
    "No pets are allowed.",
  ]],
  runbook: ["Payments deploy runbook", [
    "Every deploy ships a tagged release.",
    "If errors spike after a deploy, page on-call.",
    "To roll back, redeploy the previous release tag.",
    "Watch errors for ten minutes to confirm the rollback.",
  ]],
};

const CHUNKS: Chunk[] = Object.entries(DOCS).flatMap(([doc, [title, texts]]) =>
  texts.map((text, i) => ({ id: `${doc}#${i}`, title, text })),
);

// answer = the chunk that holds the answer. evidence = every chunk you need to trust it.
const QUERIES: Query[] = [
  { id: "rent at 48 Elm", text: "Monthly rent at 48 Elm Avenue?", answer: "elm#1", evidence: ["elm#1", "elm#0"] },
  { id: "rent at 12 Oak", text: "Monthly rent at 12 Oak Street?", answer: "oak#1", evidence: ["oak#1", "oak#0"] },
  { id: "late fee at Elm", text: "Late fee on the Elm Avenue lease?", answer: "elm#2", evidence: ["elm#2", "elm#0"] },
  { id: "payments rollback", text: "How do I roll back a payments deploy?", answer: "runbook#2", evidence: ["runbook#2", "runbook#3"] },
];

// Step 2: mock embedding models (signed feature hashing: same seed, same space)
type Model = { name: string; seed: number; dims: number; useTitle: boolean; dropStopwords: boolean };

const STOP = new Set("a an the is are of on at to for and do i how".split(" "));

function tokens(text: string, dropStopwords: boolean): string[] {
  return text.toLowerCase().split(/[^a-z0-9]+/)
    .filter((t) => t && !(dropStopwords && STOP.has(t)))
    .map((t) => (t.length > 3 && t.endsWith("s") ? t.slice(0, -1) : t));
}

function embed(m: Model, text: string): number[] {
  const v = new Array<number>(m.dims).fill(0);
  for (const t of tokens(text, m.dropStopwords)) {
    let h = 2166136261 ^ m.seed; // FNV-1a
    for (const ch of t) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
    h >>>= 0;
    v[h % m.dims] += h & 1 ? 1 : -1;
  }
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

const chunkOnly: Model = { name: "chunk-only", seed: 1, dims: 256, useTitle: false, dropStopwords: true };
const contextual: Model = { name: "contextual", seed: 7, dims: 256, useTitle: true, dropStopwords: true };
const ctxFast: Model = { ...contextual, name: "ctx-fast", dropStopwords: false }; // same seed, cheaper
const ctxV2: Model = { ...contextual, name: "ctx-v2", seed: 99 }; // next release, new seed

// Step 3: index and rank
type Index = { model: Model; vectors: Map<string, number[]> };

function buildIndex(m: Model): Index {
  const vectors = new Map(CHUNKS.map((c) => [c.id, embed(m, m.useTitle ? `${c.title}. ${c.text}` : c.text)]));
  return { model: m, vectors };
}

function rank(index: Index, queryModel: Model, text: string): string[] {
  if (queryModel.dims !== index.model.dims) throw new Error("dims mismatch: re-embed the index");
  const q = embed(queryModel, text);
  return [...index.vectors.entries()]
    .map(([id, v]) => ({ id, score: v.reduce((s, x, i) => s + x * q[i], 0) }))
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .map((r) => r.id);
}

// Step 4: the metrics
const K = 2;
type Row = { id: string; answerRank: number; hits: number; total: number };
type Score = { answer: number; evidence: number; mrr: number };

function runEval(index: Index, queryModel: Model): { rows: Row[]; score: Score } {
  const rows = QUERIES.map((q) => {
    const ranked = rank(index, queryModel, q.text);
    const hits = q.evidence.filter((e) => ranked.slice(0, K).includes(e)).length;
    return { id: q.id, answerRank: ranked.indexOf(q.answer) + 1, hits, total: q.evidence.length };
  });
  const mean = (f: (r: Row) => number) => rows.reduce((s, r) => s + f(r), 0) / rows.length;
  return {
    rows,
    score: {
      answer: mean((r) => (r.answerRank <= K ? 1 : 0)), // answer chunk in top k
      evidence: mean((r) => (r.hits === r.total ? 1 : 0)), // every evidence chunk in top k
      mrr: mean((r) => 1 / r.answerRank),
    },
  };
}

// Step 5: compare, gate, and test a mixed index
const prod = runEval(buildIndex(chunkOnly), chunkOnly);
const cand = runEval(buildIndex(contextual), contextual);
const keys: (keyof Score)[] = ["answer", "evidence", "mrr"];
const fmt = (s: Score) => keys.map((k) => `${k === "mrr" ? "MRR" : `${k}@${K}`} ${s[k].toFixed(2)}`).join("  ");

console.log(`${CHUNKS.length} chunks, ${QUERIES.length} labeled queries (example data), k=${K}\n`);
console.log(`== switch test ==\nchunk-only  ${fmt(prod.score)}\ncontextual  ${fmt(cand.score)}`);

const worse = prod.rows.filter((p, i) => cand.rows[i].answerRank > p.answerRank).map((p) => p.id);
prod.rows.forEach((p, i) => console.log(`  ${p.id.padEnd(18)} answer rank ${p.answerRank} -> ${cand.rows[i].answerRank}`));

const TOLERANCE = 0.05; // example threshold
const drops = keys.filter((k) => cand.score[k] < prod.score[k] - TOLERANCE);
console.log(`gate: ${drops.length || worse.length ? "HOLD" : "SHIP"} (drops: ${drops.join(", ") || "none"}; worse queries: ${worse.join(", ") || "none"})\n`);

console.log("== mixed-index test (index built with contextual) ==");
const ctxIndex = buildIndex(contextual);
for (const m of [contextual, ctxFast, ctxV2]) console.log(`query with ${m.name.padEnd(11)} ${fmt(runEval(ctxIndex, m).score)}`);
