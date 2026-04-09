# Autoresearch Program — Attention Kernel Challenge

You are an autonomous research agent. Your job is to improve the latency of a
block-sparse causal attention kernel running on an NVIDIA H100 GPU via Modal.
You iterate by editing code, committing, running the evaluator, and keeping or
discarding changes — forever, without stopping to ask for permission.

---

## 1. Scope

**You may edit:** everything under `autoresearch/submission/` only.
`autoresearch/submission/submission.py` is the primary file. You may add helper
files inside that directory (e.g. `kernels.py`, `manifest.py`) if useful. All
imports within submission files must come exclusively from `{torch, triton,
numpy}` — no other top-level imports are allowed.

**You must never touch:**
- `attention_kernel_challenge/` (the frozen harness)
- `example_submission/` (reference copy, read-only)
- `autoresearch/` scripts, `program.md`, baselines, or `results.tsv`
- Any other file outside `autoresearch/submission/`

**The submission contract** (from `README.md` and `attention_kernel_challenge/spec.py`):
- `submission.py` must export:
  - `block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens) -> (output, lse)`
  - `VARIANT_MANIFEST` — list of variant dicts, at most 16 entries
  - `setup(suite_specs, device, variants)` — optional but recommended
- Tolerances: `output_atol=1e-3, output_rtol=1e-2`; `lse_atol=1e-5, lse_rtol=1e-5`
  (lse is 100× tighter — always compute lse in fp32)
- `setup()` has a **30-second cap**. All JIT compilation and autotuning MUST
  happen there. After setup the harness freezes the compile cache and any
  further Triton/Inductor cache mutation during timed runs is a hard failure.
- Three workload families: `sliding_window`, `sliding_window_global`,
  `sliding_window_retrieval`. Fixed: `BLOCK_SIZE=128`, `HEAD_DIM=128`.

**The dual-path requirement:** This CPU-only machine cannot run Triton kernels.
`submission.py` must always contain a pure-PyTorch fallback path that runs on
CPU. When `device == "cpu"`, dispatch to the torch path. When `device ==
"cuda"`, dispatch to the Triton/fast path. This keeps tier 0 (`run_check.sh`)
working for free.

---

## 2. The Four-Tier Loop

Run tiers in strict order. Never pay for an expensive tier on a candidate that
failed a cheaper one.

| Tier | Script | Suite | Backend | Est. cost | Gate condition |
|---|---|---|---|---|---|
| 0 | `run_check.sh` | smoke | local CPU | $0 | `valid=true` |
| 1 | `run_smoke.sh` | smoke | Modal H100 | ~$0.10 | `valid=true` |
| 2 | `run_quick.sh` | quick | Modal H100 | ~$0.30 | `valid=true` AND `quick_geo ≤ best_quick × 1.02` |
| 3 | `run_full.sh` | full | Modal H100 | ~$1–2 | `valid=true` AND `full_geo < best_full` |
| 4 | `run_broad.sh` | broad | Modal H100 | ~$5–10 | milestone only; `broad_geo < prior_broad × 1.05` |

**Loop algorithm (execute this FOREVER):**

```
1. Pick one idea from the idea bank below (or a new hypothesis you form).
   Pick the idea most likely to attack the currently worst family — read
   the per-family columns (full_sliding, full_global, full_retrieval) in
   results.tsv to identify it.

2. Edit autoresearch/submission/ (primarily submission.py). Make ONE
   focused change per commit — small diffs are easier to interpret.

3. git add autoresearch/submission/ && git commit -m "<one-line description>"

4. Run tier 0:
     OUT=$(bash autoresearch/run_check.sh)
   If valid=false → append row to results.tsv with status=cpu_fail,
   run `git reset --hard HEAD~1`, go to step 1.

5. Run tier 1:
     OUT=$(bash autoresearch/run_smoke.sh)
   If valid=false → status=smoke_invalid, reset, go to step 1.

6. Run tier 2:
     OUT=$(bash autoresearch/run_quick.sh)
   If valid=false → status=quick_invalid, reset, go to step 1.
   Extract quick_geo from OUT. If quick_geo > best_quick_geo × 1.02:
     → status=discard (skip full, saves money), reset, go to step 1.

7. Run tier 3:
     OUT=$(bash autoresearch/run_full.sh)
   If valid=false → status=full_invalid, reset, go to step 1.
   Extract full_geo from OUT.
   If full_geo < best_full_geo (strictly better):
     → status=keep. Advance best_full_geo. Append row. Continue (step 1).
   Else:
     → status=discard. Reset. Append row. Go to step 1.

8. Every ~10 kept commits, run tier 4:
     OUT=$(bash autoresearch/run_broad.sh)
   If broad_geo > prior_broad × 1.05 (>5% regression):
     → STOP. Print "OVERFIT WARNING" and wait for human. Do not reset;
       the human needs to inspect which family regressed.
   Else update the broad milestone and continue.

9. ALWAYS append a row to autoresearch/results.tsv after each experiment
   (whether kept or discarded), following the schema below. Never skip it.
```

**How to parse tier output:** Each script prints one tab-separated line:
```
tier=X  suite=X  backend=X  valid=true/false  geo=X  worst=X  sliding=X  sw_global=X  sw_retrieval=X  speedup=X  wall_s=X  reason=X
```
Use `grep -oP '(?<=\bgeo=)[^\t]+'` or Python to extract fields.

---

## 3. results.tsv Schema

Append one row per experiment. Tab-separated. Never rewrite existing rows.

**Columns (in order):**
```
commit  status  tier_reached  cpu_ok  smoke_ok  quick_geo  quick_worst  full_geo  full_worst  full_sliding  full_global  full_retrieval  speedup_full  broad_geo  cost_usd_est  description
```

- `commit`: 7-char git hash (`git rev-parse --short HEAD`)
- `status`: `keep | discard | cpu_fail | smoke_invalid | quick_invalid | full_invalid | crash`
- `tier_reached`: `0 | 1 | 2 | 3 | 4`
- `cpu_ok`: `true | false | -`
- `smoke_ok`: `true | false | -`
- `quick_geo`, `quick_worst`: float ms or `-`
- `full_geo`, `full_worst`: float ms or `-`
- `full_sliding`, `full_global`, `full_retrieval`: float ms or `-` (per-family medians, parsed from tier-3 output)
- `speedup_full`: float or `-` (ref_full_geo / submission_full_geo; > 1.0 = faster than reference)
- `broad_geo`: float ms or `-`
- `cost_usd_est`: estimated USD for this experiment (smoke $0.10, quick $0.30, full $1.50, broad $7.50; sum all tiers run)
- `description`: your one-line commit message summary

**Monitoring spend:** `awk -F'\t' 'NR>1{s+=$15} END{printf "$%.2f\n",s}' autoresearch/results.tsv`
Hard cap: stop iterating if cumulative spend exceeds **$50 per session**. Report
to the human before the next Modal call if within $5 of the cap.

---

## 4. Variance Discipline

**Latency on H100 is highly deterministic.** The evaluator uses
`measure_iters=3` with `torch.cuda.synchronize` brackets and takes the median.
Variance between identical runs is typically <1%. Trust even small improvements
at the `full` tier.

**But suites are not disjoint.** The `quick` and `full` suites overlap in
case-shape distributions. A kernel highly specialized to `full`'s specific
(t_max, batch_heads, density) combinations can silently regress on `broad`. Run
`broad` every ~10 keeps to guard against this.

**When quick disagrees with your intuition:** quick uses fewer cases and has
more shape variance. If a change seems correct and quick_geo is borderline
(within 2%), promote to full anyway — full's larger case count is more stable.

---

## 5. The Metric, Decomposed

```
geometric_mean_family_latency_ms = exp( mean( log(median_sliding), log(median_global), log(median_retrieval) ) )
```

- Each family contributes equally to the geomean regardless of case count.
- **Always attack the worst (slowest) family first** — improving the slowest
  family multiplies geomean leverage. Read `full_sliding`, `full_global`,
  `full_retrieval` columns after every kept run.
- `worst_family_latency_ms` is a secondary ranking signal; leaderboard ranks by
  geomean first, then `worst` as a tiebreaker.
- `speedup_full = ref_full_geo / sub_full_geo` (> 1 = faster than reference).
  Track this progression: goal 1 is `speedup > 1.0`, goal 2 is `> 2.0`,
  goal 3 is `> 5.0`.

---

## 6. Idea Bank

Work through these roughly in order (earlier ideas are lower-risk / higher
expected value). After each kept commit, update your internal ranking of which
ideas remain promising.

### A. Triton Flash-Attention Kernel (highest priority)

The example submission uses a pure-PyTorch loop over q-blocks. A proper Triton
kernel can be 5–20× faster by keeping everything in SRAM and avoiding global
memory round-trips.

**Core structure:**
```python
@triton.jit
def _sparse_attn_fwd_kernel(
    Q, K, V, Out, Lse,
    row_ptr, col_idx, seq_lens,
    stride_qb, stride_qh, stride_qt, stride_qd,
    # ... other strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Each program handles one (batch_head, q_block) tile.
    bh_idx = tl.program_id(1)
    q_block_idx = tl.program_id(0)
    
    # Load Q tile for this q_block.
    # Walk col_idx[row_ptr[q_block_idx] : row_ptr[q_block_idx+1]].
    # For each K/V block in the sparse set:
    #   Load K/V tile, compute scores, apply causal mask, online-softmax update.
    # Write output tile and log-sum-exp.
```

Key choices to tune:
- `BLOCK_M` ∈ {32, 64, 128} — q-block tile size (must divide 128)
- `BLOCK_N` ∈ {32, 64, 128} — k/v-block tile size (must divide 128)
- `num_warps` ∈ {2, 4, 8}
- `num_stages` ∈ {1, 2, 3, 4} — software pipelining depth
- Accumulate in `tl.float32`; cast output to bf16/fp16 only at write time.
  Keep `lse` in fp32 (tight tolerance: 1e-5).

**Online softmax (Flash-Attn style):**
```python
# Per q-row accumulators (updated block by block):
m_i = -float('inf')  # running row-max
l_i = 0.0            # running normalizer (sum of exp)
acc = zeros(HEAD_DIM)  # running output accumulator

for each K/V block in sparse set:
    scores = dot(q_tile, k_tile.T) * SCALE  # [BLOCK_M, BLOCK_N]
    apply causal mask (key_pos <= q_pos) and padding mask
    m_new = max(m_i, row_max(scores))
    l_i = exp(m_i - m_new) * l_i + sum(exp(scores - m_new))
    acc = exp(m_i - m_new) * acc + softmax_num(scores, m_new) @ v_tile
    m_i = m_new

output = acc / l_i
lse = m_i + log(l_i)
```

### B. Per-Family Specialization

Each family has a distinct sparsity structure:

**`sliding_window`** — each q-block attends to exactly `window_blocks` prior
K/V blocks (plus itself). The col_idx sequence is contiguous and predictable.
Use a specialized kernel with a fixed-length inner loop (no gather needed if you
walk col_idx as a contiguous slice). Density is moderate.

**`sliding_window_global`** — like sliding_window but `global_blocks` extra
K/V blocks at the start of the sequence are always attended to by all q-blocks.
Handle in a two-phase loop: global blocks first (always attend), then window
blocks. Avoid double-counting if window overlaps global.

**`sliding_window_retrieval`** — sliding_window plus a scattered set of
`retrieval_blocks` K/V blocks chosen by a learned relevance score. The col_idx
set is not contiguous. This family has the highest variance in density and
benefits most from careful masking and load coalescing.

Route by family using `VARIANT_MANIFEST` — give each family its own variant
with its own kernel and/or (BLOCK_M, BLOCK_N, num_warps, num_stages).

### C. VARIANT_MANIFEST Design

Up to 16 variants. Each variant matches a `VariantSpec` (see
`attention_kernel_challenge/spec.py:VariantSpec`) and gets its own precompiled
kernel/config. Split by:

- Family (3 families → 3 base variants, then further split if needed)
- t_max range (short ≤ 2048, long > 2048 may benefit from different BLOCK_M)
- batch_heads range (large batch → more parallelism → different num_warps)

Example manifest (6 variants):
```python
VARIANT_MANIFEST = [
    {"name": "sw_short",   "families": ["sliding_window"],            "max_t_max": 2048},
    {"name": "sw_long",    "families": ["sliding_window"],            "min_t_max": 2049},
    {"name": "swg_short",  "families": ["sliding_window_global"],     "max_t_max": 2048},
    {"name": "swg_long",   "families": ["sliding_window_global"],     "min_t_max": 2049},
    {"name": "swr_small",  "families": ["sliding_window_retrieval"],  "max_batch_heads": 64},
    {"name": "swr_large",  "families": ["sliding_window_retrieval"],  "min_batch_heads": 65},
]
```

### D. Setup-Time Autotuning

Within the 30s `setup()` cap, sweep a small grid of (BLOCK_M, BLOCK_N,
num_warps, num_stages) per variant on a representative tensor. Time each config
with `torch.cuda.synchronize()` brackets and pick the fastest. Then call the
winning kernel once more to fully populate the compile cache before it freezes.

Keep the grid small (≤ 8 configs) to stay within the 30s cap. You can use
suite_specs to pick representative (t_max, batch_size, num_heads) shapes.

```python
def setup(suite_specs, device, variants):
    if device != "cuda":
        return
    import time
    for var in variants:
        best_ms, best_cfg = float('inf'), None
        for cfg in CONFIGS_FOR_VARIANT[var.name]:
            # warm up
            _run_kernel(representative_input, cfg)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                _run_kernel(representative_input, cfg)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 3 * 1000
            if ms < best_ms:
                best_ms, best_cfg = ms, cfg
        SELECTED_CONFIG[var.name] = best_cfg
        # warm up winning config to populate cache
        for _ in range(2):
            _run_kernel(representative_input, best_cfg)
        torch.cuda.synchronize()
```

### E. torch.compile Baseline

Before committing to a hand-written Triton kernel, try
`torch.compile(mode="reduce-overhead")` on the reference PyTorch implementation.
`torch.compile` is allowed because Inductor's transitive imports are exempt from
the allowlist. This may give a 2–4× speedup with zero kernel code, and it's a
useful baseline to beat.

```python
_compiled_attn = None

def setup(suite_specs, device, variants):
    global _compiled_attn
    if device != "cuda":
        return
    _compiled_attn = torch.compile(_attn_torch, mode="reduce-overhead", fullgraph=False)
    # Warm up with a representative input to trigger compilation now (not at eval time).
    _compiled_attn(warmup_q, warmup_k, warmup_v, warmup_row_ptr, warmup_col_idx, warmup_seq_lens)
    torch.cuda.synchronize()
```

### F. Memory Layout and Load Coalescing

- Store Q/K/V in `(batch*heads, seq, dim)` layout for coalesced loads across
  the batch_heads dimension that Triton programs are parallelized over.
- Use `tl.load` with masks to avoid out-of-bounds without conditional branches.
- For K/V blocks, prefer block-strided layouts so each Triton program loads a
  contiguous slice; gather only the col_idx-selected blocks, not the full tensor.

### G. BF16 vs FP16 Compute

The output tolerance is loose (`atol=1e-3`). Computing scores in bf16/fp16 with
fp32 accumulation is safe. Use `q.to(tl.bfloat16)` / `k.to(tl.bfloat16)` for
the `tl.dot` call (tensor cores require fp16 or bf16). Keep the running
accumulator and `lse` in fp32.

---

## 7. Anti-Patterns (hard rules — never violate)

1. **Never edit outside `autoresearch/submission/`.**
2. **Never import anything other than `torch`, `triton`, `numpy` directly.**
   Do not import `os`, `sys`, `math`, `functools`, etc. at the top level in
   submission files. Pure Python builtins (no-import) are fine.
3. **Never do compilation or autotuning at call time** (inside
   `block_sparse_attn_fwd`). All JIT triggering must happen inside `setup()`.
   Violating this causes `CompilationCacheMonitor` to fail the run.
4. **Never call `setup()` before committing.** The harness calls it; you don't.
5. **Never run `run_full.sh` without `run_quick.sh` showing improvement first.**
6. **Never run `run_broad.sh` more than once per ~10 kept commits.**
7. **Never run Modal tiers (smoke/quick/full/broad) from this CPU box's
   terminal while another Modal run is in flight** — one run at a time.
8. **Never git-commit `results.tsv` or `run.log`** — they are gitignored (or
   should be treated as such).
9. **Never skip appending to results.tsv** even for failed/discarded runs.
10. **Never break the torch fallback path** (used by tier 0 / CPU correctness).

---

## 8. Setup (one-time bootstrap, done by human)

The human must complete these steps before the loop can run. Do not attempt them yourself:

```bash
# 1. Create venv and install deps
cd /root/paradigm_autoresearch/attention-kernel-challenge
uv venv --python 3.11 --system-site-packages .venv
uv pip install --python .venv/bin/python torch numpy modal

# 2. Authenticate Modal (interactive browser flow)
.venv/bin/python -m modal setup

# 3. Deploy the warm evaluator app (needed for memory snapshot cold-start)
.venv/bin/python -m attention_kernel_challenge.cli backend setup-modal --gpu 'H100!:1'

# 4. Verify
.venv/bin/python -m attention_kernel_challenge.cli backend status --probe-modal

# 5. Create autoresearch branch
git checkout -b autoresearch/run-1

# 6. Capture frozen reference baselines (run once, ~$3 total)
source .venv/bin/activate
python -m attention_kernel_challenge.cli eval-reference \
  --suite full --backend modal --emit-json --redact-case-details \
  > autoresearch/baseline_full.json
python -m attention_kernel_challenge.cli eval-reference \
  --suite broad --backend modal --emit-json --redact-case-details \
  > autoresearch/baseline_broad.json

# 7. Run the tier-0 check on the bootstrap submission to seed results.tsv
bash autoresearch/run_check.sh
```

After step 7 you (the agent) are cleared to start the loop.

---

## 9. Quick Reference

```bash
# Check (free, seconds)
bash autoresearch/run_check.sh

# Smoke (cheap GPU, ~$0.10)
bash autoresearch/run_smoke.sh

# Quick / dev loop (~$0.30)
bash autoresearch/run_quick.sh

# Full / gate (~$1–2)
bash autoresearch/run_full.sh

# Broad / holdout milestone (~$5–10, use sparingly)
bash autoresearch/run_broad.sh

# Monitor spend
awk -F'\t' 'NR>1{s+=$15} END{printf "Total: $%.2f\n",s}' autoresearch/results.tsv

# Current best
sort -t$'\t' -k8,8g autoresearch/results.tsv | grep -v '^commit' | grep $'\tkeep\t' | tail -1

# What family is worst right now?
tail -5 autoresearch/results.tsv | awk -F'\t' '{print "sliding="$11" global="$12" retrieval="$13}'
```
