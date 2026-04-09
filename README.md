# Attention Kernel Challenge Harness

Public evaluation harness for the Attention Kernel Challenge block-sparse
forward-attention track.

This repository includes:

- fixed block-sparse causal semantics with `head_dim=128` and `block_size=128`
- built-in local suites plus published benchmark distributions
- a numerically faithful FP32 reference implementation
- correctness checks for both `o` and `lse`
- submission loading from `submission.py` with required `VARIANT_MANIFEST`
- local execution, optional Modal-backed H100 execution, and an opt-in local
  `nsjail` sandbox
- JSON suite-manifest loading for custom evaluation splits

## Quick Start

Create a repo-local Python 3.11 environment. The commands below assume `torch`
and `numpy` are already available through `--system-site-packages`.

```bash
uv venv --python 3.11 --system-site-packages .venv
```

Install Modal into that environment if you want remote H100 runs:

```bash
uv pip install --python .venv/bin/python modal
```

Check local and backend readiness:

```bash
.venv/bin/python -m attention_kernel_challenge.cli doctor
.venv/bin/python -m attention_kernel_challenge.cli backend status --probe-modal
```

Authenticate Modal once:

```bash
.venv/bin/python -m modal setup
```

Set up the default Modal backend and deploy the evaluator app:

```bash
.venv/bin/python -m attention_kernel_challenge.cli backend setup-modal --gpu H100!:1
```

Inspect the built-in suites:

```bash
.venv/bin/python -m attention_kernel_challenge.cli show-suite --suite smoke
.venv/bin/python -m attention_kernel_challenge.cli show-suite --suite quick
.venv/bin/python -m attention_kernel_challenge.cli show-suite --suite full
.venv/bin/python -m attention_kernel_challenge.cli show-suite --suite broad
```

Run the reference path:

```bash
.venv/bin/python -m attention_kernel_challenge.cli eval-reference --suite smoke --backend local
.venv/bin/python -m attention_kernel_challenge.cli eval-reference --suite quick --backend modal
```

Evaluate the example submission:

```bash
.venv/bin/python -m attention_kernel_challenge.cli eval-submission --submission-dir example_submission --suite smoke --backend local
.venv/bin/python -m attention_kernel_challenge.cli eval-submission --submission-dir example_submission --suite smoke --backend local --device mps --serverlike
.venv/bin/python -m attention_kernel_challenge.cli eval-submission --submission-dir example_submission --suite full --backend modal
```

Use `--device cuda` on a local NVIDIA machine, `--device mps` on Apple
Silicon, or `--device cpu` if you only want an isolated dry run.

If you want to ignore any deployed Modal evaluator for a single run, add
`--modal-one-shot`.

## Recommended Workflow

1. Use `smoke` locally to get correctness working first.
2. Use local `eval-submission --serverlike` to catch isolated setup/import
   issues before paying for a remote run.
3. Use `quick` for inexpensive remote H100 sanity checks.
4. Use `full` for more serious benchmark iteration.
5. Use `broad` more sparingly for wider coverage.

Local CPU runs are correctness-only. Local GPU runs can be useful for
iteration, but only remote H100 runs are challenge-relevant performance
signals. Modal-backed runs always include correctness checking, so treat local
timing as a coarse pruning signal rather than a final ranking.

If you are experimenting with setup-time compilation, local `--serverlike`
runs are still only a partial rehearsal on non-CUDA machines. They catch
import-policy and isolation issues, but they do not fully validate the H100
codegen path.

## Suites And Manifests

- `smoke` and `local-dev` are explicit local suites; `show-suite` prints exact
  case specs for them.
- `quick`, `full`, and `broad` publish workload distributions; `show-suite`
  prints distribution metadata without exposing exact benchmark seeds.
- `--suite-manifest /path/to/manifest.json` loads an explicit manifest instead
  of a built-in suite.
- `--emit-json` produces machine-readable output.

Example manifest usage:

```bash
.venv/bin/python -m attention_kernel_challenge.cli show-suite --suite-manifest /path/to/public_manifest.json
.venv/bin/python -m attention_kernel_challenge.cli eval-submission --submission-dir example_submission --suite-manifest /path/to/public_manifest.json --backend local
```

If you run a hosted event with hidden manifests, keep those manifests out of the
public repository and use `--redact-case-details` when you need to return
results without exposing case-level information.

## Submission Contract

Main-track submissions are a directory rooted at `submission.py`.

Required:

- `submission.py`
- callable `block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens) -> (o, lse)`
- `VARIANT_MANIFEST`, with at least one declared variant

Each `VARIANT_MANIFEST` entry may specialize on:

- `families`
- `t_max` or `min_t_max` / `max_t_max`
- `batch_heads` or `min_batch_heads` / `max_batch_heads`
- `batch_size` or `min_batch_size` / `max_batch_size`
- `num_heads` or `min_num_heads` / `max_num_heads`
- `window_blocks` or `min_window_blocks` / `max_window_blocks`
- `global_blocks` or `min_global_blocks` / `max_global_blocks`
- `retrieval_blocks` or `min_retrieval_blocks` / `max_retrieval_blocks`
- `retrieval_local_bias` or `min_retrieval_local_bias` / `max_retrieval_local_bias`

Use either the exact field or the min/max bounds for a given key, not both.

Optional:

- `setup()`
- helper local Python modules and small metadata files

Allowed third-party imports:

- `torch`
- `triton`
- `numpy`

The harness statically rejects non-allowlisted third-party imports, enforces
setup/build limits, rejects oversized submissions, freezes selected cache paths
after setup, and blocks subprocess/network escape paths during submission
execution.

Compiler-owned transitive imports are allowed when they are initiated by
`torch` or `triton` during setup or execution. Direct submission imports still
have to stay inside the allowlist above.

## Execution Model

- `setup()` receives only the public specialization surface needed for build
  planning.
- The evaluator warms one synthetic setup case per public descriptor, freezes
  compilation-cache state, then times fresh suite-shaped realizations.
- The default setup cap is 30 seconds. Local runs can override it with
  `--setup-timeout-s`; Modal-backed runs use the default cap.
- Setup-time `torch.compile(...)` on CUDA is a supported path. The harness
  seeds Triton's fixed runtime helper cache outside the submission sandbox and
  keeps Inductor in single-process compile mode so the hosted H100 path can
  compile during `setup()`.
- Raw `@triton.jit` kernels are also supported during `setup()`. The sandbox
  allows only the narrow Triton launcher build path: trusted host compilers,
  trusted Triton build call sites, and launcher artifacts rooted under the
  monitored cache/temp directory. Late compilation after setup is still
  rejected by cache freezing.
- Local `--serverlike` runs isolate submission execution and report setup device
  as `cuda` to catch some server-only setup surprises.
- Modal-backed submission runs execute untrusted submission code in a separate
  worker process from the trusted evaluator.

Minimal supported pattern for setup-time compilation:

```python
def setup(suite_specs, device, variants):
    if str(device) != "cuda" or not torch.cuda.is_available():
        return None

    def helper(x):
        return x + 1

    compiled = torch.compile(helper, fullgraph=True)
    compiled(torch.ones(8, device="cuda"))
```

## Sandbox

`--sandbox` is opt-in, Linux-only, and supported only with `--backend local`.
When enabled, the harness runs under `nsjail` with dropped capabilities, no
loopback interface, no `/proc`, and a writable `/tmp`.

## Example Submission

[`example_submission/submission.py`](example_submission/submission.py) is a
sparse-aware correctness baseline and a compact template for the expected
submission shape.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
