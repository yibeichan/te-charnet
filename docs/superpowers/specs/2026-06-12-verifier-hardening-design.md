# Verifier hardening: corrupt-NPZ handling + `--re-embed N` deep check

**Date:** 2026-06-12
**Status:** Approved
**Scope:** `scripts/verify_dialogue_embeddings.py` and its tests. The other
two verifiers (`verify_network_export.py`, `verify_stage2_network.py`) read
TSV/CSV only and are unaffected.

## Problem

Two residuals documented when the dialogue-embeddings verifier shipped
(PR #6, spec `2026-06-12-dialogue-embeddings-export-design.md`):

1. **Corrupt NPZ crashes the run.** `np.load()` is called bare at the
   product- and cache-NPZ load sites. A truncated or garbage file — or an
   archive missing the `key`/`vecs` members — raises and aborts the whole
   verification with a raw traceback instead of a clean per-episode FAIL
   and exit 1. A trust tool must fail cleanly on the exact class of damage
   it exists to detect.
2. **Encoder correctness is unvouched.** The verifier proves product
   vectors are *the cached vectors for exactly these texts* (key check +
   vector binding), but not that the encoder computed them correctly. The
   original spec sketched an optional `--re-embed N` deep check and
   deferred it as YAGNI; it is now being built.

## Design

### 1. Corrupt-NPZ handling

New helper in `verify_dialogue_embeddings.py`:

```python
def _load_npz(path: Path, members: tuple[str, ...] = ("key", "vecs")):
    """(dict, None) on success, (None, reason) on unreadable/incomplete NPZ."""
```

Wraps `np.load(path, allow_pickle=False)` plus member extraction, catching
`zipfile.BadZipFile`, `OSError`, `EOFError`, `ValueError`, and `KeyError`,
and returns a one-line reason string on failure. Both load sites use it:

- **Product NPZ unreadable** → `FAIL <ep>: product NPZ unreadable
  (<reason>)`; that episode's vector checks are skipped; the run continues
  to the remaining episodes.
- **Cache NPZ unreadable** → failure with the same shape as today's
  stale-cache case: vectors cannot be vouched for; regenerate via
  `scripts/export_dialogue_embeddings.py`.

Exit codes unchanged (0 pass / 1 any failure / 2 nothing checkable). No
traceback ever reaches the user for a bad data file. Deliberately **not** a
blanket `try/except` around `check_episode` — that would swallow genuine
verifier bugs and misreport them as data failures.

### 2. `--re-embed N` deep check

New CLI flags: `--re-embed N` (int, default 0 = off) and `--seed` (int,
default `None` = nondeterministic). After the normal per-episode pass:

- **Eligible pool:** episodes whose product NPZ loaded and passed the key
  check (re-embedding an already-failed episode adds nothing). Sample
  `min(N, len(pool))` episodes with `random.Random(seed)`; print the chosen
  episodes (and the seed when given) so a failure is reproducible.
- **Re-encode:** the sampled episodes' rebuilt texts go through a
  verifier-local encoder that replicates the export's settings exactly —
  `SentenceTransformer("all-MiniLM-L6-v2", device="cpu")`, `batch_size=64`,
  `normalize_embeddings=False`, cast to float32 (mirrors
  `src/charnet/topic_shift.py::minilm_encoder`, but implemented inline to
  preserve the verifier's "imports NOTHING from charnet" contract). The
  `sentence_transformers` import is lazy — only executed when
  `--re-embed > 0`; if unavailable, clean one-line error and exit 1.
- **Compare:** `np.allclose(product_vecs, fresh_vecs, atol=1e-5, rtol=0)`.
  Mismatch → `FAIL <ep>: re-embedded vecs differ from product (max abs
  diff <x>)`.
- **Testability:** `run()` gains `encoder_factory=None`; `None` builds the
  real model, tests inject a fake deterministic encoder (same style as the
  export tests — no model download in CI).
- **Summary:** final report gains a `re-embedded K episodes` line.

### 3. Tests (TDD, extend `tests/test_verify_dialogue_embeddings.py`)

- Truncated/garbage product NPZ → exit 1, message contains "unreadable",
  remaining episodes still checked.
- Product NPZ missing the `vecs` member → same clean failure.
- Corrupt cache NPZ → clean failure, not a crash.
- `--re-embed` with fake encoder: matching vecs → exit 0; product+cache
  vecs perturbed consistently (so key check and binding still pass) →
  re-embed catches it → exit 1.
- Seeded sampling deterministic; `N > len(pool)` clamps cleanly.

### 4. Docs

- Verifier module docstring: add the two new behaviors.
- Original export spec's "Residual limit" paragraph: one-line note that
  `--re-embed` now exists.

## Out of scope

- The two TSV/CSV verifiers (no NPZ surface).
- Any change to the export script or cache format.
- Tolerance tuning beyond `atol=1e-5` (revisit only if a real-hardware
  deep check produces false positives).
