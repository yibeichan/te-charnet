# Verifier Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/verify_dialogue_embeddings.py` fail cleanly (exit 1, per-episode FAIL line) on corrupt/incomplete NPZ files, and add an optional `--re-embed N` deep check that re-encodes N sampled episodes with the real model and compares to product vectors.

**Architecture:** All changes live in one script (`scripts/verify_dialogue_embeddings.py`) and its test file. A `_load_npz` helper converts NPZ read errors into `(None, reason)` so both load sites report per-episode failures instead of crashing. `check_episode` additionally returns a deep-check payload (texts + product vecs) for fully-passing episodes; `run()` samples from that pool and compares re-encoded vectors. The real encoder is built inline (lazy `sentence_transformers` import) to preserve the verifier's no-charnet-imports independence contract; tests inject a fake via a new `encoder_factory` parameter on `run()`.

**Tech Stack:** Python 3.10+, numpy, pandas, pytest. `sentence-transformers` is only imported at runtime when `--re-embed > 0` — never in tests.

**Spec:** `docs/superpowers/specs/2026-06-12-verifier-hardening-design.md`

**Working branch:** `008-verifier-hardening` (already created; spec committed).

**Test command (repo root):** `pytest tests/test_verify_dialogue_embeddings.py -v` — full suite + lint before final commit: `pytest && ruff check .`

---

### Task 1: `_load_npz` helper + clean failure on corrupt/incomplete product NPZ

**Files:**
- Modify: `scripts/verify_dialogue_embeddings.py`
- Test: `tests/test_verify_dialogue_embeddings.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_verify_dialogue_embeddings.py` (after `test_nan_scene_rows_are_accounted`). Also add `import shutil` to the imports at the top of the file.

```python
def _clone_episode(sentences, product, cache, ep="s01e02a"):
    """Copy the fixture episode under a second episode id (same texts/key)."""
    shutil.copy(sentences / "s1" / "friends_s01e01a_sentence_speaker_table.tsv",
                sentences / "s1" / f"friends_{ep}_sentence_speaker_table.tsv")
    shutil.copy(product / "s1" / "friends_s01e01a_dialogue_turns.tsv",
                product / "s1" / f"friends_{ep}_dialogue_turns.tsv")
    shutil.copy(product / "s1" / "friends_s01e01a_dialogue_embeddings.npz",
                product / "s1" / f"friends_{ep}_dialogue_embeddings.npz")
    shutil.copy(cache / "s1" / "s01e01a.npz", cache / "s1" / f"{ep}.npz")


def test_truncated_product_npz_fails_cleanly(tmp_path, capsys):
    sentences, product, cache = _build_fixture(tmp_path)
    _clone_episode(sentences, product, cache)
    npz = product / "s1" / "friends_s01e01a_dialogue_embeddings.npz"
    npz.write_bytes(b"PK\x03\x04 this is not a real zip archive")
    # must not raise; bad episode FAILs, the cloned episode still passes
    rc = _run(sentences, product, cache, extra=("--expected-dim", "4"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "unreadable" in out
    assert "s01e02a" not in "".join(l for l in out.splitlines() if "FAIL" in l)


def test_product_npz_missing_vecs_member_fails_cleanly(tmp_path, capsys):
    sentences, product, cache = _build_fixture(tmp_path)
    npz = product / "s1" / "friends_s01e01a_dialogue_embeddings.npz"
    d = dict(np.load(npz, allow_pickle=False))
    np.savez(npz, key=d["key"])  # drop the vecs member
    rc = _run(sentences, product, cache, extra=("--expected-dim", "4"))
    assert rc == 1
    assert "unreadable" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verify_dialogue_embeddings.py -k "unreadable or truncated or missing_vecs" -v`

Expected: both new tests FAIL — `test_truncated_product_npz_fails_cleanly` errors with an exception escaping `run()` (zipfile/ValueError traceback), `test_product_npz_missing_vecs_member_fails_cleanly` errors with `KeyError`.

- [ ] **Step 3: Implement `_load_npz` and use it at the product load site**

In `scripts/verify_dialogue_embeddings.py`:

Add `import zipfile` to the imports (stdlib group, after `import sys`).

Add the helper after `_texts_key`:

```python
def _load_npz(path: Path, members: tuple[str, ...] = ("key", "vecs")):
    """(dict, None) on success, (None, reason) on unreadable/incomplete NPZ."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {m: npz[m] for m in members}, None
    except KeyError as e:
        return None, f"missing member {e}"
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"
```

In `check_episode`, replace the product load (currently `prod = np.load(npz_path, allow_pickle=False)` and the following key check):

```python
    key = _texts_key(texts)
    prod, load_err = _load_npz(npz_path)
    if load_err:
        errs.append(f"{ep}: product NPZ unreadable ({load_err})")
        return errs, False
    if str(prod["key"]) != key:
        errs.append(f"{ep}: product NPZ key != recomputed text hash")
    vecs = prod["vecs"]
```

(The dtype/shape/finite checks and everything below stay as they are.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verify_dialogue_embeddings.py -v`

Expected: all tests PASS (the two new ones plus the existing ten).

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_dialogue_embeddings.py tests/test_verify_dialogue_embeddings.py
git commit -m "Verifier: fail cleanly (exit 1) on corrupt/incomplete product NPZ"
```

---

### Task 2: Clean failure on corrupt cache NPZ

**Files:**
- Modify: `scripts/verify_dialogue_embeddings.py`
- Test: `tests/test_verify_dialogue_embeddings.py`

- [ ] **Step 1: Write the failing test**

```python
def test_corrupt_cache_npz_fails_cleanly(tmp_path, capsys):
    sentences, product, cache = _build_fixture(tmp_path)
    (cache / "s1" / "s01e01a.npz").write_bytes(b"\x00\x01garbage")
    rc = _run(sentences, product, cache, extra=("--expected-dim", "4"))
    assert rc == 1
    assert "cache NPZ unreadable" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verify_dialogue_embeddings.py::test_corrupt_cache_npz_fails_cleanly -v`

Expected: FAIL — exception escapes from the bare `np.load(cache_path, ...)`.

- [ ] **Step 3: Use `_load_npz` at the cache load site**

In `check_episode`, replace the cache branch (currently `cached = np.load(cache_path, allow_pickle=False)` and the two checks after it):

```python
    else:
        cached, load_err = _load_npz(cache_path)
        if load_err:
            errs.append(f"{ep}: cache NPZ unreadable ({load_err}) — vectors "
                        f"cannot be vouched for (regenerate via "
                        f"scripts/export_dialogue_embeddings.py)")
        elif str(cached["key"]) != key:
            errs.append(f"{ep}: cache NPZ key != recomputed text hash (stale cache)")
        elif not np.array_equal(vecs, cached["vecs"]):
            errs.append(f"{ep}: product vecs != cache vecs (binding broken)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verify_dialogue_embeddings.py -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_dialogue_embeddings.py tests/test_verify_dialogue_embeddings.py
git commit -m "Verifier: fail cleanly on corrupt cache NPZ"
```

---

### Task 3: `--re-embed N` deep check

**Files:**
- Modify: `scripts/verify_dialogue_embeddings.py`
- Test: `tests/test_verify_dialogue_embeddings.py`

- [ ] **Step 1: Write the failing tests**

First extend the `_run` helper so tests can inject an encoder factory — replace the existing `_run` with:

```python
def _run(sentences, product, cache, extra=(), **kw):
    argv = ["verify_dialogue_embeddings.py",
            "--tables-root", str(sentences), "--product-root", str(product),
            "--cache-root", str(cache), *extra]
    return V.run(argv[1:], **kw)
```

Then add the tests:

```python
def _fake_encoder_factory():
    """Reproduces the fixture vecs: arange over (n_texts, 4)."""
    def encode(texts):
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)
    return encode


def test_re_embed_matching_vecs_passes(tmp_path):
    roots = _build_fixture(tmp_path)
    assert _run(*roots, extra=("--expected-dim", "4", "--re-embed", "1"),
                encoder_factory=_fake_encoder_factory) == 0


def test_re_embed_catches_consistent_perturbation(tmp_path):
    # perturb product AND cache identically: key check and binding both pass,
    # only re-embedding can catch it
    sentences, product, cache = _build_fixture(tmp_path)
    for path in (product / "s1" / "friends_s01e01a_dialogue_embeddings.npz",
                 cache / "s1" / "s01e01a.npz"):
        d = dict(np.load(path, allow_pickle=False))
        d["vecs"] = d["vecs"].copy()
        d["vecs"][1, 2] += 0.5
        np.savez(path, **d)
    assert _run(sentences, product, cache,
                extra=("--expected-dim", "4")) == 0  # binding intact
    assert _run(sentences, product, cache,
                extra=("--expected-dim", "4", "--re-embed", "1"),
                encoder_factory=_fake_encoder_factory) == 1


def test_re_embed_clamps_n_to_pool_size(tmp_path, capsys):
    roots = _build_fixture(tmp_path)
    rc = _run(*roots, extra=("--expected-dim", "4", "--re-embed", "5"),
              encoder_factory=_fake_encoder_factory)
    assert rc == 0
    assert "re-embedded 1 episode" in capsys.readouterr().out


def test_re_embed_seed_makes_sampling_deterministic(tmp_path, capsys):
    sentences, product, cache = _build_fixture(tmp_path)
    _clone_episode(sentences, product, cache)

    def chosen_line():
        _run(sentences, product, cache,
             extra=("--expected-dim", "4", "--re-embed", "1", "--seed", "7"),
             encoder_factory=_fake_encoder_factory)
        out = capsys.readouterr().out
        return next(l for l in out.splitlines() if "Re-embed" in l)

    assert chosen_line() == chosen_line()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verify_dialogue_embeddings.py -k re_embed -v`

Expected: all four FAIL — `run()` does not accept `encoder_factory` (TypeError) and `--re-embed` is an unknown argument (argparse SystemExit).

- [ ] **Step 3: Implement `--re-embed`**

In `scripts/verify_dialogue_embeddings.py`:

Add `import random` to the stdlib imports.

Add the real-encoder builder after `_load_npz` (mirrors `charnet.topic_shift.minilm_encoder` deliberately, but inline — the verifier imports nothing from charnet):

```python
def _build_real_encoder():
    """Lazy MiniLM encoder matching the export's settings exactly.

    Returns None when sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer(MODEL_ID, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()),
                            dtype=np.float32)
        return np.asarray(
            model.encode(texts, batch_size=64, show_progress_bar=False,
                         normalize_embeddings=False),
            dtype=np.float32,
        )

    return encode
```

Change `check_episode`'s signature and return value to expose a deep-check payload — returns `(errs, skipped, deep)` where `deep` is `{"texts": texts, "vecs": vecs}` only when the product NPZ loaded and the episode had no errors:

- the docstring becomes: `"""Returns (mismatches, skipped, deep). skipped=True when the product NPZ is absent; deep carries texts+vecs for re-embedding when the episode fully passed."""`
- every existing `return errs, False` inside `check_episode` becomes `return errs, False, None`
- `return errs, True` (NPZ-absent skip) becomes `return errs, True, None`
- the final return becomes:

```python
    deep = {"texts": texts, "vecs": vecs} if not errs else None
    return errs, False, deep
```

In `run()`:

- signature: `def run(argv: list[str] | None = None, encoder_factory=None) -> int:`
- add the flags after `--expected-dim`:

```python
    ap.add_argument("--re-embed", type=int, default=0, metavar="N",
                    help="re-encode N sampled passing episodes with the real "
                         "model and compare to product vecs")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for --re-embed episode sampling")
```

- collect the pool in the episode loop (note: argparse turns `--re-embed` into `args.re_embed`):

```python
    all_errs: list[str] = []
    pool: list[tuple[str, dict]] = []
    n_checked = n_skipped = 0
    for tpath in tables:
        ep = _episode_id(tpath)
        errs, skipped, deep = check_episode(ep, tpath, Path(args.product_root),
                                            Path(args.cache_root), args.expected_dim)
        all_errs.extend(errs)
        if deep is not None:
            pool.append((ep, deep))
        if skipped and not errs:
            n_skipped += 1
            print(f"  skip {ep}: product NPZ absent (TSV checks only)")
        else:
            n_checked += 1
```

- add the deep check between the loop and the summary print:

```python
    if args.re_embed > 0 and pool:
        rng = random.Random(args.seed)
        chosen = rng.sample(sorted(pool), min(args.re_embed, len(pool)))
        print(f"Re-embed deep check: {[ep for ep, _ in chosen]}"
              + (f" (seed={args.seed})" if args.seed is not None else ""))
        encoder = encoder_factory() if encoder_factory else _build_real_encoder()
        if encoder is None:
            print("  FAIL --re-embed requires sentence-transformers (not installed)")
            all_errs.append("--re-embed: sentence-transformers not installed")
        else:
            for ep, deep in chosen:
                fresh = encoder(deep["texts"])
                if fresh.shape != deep["vecs"].shape:
                    all_errs.append(f"{ep}: re-embedded shape {fresh.shape} != "
                                    f"product {deep['vecs'].shape}")
                elif not np.allclose(deep["vecs"], fresh, atol=1e-5, rtol=0):
                    diff = float(np.abs(deep["vecs"] - fresh).max())
                    all_errs.append(f"{ep}: re-embedded vecs differ from product "
                                    f"(max abs diff {diff:.3g})")
            print(f"  re-embedded {len(chosen)} episode(s)")
```

(`sorted(pool)` makes `rng.sample` independent of filesystem glob order, so a seed is reproducible across machines. Tuples sort by episode id — distinct by construction — so the dicts are never compared.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_verify_dialogue_embeddings.py -v`

Expected: all PASS (17 tests: 10 original + 3 from Tasks 1–2 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_dialogue_embeddings.py tests/test_verify_dialogue_embeddings.py
git commit -m "Verifier: add --re-embed N deep check (encoder-correctness residual)"
```

---

### Task 4: Docs + full-suite verification

**Files:**
- Modify: `scripts/verify_dialogue_embeddings.py` (module docstring)
- Modify: `docs/superpowers/specs/2026-06-12-dialogue-embeddings-export-design.md` (residual-limit paragraph)

- [ ] **Step 1: Update the verifier module docstring**

In the numbered-check list of the docstring, after item 5 (`sanity`), add:

```
  6. deep check     — with --re-embed N, re-encodes N sampled passing
     episodes with the real model (lazy import) and compares to product
     vecs within atol=1e-5; --seed makes the sample reproducible.

Corrupt or member-incomplete NPZ files (product or cache) are clean
per-episode FAILures (exit 1), never tracebacks.
```

- [ ] **Step 2: Update the export spec's residual paragraph**

In `docs/superpowers/specs/2026-06-12-dialogue-embeddings-export-design.md`, the paragraph beginning `**Residual limit (documented, accepted):**` — append one sentence at its end:

```
*(Update 2026-06-12: `--re-embed N` and clean corrupt-NPZ handling shipped —
see `2026-06-12-verifier-hardening-design.md`.)*
```

- [ ] **Step 3: Run the full suite and lint**

Run (repo root): `pytest && ruff check .`

Expected: all tests pass (176 pre-existing + 7 new = 183), ruff clean.

- [ ] **Step 4: Smoke-run the verifier against real data (no re-embed)**

Run: `python scripts/verify_dialogue_embeddings.py`

Expected: `Checked 341 episodes (0 NPZ-absent skips)` / `All checks passed` / exit 0 — confirms the refactor didn't change real-data behavior. (If NPZ files are absent on this checkout, skips are expected instead; that's also fine.)

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_dialogue_embeddings.py docs/superpowers/specs/2026-06-12-dialogue-embeddings-export-design.md
git commit -m "Docs: record --re-embed + corrupt-NPZ hardening in verifier docstring and export spec"
```
