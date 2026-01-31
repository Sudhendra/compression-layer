# Tinker SDK Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the legacy Tinker training pipeline with a robust adapter for the latest Tinker SDK, preserving the CLI workflow (upload/train/status/download) and updating docs.

**Architecture:** Introduce a Tinker SDK adapter that wraps ServiceClient/TrainingClient/RestClient, add a dataset renderer that converts JSONL pairs to `tinker.types.Datum`, and implement a synchronous training loop that records run metadata and checkpoints for status/download commands.

**Tech Stack:** Python 3.11+, Tinker SDK, Pydantic dataclasses, Rich CLI output, JSONL data handling.

---

### Task 1: Add Tinker SDK dependency + adapter skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/training/tinker_sdk.py`
- Modify: `src/training/__init__.py`
- Test: `tests/test_tinker_sdk_client.py`

**Step 1: Write the failing test**

```python
def test_tinker_client_requires_api_key() -> None:
    client = TinkerSDKClient(api_key="")
    assert not client.is_available
```

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_tinker_sdk_client.py::test_tinker_client_requires_api_key -v`
Expected: FAIL with "TinkerSDKClient not found" or similar

**Step 3: Write minimal implementation**

```python
class TinkerSDKClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("TINKER_API_KEY", "")
        self.is_available = bool(self.api_key)
```

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_tinker_sdk_client.py::test_tinker_client_requires_api_key -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml src/training/tinker_sdk.py src/training/__init__.py tests/test_tinker_sdk_client.py
git commit -m "feat: add tinker sdk adapter skeleton"
```

---

### Task 2: Implement dataset renderer for Tinker Datum

**Files:**
- Create: `src/training/tinker_data.py`
- Test: `tests/test_tinker_data.py`

**Step 1: Write the failing test**

```python
def test_render_chat_example_to_datum() -> None:
    tokenizer = FakeTokenizer()
    datum = render_chat_example(DUMMY_MESSAGES, tokenizer)
    assert datum.loss_fn_inputs["target_tokens"]
```

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_tinker_data.py::test_render_chat_example_to_datum -v`
Expected: FAIL with "render_chat_example not found"

**Step 3: Write minimal implementation**

```python
def render_chat_example(messages: list[dict[str, str]], tokenizer: TokenizerLike) -> types.Datum:
    prompt, completion = split_prompt_completion(messages)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )
```

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_tinker_data.py::test_render_chat_example_to_datum -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/tinker_data.py tests/test_tinker_data.py
git commit -m "feat: add tinker datum renderer"
```

---

### Task 3: Implement SDK-backed training loop + run metadata

**Files:**
- Create: `src/training/train_tinker.py`
- Modify: `src/training/__init__.py`
- Test: `tests/test_train_tinker.py`

**Step 1: Write the failing test**

```python
def test_train_on_tinker_records_run_metadata(tmp_path: Path) -> None:
    result = train_on_tinker(config, api_key="test", output_dir=tmp_path)
    assert result.run_id is not None
    assert (tmp_path / "runs" / f"{result.run_id}.json").exists()
```

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_train_tinker.py::test_train_on_tinker_records_run_metadata -v`
Expected: FAIL with "train_on_tinker not found"

**Step 3: Write minimal implementation**

```python
def train_on_tinker(config: TinkerTrainingConfig, api_key: str | None = None, output_dir: Path | None = None) -> TinkerTrainingResult:
    client = TinkerSDKClient(api_key=api_key)
    training_client = client.create_training_client(config)
    metadata = run_training_loop(training_client, config)
    metadata_path = write_run_metadata(metadata, output_dir)
    return TinkerTrainingResult(success=True, run_id=metadata.run_id, metadata_path=metadata_path)
```

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_train_tinker.py::test_train_on_tinker_records_run_metadata -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/train_tinker.py src/training/__init__.py tests/test_train_tinker.py
git commit -m "feat: add tinker sdk training loop"
```

---

### Task 4: Update CLI script for new SDK flow

**Files:**
- Create: `scripts/train_tinker.py`
- Modify: `src/utils/config.py`
- Test: `tests/test_train_tinker_cli.py`

**Step 1: Write the failing test**

```python
def test_cli_status_reads_run_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "run-123"
    (tmp_path / "runs" / f"{run_id}.json").write_text("{}")
    exit_code = main(["--status", run_id, "--output", str(tmp_path)])
    assert exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_train_tinker_cli.py::test_cli_status_reads_run_metadata -v`
Expected: FAIL with "main not found"

**Step 3: Write minimal implementation**

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        return print_status(args.status, output_dir=args.output)
    result = train_on_tinker(config, api_key=settings.tinker_api_key, output_dir=args.output)
    return 0 if result.success else 1
```

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_train_tinker_cli.py::test_cli_status_reads_run_metadata -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/train_tinker.py src/utils/config.py tests/test_train_tinker_cli.py
git commit -m "feat: add tinker sdk cli"
```

---

### Task 5: Update docs and config to new SDK workflow

**Files:**
- Modify: `docs/SETUP.md`
- Modify: `docs/AGENTS.md`
- Modify: `docs/COMPRESSION_LAYER_IMPLEMENTATION_PLAN.md`
- Modify: `configs/training.yaml`

**Step 1: Update docs with new SDK flow**

Replace legacy `tinker.Client` / `tinker train` examples with `ServiceClient` / `TrainingClient` usage and the `python scripts/train_tinker.py` CLI workflow.

**Step 2: Run doc checks (optional)**

Run: `./.venv/bin/pytest tests -k "docs" -v`
Expected: PASS (or no matching tests)

**Step 3: Commit**

```bash
git add docs/SETUP.md docs/AGENTS.md docs/COMPRESSION_LAYER_IMPLEMENTATION_PLAN.md configs/training.yaml
git commit -m "docs: update tinker sdk workflow"
```

---

### Task 6: Full test pass

**Step 1: Run tests**

Run: `./.venv/bin/pytest`
Expected: PASS (allow the existing skipped test)

**Step 2: Commit final verification note (optional)**

```bash
git status --short
```

---

## Notes + Assumptions

- Use `tinker.ServiceClient` + `create_lora_training_client` with `rank` from config.
- Use `tinker.types.AdamParams` for optimizer settings.
- Use `RestClient.get_checkpoint_archive_url_from_tinker_path` for downloads (SDK docs).
- Store run metadata in `models/adapters/tinker/runs/<run_id>.json` to keep status checks deterministic.
- Keep CLI sync-only (no async paths).
