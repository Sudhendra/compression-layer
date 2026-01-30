# MLX Run Storage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Store MLX local training runs with per-run checkpoints, logs, and metadata so results are never overwritten.

**Architecture:** Create a run directory under `models/runs/mlx/<timestamp>` and route MLX LoRA outputs there. Persist `run.json`, the LoRA config, and training logs in that directory, then update `models/runs/mlx/latest` as a symlink to the newest run.

**Tech Stack:** Python (stdlib + pydantic), MLX LoRA CLI, filesystem symlinks.

---

### Task 1: Add run storage utilities

**Files:**
- Create: `src/training/run_storage.py`
- Modify: `src/training/__init__.py`
- Test: `tests/test_run_storage.py`

**Step 1: Write the failing test**

```python
def test_create_run_dir_creates_expected_structure(tmp_path):
    run_dir = create_run_dir(tmp_path)
    assert run_dir.exists()
    assert run_dir.name
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_storage.py::test_create_run_dir_creates_expected_structure -v`
Expected: FAIL with "NameError: name 'create_run_dir' is not defined"

**Step 3: Write minimal implementation**

```python
def create_run_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_storage.py::test_create_run_dir_creates_expected_structure -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/run_storage.py src/training/__init__.py tests/test_run_storage.py
git commit -m "feat: add MLX run storage helpers"
```

---

### Task 2: Route MLX training output into per-run directories

**Files:**
- Modify: `src/training/train_mlx.py`
- Test: `tests/test_train_mlx_runs.py`

**Step 1: Write the failing test**

```python
def test_prepare_run_paths_writes_metadata(tmp_path):
    config = MLXTrainingConfig(adapter_path=tmp_path / "adapter")
    paths = prepare_run_paths(config, tmp_path)
    assert paths.meta_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_mlx_runs.py::test_prepare_run_paths_writes_metadata -v`
Expected: FAIL with "NameError: name 'prepare_run_paths' is not defined"

**Step 3: Write minimal implementation**

```python
def prepare_run_paths(config: MLXTrainingConfig, runs_root: Path) -> RunPaths:
    run_dir = create_run_dir(runs_root)
    meta = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha(),
        "data_dir": str(config.data_dir),
        "model": config.model,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "iters": config.iters,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }
    meta_path = run_dir / "run.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return RunPaths(run_dir=run_dir, meta_path=meta_path)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_mlx_runs.py::test_prepare_run_paths_writes_metadata -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/train_mlx.py tests/test_train_mlx_runs.py
git commit -m "feat: store MLX training output per run"
```

---

### Task 3: Capture training logs and update latest symlink

**Files:**
- Modify: `src/training/train_mlx.py`
- Modify: `scripts/train_local.py`
- Test: `tests/test_train_mlx_runs.py`

**Step 1: Write the failing test**

```python
def test_update_latest_symlink(tmp_path):
    run_dir = tmp_path / "2026-01-30_12-00-00"
    run_dir.mkdir()
    update_latest_symlink(tmp_path, run_dir)
    assert (tmp_path / "latest").is_symlink()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_mlx_runs.py::test_update_latest_symlink -v`
Expected: FAIL with "NameError: name 'update_latest_symlink' is not defined"

**Step 3: Write minimal implementation**

```python
def update_latest_symlink(runs_root: Path, run_dir: Path) -> None:
    latest_path = runs_root / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir, target_is_directory=True)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_mlx_runs.py::test_update_latest_symlink -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/train_mlx.py scripts/train_local.py tests/test_train_mlx_runs.py
git commit -m "feat: log MLX runs and update latest symlink"
```

---

### Task 4: Update documentation to reflect run storage

**Files:**
- Modify: `docs/SETUP.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/COMPRESSION_LAYER_IMPLEMENTATION_PLAN.md`

**Step 1: Update setup instructions**

Add a “Run outputs” section explaining `models/runs/mlx/<timestamp>` layout and the `latest` symlink.

**Step 2: Update task checklist**

Add a Phase 3 task for “MLX run storage + checkpoint logging” under Training.

**Step 3: Update implementation plan**

Add a note in Phase 3 that MLX runs are stored in per-run directories for repeatability.

**Step 4: Commit**

```bash
git add docs/SETUP.md docs/TASKS.md docs/COMPRESSION_LAYER_IMPLEMENTATION_PLAN.md
git commit -m "docs: document MLX run storage"
```
