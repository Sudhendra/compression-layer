# Contributing to LLM Compression Layer

## Branch Strategy

We use a phase-based branching strategy:

```
main (protected)
  └── phase-1-foundation
  └── phase-2-generation
  └── phase-3-training
  └── phase-4-inference
  └── phase-5-evaluation
```

### Branch Rules

1. **main** - Protected branch, requires:
   - All CI checks passing
   - At least 1 approval (if collaborators)
   - No direct pushes

2. **phase-X-*** - Feature branches for each implementation phase
   - Branch from `main`
   - PR back to `main` when phase complete
   - Delete after merge

3. **fix/*** - Bug fix branches
4. **docs/*** - Documentation updates

## Development Workflow

### Setup

```bash
# Clone the repo
git clone https://github.com/Sudhendra/compression-layer.git
cd compression-layer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Making Changes

```bash
# Create feature branch
git checkout -b phase-2-generation

# Make changes...

# Run checks before committing
make check  # or: ruff check src/ && mypy src/ && pytest tests/

# Commit with descriptive message
git add .
git commit -m "feat: add seed generator for compression pairs"

# Push and create PR
git push -u origin phase-2-generation
```

### Commit Message Format

Follow conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### Pull Request Process

1. Ensure all CI checks pass
2. Update TASKS.md to mark completed items
3. Add description of changes
4. Request review if needed
5. Squash and merge

## CI Requirements

All PRs must pass:

- **Linting**: `ruff check src/ tests/`
- **Formatting**: `ruff format --check src/ tests/`
- **Type checking**: `mypy src/`
- **Tests**: `pytest tests/ -v`

## Data Management

Large data files are **not** tracked in git. Use:

- **DVC** (Data Version Control) for dataset versioning (future)
- **Git LFS** for model checkpoints if needed
- Store datasets in cloud storage with download scripts

### Tracked vs Untracked

| Tracked | Not Tracked |
|---------|-------------|
| `src/` code | `data/` directories |
| `tests/` | `.env` files |
| `configs/` YAML | `models/` checkpoints |
| `docs/` | `.venv/` |
| Prompts in `src/generation/prompts/` | Cache files |
