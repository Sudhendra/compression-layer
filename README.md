# LLM Compression Layer

Universal semantic compression layer for LLM inputs. Compresses memories, code, and context before API calls while preserving reasoning equivalence across Claude, GPT, and Gemini.

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Target Metrics

- **Token reduction**: 40-60%
- **Task equivalence**: >95%
- **Compression latency**: <50ms
- **Cross-model transfer**: >90%

## Project Structure

```
compression-layer/
├── src/
│   ├── validation/     # Cross-model equivalence testing
│   ├── generation/     # Compression pair generation
│   ├── training/       # Fine-tuning pipeline
│   ├── inference/      # Production compressor service
│   └── utils/          # Tokenizers, caching, cost tracking
├── data/               # Corpora and generated datasets
├── models/             # Checkpoints, adapters
├── configs/            # YAML configs
├── scripts/            # Entry points
└── tests/
```

## Environment Variables

Create a `.env` file with:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
HF_TOKEN=hf_...
TINKER_API_KEY=tk_...
```

## License

MIT
