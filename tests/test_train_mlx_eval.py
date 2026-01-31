from pathlib import Path
from types import SimpleNamespace

from src.training.train_mlx import evaluate_adapter


def test_evaluate_adapter_parses_ppl_without_colon(
    tmp_path: Path, monkeypatch: SimpleNamespace
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.jsonl").write_text("", encoding="utf-8")

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="Test loss 2.229, Test ppl 9.290.\n",
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    ppl = evaluate_adapter(
        "mlx-community/Qwen3-4B-Instruct-2507-8bit",
        tmp_path / "adapter",
        data_dir,
    )

    assert ppl == 9.29
