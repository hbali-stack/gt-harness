from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_tb2_treatment_requires_an_explicit_openrouter_model() -> None:
    source = _workflow("tb2_miniswe_central.yml")
    model_block = source.split("      model:\n", 1)[1].split("      provider:\n", 1)[0]

    assert "required: true" in model_block
    assert "\n        default:" not in model_block
    assert "options: [openrouter]" in source
    assert 'default: "https://openrouter.ai/api/v1"' in source
    assert "secrets.OPENROUTER_NEW || secrets.OPENROUTER_API_KEY" in source
    assert "DEEPSEEK_API_KEY" not in source
    assert "ANTHROPIC_API_KEY" not in source
    assert "GT_LITELLM_MODEL: ${{ inputs.model }}" in source


def test_tb2_control_uses_the_same_explicit_openrouter_contract() -> None:
    source = _workflow("tb2_miniswe_baseline_matrix.yml")
    model_block = source.split("      model:\n", 1)[1].split("      base_url:\n", 1)[0]

    assert "required: true" in model_block
    assert "\n        default:" not in model_block
    assert 'default: "https://openrouter.ai/api/v1"' in source
    assert "secrets.OPENROUTER_NEW || secrets.OPENROUTER_API_KEY" in source
    assert "provider_secret" not in source
    assert "inputs.api_key" not in source
    assert "DEEPSEEK_API_KEY" not in source
    assert "ANTHROPIC_API_KEY" not in source
