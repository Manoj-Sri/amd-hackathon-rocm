"""Tests for ``agent.tools.parse_config._parse_config``.

Covers all three input shapes (Python AST, JSON, YAML), redaction of every
secret pattern we ship, error paths for missing/malformed inputs, and the
WorkloadConfig field mapping for HF TrainingArguments + DataLoader kwargs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.schemas import WorkloadConfig
from agent.tools.parse_config import PARSE_CONFIG, _parse_config, _parse_config_full

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Python script path
# ---------------------------------------------------------------------------


class TestPythonScript:
    def test_returns_ok(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        assert result.ok, result.error
        assert result.result is not None

    def test_extracts_training_arguments_kwargs(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        cfg = result.result
        assert cfg["batch_size"] == 4
        assert cfg["grad_accum_steps"] == 8
        assert cfg["lr"] == pytest.approx(2e-4)
        assert cfg["warmup_steps"] == 100
        assert cfg["optimizer"] == "adamw_torch"
        # fp16=True in TrainingArguments → precision should resolve to fp16.
        assert cfg["precision"] == "fp16"

    def test_dataloader_kwargs_captured(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        cfg = result.result
        assert cfg["dataloader_workers"] == 0
        assert cfg["dataloader_pin_memory"] is False
        assert cfg["dataloader_prefetch_factor"] == 2
        assert cfg["dataloader_persistent_workers"] is False

    def test_torch_compile_call_flips_flag(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        # The script calls torch.compile(model, ...), even though the
        # TrainingArguments has torch_compile=False — the explicit call wins.
        assert result.result["torch_compile"] is True

    def test_gradient_checkpointing_enable_call(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        assert result.result["gradient_checkpointing"] is True

    def test_env_vars_captured(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        env = result.result["env_vars"]
        assert env["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"
        assert env["MIOPEN_FIND_MODE"] == "3"
        assert env["NCCL_MIN_NCHANNELS"] == "112"

    def test_lora_rank_extracted(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        assert result.result["lora_rank"] == 16

    def test_attention_impl_from_from_pretrained(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        assert result.result["attention_impl"] == "eager"

    def test_model_name_resolved(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        assert result.result["model_name"] == "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# JSON path
# ---------------------------------------------------------------------------


class TestJsonConfig:
    def test_returns_ok(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.json"))
        assert result.ok, result.error

    def test_field_mapping(self) -> None:
        cfg = _parse_config(str(FIXTURES / "sample_train.json")).result
        assert cfg["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg["batch_size"] == 8
        assert cfg["grad_accum_steps"] == 4
        assert cfg["seq_len"] == 4096
        assert cfg["precision"] == "bf16"
        assert cfg["optimizer"] == "adamw_torch_fused"
        assert cfg["torch_compile"] is True
        assert cfg["gradient_checkpointing"] is True
        assert cfg["dataloader_workers"] == 4
        assert cfg["dataloader_pin_memory"] is True
        assert cfg["dataloader_persistent_workers"] is True
        assert cfg["attention_impl"] == "flash"

    def test_env_vars_dict(self) -> None:
        cfg = _parse_config(str(FIXTURES / "sample_train.json")).result
        assert cfg["env_vars"]["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"

    def test_extras_collects_unmapped_fields(self) -> None:
        cfg = _parse_config(str(FIXTURES / "sample_train.json")).result
        # num_train_epochs has no slot in WorkloadConfig — must land in extras.
        assert cfg["extras"]["num_train_epochs"] == 3
        assert cfg["extras"]["save_steps"] == 500


# ---------------------------------------------------------------------------
# YAML path
# ---------------------------------------------------------------------------


class TestYamlConfig:
    def test_returns_ok(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.yaml"))
        assert result.ok, result.error

    def test_field_mapping(self) -> None:
        cfg = _parse_config(str(FIXTURES / "sample_train.yaml")).result
        assert cfg["batch_size"] == 2
        assert cfg["grad_accum_steps"] == 16
        assert cfg["seq_len"] == 8192
        assert cfg["precision"] == "bf16"
        assert cfg["torch_compile"] is False
        assert cfg["gradient_checkpointing"] is True
        assert cfg["attention_impl"] == "sdpa"
        assert cfg["dataloader_workers"] == 8
        assert cfg["dataloader_persistent_workers"] is True


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_python_script_redactions(self) -> None:
        cfg = _parse_config(str(FIXTURES / "sample_train.py")).result
        labels = set(cfg["redactions"])
        # Every secret pattern in sample_train.py should fire.
        assert "hf_token" in labels
        assert "openai_key" in labels
        assert "github_token" in labels
        assert "bearer_token" in labels
        assert "home_path" in labels
        assert "s3_uri" in labels
        assert "ws_uri" in labels

    def test_raw_source_is_scrubbed(self) -> None:
        # `raw_source` is intentionally stripped from the tool result envelope
        # (keeps the LLM conversation small) — use the `_full` helper to read
        # it. The redaction labels list still proves which patterns fired.
        cfg = _parse_config_full(str(FIXTURES / "sample_train.py"))
        assert isinstance(cfg, WorkloadConfig)
        raw = cfg.raw_source
        assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in raw
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in raw
        assert "gho_abcdefghijklmnopqrstuvwxyz123456" not in raw
        assert "/home/researcher/datasets/alpaca" not in raw
        assert "s3://my-team/checkpoints/qwen-lora/" not in raw
        assert "wss://logs.internal.example.com/stream" not in raw
        assert "<REDACTED:hf_token>" in raw
        assert "<REDACTED:openai_key>" in raw

    def test_raw_source_excluded_from_tool_result(self) -> None:
        # The tool result MUST NOT carry raw_source — it bloated the audit
        # conversation past 8K on Qwen2.5-7B during the live AMD GPU run.
        cfg = _parse_config(str(FIXTURES / "sample_train.py")).result
        assert "raw_source" not in cfg

    def test_json_redactions(self) -> None:
        cfg = _parse_config(str(FIXTURES / "sample_train.json")).result
        labels = set(cfg["redactions"])
        assert "hf_token" in labels
        assert "s3_uri" in labels
        # raw_source is no longer in the result; verify scrubbing via the
        # full-config helper.
        full = _parse_config_full(str(FIXTURES / "sample_train.json"))
        assert isinstance(full, WorkloadConfig)
        assert "hf_jsonsamplehfabcdefghijklmnopqrs" not in full.raw_source

    def test_extras_values_are_scrubbed(self) -> None:
        # Secret-shaped values that landed in extras must also be redacted —
        # otherwise the leak just moves from raw_source into extras.
        cfg = _parse_config(str(FIXTURES / "sample_train.json")).result
        extras = cfg["extras"]
        assert "hf_jsonsamplehfabcdefghijklmnopqrs" not in extras.get("hub_token", "")
        assert extras.get("hub_token", "").startswith("<REDACTED:")
        assert extras.get("checkpoint_uri", "").startswith("<REDACTED:")

    def test_yaml_redactions(self) -> None:
        cfg = _parse_config(str(FIXTURES / "sample_train.yaml")).result
        labels = set(cfg["redactions"])
        assert "hf_token" in labels
        assert "bearer_token" in labels
        assert "home_path" in labels


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_file(self) -> None:
        result = _parse_config("/tmp/definitely-does-not-exist-xyz.py")
        assert result.ok is False
        assert "not found" in (result.error or "").lower()

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.toml"
        bad.write_text("model_name = 'foo'\n")
        result = _parse_config(str(bad))
        assert result.ok is False
        assert "unsupported" in (result.error or "").lower()

    def test_malformed_python(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.py"
        bad.write_text("def oops(:\n  pass\n")
        result = _parse_config(str(bad))
        assert result.ok is False
        assert "parse error" in (result.error or "").lower()

    def test_malformed_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.json"
        bad.write_text("{not really json")
        result = _parse_config(str(bad))
        assert result.ok is False
        assert "json" in (result.error or "").lower()

    def test_json_top_level_must_be_dict(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.json"
        bad.write_text(json.dumps([{"foo": 1}]))
        result = _parse_config(str(bad))
        assert result.ok is False

    def test_yaml_top_level_must_be_mapping(self, tmp_path: Path) -> None:
        bad = tmp_path / "scalar.yaml"
        bad.write_text("- 1\n- 2\n")
        result = _parse_config(str(bad))
        assert result.ok is False


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


class TestSchema:
    def test_result_round_trips_through_workload_config(self) -> None:
        result = _parse_config(str(FIXTURES / "sample_train.py"))
        # Must be reconstructible — guards against extras-vs-fields collisions.
        cfg = WorkloadConfig(**result.result)
        assert cfg.model_name == "Qwen/Qwen2.5-7B-Instruct"

    def test_defaults_when_field_absent(self, tmp_path: Path) -> None:
        # Minimal config — only model_name. Everything else should fall back to schema defaults.
        path = tmp_path / "tiny.json"
        path.write_text(json.dumps({"model_name": "test/tiny"}))
        result = _parse_config(str(path))
        assert result.ok
        cfg = result.result
        assert cfg["batch_size"] == 1
        assert cfg["precision"] == "fp16"
        assert cfg["optimizer"] == "adamw_torch"
        assert cfg["gradient_checkpointing"] is False
        assert cfg["redactions"] == []

    def test_tool_definition_unchanged_in_shape(self) -> None:
        # The Tool definition should still expose name/description/input_schema/fn.
        assert PARSE_CONFIG.name == "parse_config"
        assert PARSE_CONFIG.fn is _parse_config
        assert "file_path" in PARSE_CONFIG.input_schema["properties"]


# ---------------------------------------------------------------------------
# Regression: canonical + scenario workloads must parse with all the right
# audit-relevant fields. These are what the live agent actually sees, so a
# regression here directly degrades audit quality (the agent reasons over
# HF defaults instead of the script's settings).
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCanonicalWorkload:
    def test_canonical_workload_extracts_full_config(self) -> None:
        """The canonical demo workload must yield batch_size=4, lr=2e-4, etc.
        — not HF defaults. Catches the `**dict_var` splat regression where
        every TrainingArguments kwarg disappears.
        """
        result = _parse_config(str(REPO_ROOT / "workloads" / "train_qwen_lora.py"))
        assert result.ok, result.error
        cfg = result.result
        assert cfg["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg["batch_size"] == 4, (
            "expected batch_size=4 from per_device_train_batch_size; "
            "did `**_ta_kwargs` splat hide the kwargs?"
        )
        assert cfg["grad_accum_steps"] == 8
        assert cfg["lr"] == 2e-4
        assert cfg["warmup_steps"] == 100
        assert cfg["precision"] == "fp16"
        assert cfg["attention_impl"] == "eager"
        assert cfg["dataloader_workers"] == 0
        assert cfg["dataloader_pin_memory"] is False
        assert cfg["lora_rank"] == 16
        assert cfg["torch_compile"] is False
        assert cfg["env_vars"]["HSA_FORCE_FINE_GRAIN_PCIE"] == "1"


class TestSplatKwargsResolution:
    """`_ta = dict(k=v); Foo(**_ta)` must resolve back through the dict
    constant. Defensive — the canonical workload no longer uses this
    pattern, but third-party scripts often do.
    """

    def test_dict_function_call_splat(self, tmp_path) -> None:
        src = """
from transformers import TrainingArguments

_ta = dict(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,
    optim=\"adamw_torch_fused\",
)
training_args = TrainingArguments(output_dir=\"./out\", **_ta)
"""
        p = tmp_path / "splat.py"
        p.write_text(src)
        cfg = _parse_config(str(p)).result
        assert cfg["batch_size"] == 8
        assert cfg["grad_accum_steps"] == 2
        assert cfg["precision"] == "fp16"
        assert cfg["optimizer"] == "adamw_torch_fused"

    def test_dict_literal_splat(self, tmp_path) -> None:
        src = """
from transformers import TrainingArguments

_ta = {
    "per_device_train_batch_size": 16,
    "bf16": True,
}
training_args = TrainingArguments(output_dir=\"./out\", **_ta)
"""
        p = tmp_path / "splat_literal.py"
        p.write_text(src)
        cfg = _parse_config(str(p)).result
        assert cfg["batch_size"] == 16
        assert cfg["precision"] == "bf16"

    def test_explicit_kwarg_overrides_splat(self, tmp_path) -> None:
        src = """
from transformers import TrainingArguments

_ta = dict(per_device_train_batch_size=8)
training_args = TrainingArguments(per_device_train_batch_size=32, **_ta)
"""
        p = tmp_path / "splat_override.py"
        p.write_text(src)
        cfg = _parse_config(str(p)).result
        # Explicit kwarg wins over splat (both occur in the kwargs list,
        # explicit comes first in the AST → setdefault keeps it).
        assert cfg["batch_size"] == 32

