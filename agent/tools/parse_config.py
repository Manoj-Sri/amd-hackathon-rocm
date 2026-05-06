"""parse_config tool — extract WorkloadConfig from a user's training script/args.

STUB IMPLEMENTATION — Phase 2 agent A replaces the body. Should:
  - Parse Python AST for HF TrainingArguments(...) calls; extract kwargs.
  - Parse JSON/YAML config files.
  - Run a regex redaction pass for tokens/keys/paths BEFORE returning.
  - Populate WorkloadConfig.redactions with labels of what was removed.

The current stub returns a deliberately-bad baseline config so the rest of
the loop can be smoke-tested end-to-end with FakeRunner.
"""

from __future__ import annotations

from agent.schemas import ToolResult, WorkloadConfig
from agent.tools import Tool


def _parse_config(file_path: str) -> ToolResult:  # noqa: ARG001 — stub
    cfg = WorkloadConfig(
        model_name="meta-llama/Meta-Llama-3-8B",
        batch_size=4,
        seq_len=2048,
        precision="fp16",
        attention_impl="eager",
        dataloader_workers=0,
        dataloader_pin_memory=False,
        lora_rank=16,
        gradient_checkpointing=False,
        torch_compile=False,
        raw_source="# STUB: parse_config not implemented yet",
        redactions=[],
    )
    return ToolResult(ok=True, result=cfg.model_dump())


PARSE_CONFIG = Tool(
    name="parse_config",
    description=(
        "Parse a user-uploaded training script or HF TrainingArguments JSON/YAML "
        "into a normalized WorkloadConfig. Redacts tokens, keys, and filesystem "
        "paths before returning."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the uploaded training script or config file.",
            }
        },
        "required": ["file_path"],
    },
    fn=_parse_config,
)
