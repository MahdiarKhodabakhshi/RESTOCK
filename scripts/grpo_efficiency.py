#!/usr/bin/env python3
"""GRPO efficiency-RL on 3.20: selective reads with a cost for schema tokens.

Starts from the memorised SFT weights. The model may emit ``<read>Type</read>``
tags before a ``<query>``. Reward is ``restock.reward.reward`` after executing
the query against the local Saleor 3.20 store.

    sbatch scripts/grpo_efficiency.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from graphql import build_schema
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from restock.agent.efficiency import chat_prompt, score_completion
from restock.agent.signal import version_signal
from restock.agent.tools import SchemaTools
from restock.reward import RewardConfig
from restock.saleor.client import SaleorClient
from restock.saleor.runtime import Runtime, api_url
from restock.schema.fetch import load_schema
from restock.tasks.cli import served_version
from restock.tasks.task import read_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/train.jsonl"))
    parser.add_argument("--model", default="data/models/memorised-3.20/merged")
    parser.add_argument("--out", type=Path, default=Path("data/models/efficiency-3.20"))
    parser.add_argument("--release", default="3.20")
    parser.add_argument("--schema-version", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("data"))
    return parser.parse_args()


def build_dataset(tasks_path: Path, version: str, limit: int | None) -> Dataset:
    rows = []
    for task in read_tasks(tasks_path):
        rows.append(
            {
                "prompt": chat_prompt(
                    signal=version_signal(version, trained_on=version).text(),
                    question=task.question,
                ),
                "expected": json.dumps(task.answer, ensure_ascii=False),
                "task_id": task.id,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    client = SaleorClient(api_url(args.release), Runtime(args.data).token(), timeout=120)
    served = served_version(client)
    schema_tag = args.schema_version or served
    schema = build_schema(load_schema(schema_tag))
    reward_config = RewardConfig()

    def efficiency_reward(completions, expected, **kwargs):
        scores = []
        for completion, expected_json in zip(completions, expected, strict=True):
            if isinstance(completion, list):
                # chat-style completion: last assistant message
                text = completion[-1]["content"] if completion else ""
            else:
                text = completion
            tools = SchemaTools(schema, version_signal(served, trained_on=args.release))
            try:
                answer = json.loads(expected_json)
                scores.append(score_completion(text, tools, client, answer, config=reward_config))
            except Exception:  # noqa: BLE001 - never break the trainer on one bad sample
                scores.append(0.0)
        return scores

    dataset = build_dataset(args.tasks, args.release, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    config = GRPOConfig(
        output_dir=str(args.out / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        temperature=0.7,
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=efficiency_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    adapter = args.out / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))

    merged = trainer.model.merge_and_unload()
    merged_dir = args.out / "merged"
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    client.close()
    print(f"adapter -> {adapter}")
    print(f"merged  -> {merged_dir}")


if __name__ == "__main__":
    main()
