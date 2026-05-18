"""
Inspect-AI task: JEE Main 2026 Jan 24 (Morning) — Hindi.

Dataset: data/benchmark/jee_main_2026_hindi_benchmark.jsonl (74 rows).
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate, system_message

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .scorers import jee_scorer


DEFAULT_DATASET = "jee_main_2026_hindi_benchmark.jsonl"


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=build_user_prompt(record),
        target=str(record.get("answer", "")).strip(),
        id=record.get("id"),
        metadata={
            "exam": record.get("exam", "jee_main"),
            "year": record.get("year"),
            "paper": record.get("paper"),
            "subject": (record.get("subject") or "unknown").lower(),
            "question_type": record.get("question_type", "single_correct"),
            "has_diagram": bool(record.get("has_diagram")),
        },
    )


@task
def jee_main_2026(dataset_path: str = DEFAULT_DATASET) -> Task:
    """JEE Main 2026 Jan 24 morning, Hindi, 74 questions."""
    path = Path(dataset_path)
    if not path.is_absolute():
        # Resolve relative to current working dir
        path = Path.cwd() / path
    return Task(
        dataset=json_dataset(str(path), sample_fields=record_to_sample),
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=jee_scorer(),
    )
