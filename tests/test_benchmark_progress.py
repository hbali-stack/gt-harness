from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_progress import (
    aggregate,
    emit_harbor,
    emit_live,
    expected_task_ids,
    outcome_from_official,
)


def _official(task_id: str, reward: int | None, status: str = "GRADED") -> dict[str, object]:
    return {
        "schema": "gt.official_verifier_result.v1",
        "task_id": task_id,
        "status": status,
        "reward": reward,
        "failure_class": "graded" if status == "GRADED" else "setup_failure",
        "error_code": "" if status == "GRADED" else "runner_failed",
    }


@pytest.mark.parametrize(("reward", "state"), [(1, "passed"), (0, "verifier_failed")])
def test_only_official_rewards_create_graded_outcomes(reward: int, state: str) -> None:
    row = outcome_from_official(_official("task-a", reward), "task-a")
    assert row["state"] == state
    assert row["official_verifier"] is True
    assert row["reward"] == reward


def test_official_error_is_infrastructure_not_reward_zero() -> None:
    row = outcome_from_official(_official("task-a", None, "ERROR"), "task-a")
    assert row["state"] == "infrastructure_failed"
    assert row["official_verifier"] is False
    assert row["reward"] is None


def test_live_receipt_conserves_pass_fail_and_missing(tmp_path: Path) -> None:
    results = tmp_path / "official.json"
    results.write_text(json.dumps({"a": {"reward": 1}, "b": {"reward": 0}}), encoding="utf-8")
    receipt = emit_live("a,b,c", results)
    assert [row["state"] for row in receipt["tasks"]] == [
        "passed", "verifier_failed", "infrastructure_failed"
    ]


def test_harbor_uses_nested_official_verifier_reward(tmp_path: Path) -> None:
    trial = tmp_path / "job" / "task-a__trial" / "result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(json.dumps({
        "task_name": "terminal-bench/task-a",
        "trial_name": "task-a__trial",
        "verifier_result": {"rewards": {"reward": 0}},
        "exception_info": None,
    }), encoding="utf-8")
    receipt = emit_harbor(tmp_path, "task-a")
    assert receipt["tasks"][0]["state"] == "verifier_failed"
    assert receipt["tasks"][0]["official_verifier"] is True


def test_aggregate_conserves_all_expected_tasks(tmp_path: Path) -> None:
    first = {
        "schema": "gt.benchmark_progress.v1",
        "benchmark_suite": "suite",
        "tasks": [outcome_from_official(_official("a", 1), "a")],
    }
    path = tmp_path / "one" / "benchmark-progress.json"
    path.parent.mkdir()
    path.write_text(json.dumps(first), encoding="utf-8")
    snapshot = aggregate(tmp_path, ["a", "b"], finalize_missing=True)
    assert snapshot["total"] == 2
    assert snapshot["completed"] == 2
    assert snapshot["passed"] == 1
    assert snapshot["verifier_failed"] == 0
    assert snapshot["infrastructure_failed"] == 1
    assert snapshot["remaining"] == 0


def test_conflicting_duplicate_receipts_fail_closed(tmp_path: Path) -> None:
    for folder, reward in (("one", 1), ("two", 0)):
        payload = {
            "schema": "gt.benchmark_progress.v1",
            "benchmark_suite": "suite",
            "tasks": [outcome_from_official(_official("a", reward), "a")],
        }
        path = tmp_path / folder / "benchmark-progress.json"
        path.parent.mkdir()
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        aggregate(tmp_path, ["a"])


def test_expected_tasks_supports_each_workflow_matrix_shape() -> None:
    assert expected_task_ids('["a","b"]') == ["a", "b"]
    assert expected_task_ids('{"include":[{"task":"a"},{"task":"b"}]}') == ["a", "b"]
    assert expected_task_ids('{"include":[{"tasks":"a,b"},{"tasks":"c"}]}') == ["a", "b", "c"]


def test_tb2_workflow_publishes_harbor_verifier_receipts_to_live_monitor() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/tb2_miniswe_central.yml").read_text(encoding="utf-8")
    monitor = (root / ".github/workflows/task_progress.yml").read_text(encoding="utf-8")
    assert "scripts.benchmark_progress emit-harbor" in workflow
    assert "expected_tasks_json: ${{ needs.plan.outputs.tasks }}" in workflow
    assert "Passed (official reward 1)" in monitor
    assert "Successful task jobs" not in monitor
    assert workflow.index("Pull task image before agent execution on cache miss") < workflow.index("Run harbor - task")
    assert workflow.index("Require the task image before Harbor starts") < workflow.index("Run harbor - task")
