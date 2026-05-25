"""Orchestration tests for the fault-tolerant pipeline runner.

These use trivial in-process stages (no subprocess, no ML deps) to pin down the
behaviour that matters: finished work is skipped, a failure doesn't abort the
run, dependents of a failed stage are blocked rather than crashed, independent
stages still run, and a critical failure stops everything.
"""

from __future__ import annotations

from drc.pipeline import BLOCKED, DONE, FAILED, SKIPPED, Stage, run_pipeline


def _ok():
    return None


def _boom():
    raise RuntimeError("kaboom")


def test_failure_is_isolated_and_run_continues():
    """A non-critical failure is recorded; later independent stages still run."""
    stages = [
        Stage("a", _boom),
        Stage("b", _ok),  # no dependency on a — must still run
    ]
    results = run_pipeline(stages)
    assert results["a"].status == FAILED
    assert results["b"].status == DONE


def test_dependents_of_failure_are_blocked():
    """A stage requiring a failed stage is blocked, not executed."""
    ran = []
    stages = [
        Stage("a", _boom),
        Stage("b", lambda: ran.append("b"), requires=("a",)),
    ]
    results = run_pipeline(stages)
    assert results["a"].status == FAILED
    assert results["b"].status == BLOCKED
    assert "b" not in ran  # its action never fired


def test_finished_stage_is_skipped():
    """is_done short-circuits work — this is what makes a re-run resume."""
    ran = []
    stages = [Stage("a", lambda: ran.append("a"), is_done=lambda: True)]
    results = run_pipeline(stages)
    assert results["a"].status == SKIPPED
    assert ran == []


def test_skipped_dependency_still_satisfies_dependents():
    """A skipped (already-done) upstream counts as satisfied for downstream."""
    stages = [
        Stage("a", _ok, is_done=lambda: True),
        Stage("b", _ok, requires=("a",)),
    ]
    results = run_pipeline(stages)
    assert results["a"].status == SKIPPED
    assert results["b"].status == DONE


def test_critical_failure_aborts_remaining_stages():
    """A critical stage that fails stops the pipeline."""
    ran = []
    stages = [
        Stage("a", _boom, critical=True),
        Stage("b", lambda: ran.append("b")),
    ]
    results = run_pipeline(stages)
    assert results["a"].status == FAILED
    assert "b" not in results  # never reached
    assert ran == []


def test_force_reruns_a_completed_stage():
    ran = []
    stages = [Stage("a", lambda: ran.append("a"), is_done=lambda: True)]
    results = run_pipeline(stages, force={"a"})
    assert results["a"].status == DONE
    assert ran == ["a"]


def test_status_is_written(tmp_path):
    status = tmp_path / "status.json"
    run_pipeline([Stage("a", _ok)], status_path=status)
    assert status.exists()
    assert '"a"' in status.read_text()
