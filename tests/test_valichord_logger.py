"""Tests for lm_eval/loggers/valichord_logger.py.

All tests mock valichord_attestation so the dependency is not required
in the test environment.  Integration behaviour (actual bundle hash
stability) is validated in valichord_attestation's own test suite.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build fixture data matching lm-eval's results dict shape
# ---------------------------------------------------------------------------

def _make_results(
    model_source: str = "hf",
    model_name: str = "EleutherAI/pythia-70m",
    tasks: dict[str, dict] | None = None,
    git_hash: str | None = "abc123",
    date: str | None = "2024-01-01T00:00:00",
) -> dict[str, Any]:
    if tasks is None:
        tasks = {
            "arc_easy": {
                "acc,none": 0.5,
                "acc_stderr,none": 0.1,
                "acc_norm,none": 0.75,
                "acc_norm_stderr,none": 0.05,
            }
        }
    return {
        "model_source": model_source,
        "model_name": model_name,
        "model_name_sanitized": model_name.replace("/", "__"),
        "results": tasks,
        "configs": {t: {"output_type": "multiple_choice"} for t in tasks},
        "git_hash": git_hash,
        "date": date,
    }


def _make_samples(task_names: list[str] | None = None) -> dict[str, list[dict]]:
    if task_names is None:
        task_names = ["arc_easy"]
    out: dict[str, list[dict]] = {}
    for task in task_names:
        out[task] = [
            {
                "doc_id": i,
                "target": str(i % 4),
                "filtered_resps": [str(i % 4)],
                "filter": "none",
                "arguments": {"arg_0": {"arg_0": f"question {i}"}},
                "resps": [[0.9, 0.1, 0.05, 0.05]],
                "metrics": ["acc"],
                "doc_hash": "a" * 64,
                "prompt_hash": "b" * 64,
                "target_hash": "c" * 64,
            }
            for i in range(3)
        ]
    return out


# ---------------------------------------------------------------------------
# Fake valichord_attestation module injected for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_valichord(monkeypatch):
    """Inject a lightweight fake valichord_attestation into sys.modules."""
    mod = types.ModuleType("valichord_attestation")

    class FakeBundle:
        pass

    def fake_build_bundle(**kwargs):
        b = FakeBundle()
        b._kwargs = kwargs
        return b

    def fake_hash_bundle(bundle):
        return "a" * 64

    def fake_bundle_to_dict(bundle):
        return {"fake": True, "model_id": bundle._kwargs.get("model_id")}

    mod.build_bundle = fake_build_bundle
    mod.hash_bundle = fake_hash_bundle
    mod.bundle_to_dict = fake_bundle_to_dict

    monkeypatch.setitem(sys.modules, "valichord_attestation", mod)
    yield mod


@pytest.fixture()
def logger_cls():
    # Re-import after fake is in sys.modules
    from lm_eval.loggers.valichord_logger import ValiChordLogger
    return ValiChordLogger


# ---------------------------------------------------------------------------
# _derive_model_id
# ---------------------------------------------------------------------------

def test_derive_model_id_top_level_fields():
    from lm_eval.loggers.valichord_logger import _derive_model_id
    assert _derive_model_id({"model_source": "hf", "model_name": "foo/bar"}) == "hf/foo/bar"


def test_derive_model_id_config_string_fallback():
    from lm_eval.loggers.valichord_logger import _derive_model_id
    doc = {"config": {"model": "vllm", "model_args": "pretrained=foo/bar,dtype=auto"}}
    assert _derive_model_id(doc) == "vllm/foo/bar"


def test_derive_model_id_config_dict_fallback():
    from lm_eval.loggers.valichord_logger import _derive_model_id
    doc = {"config": {"model": "vllm", "model_args": {"pretrained": "foo/bar"}}}
    assert _derive_model_id(doc) == "vllm/foo/bar"


def test_derive_model_id_unknown_when_empty():
    from lm_eval.loggers.valichord_logger import _derive_model_id
    assert _derive_model_id({}) == "unknown"


def test_derive_model_id_peft_takes_precedence():
    from lm_eval.loggers.valichord_logger import _derive_model_id
    doc = {"config": {"model": "hf", "model_args": "pretrained=base/m,peft=adapter/lora"}}
    assert _derive_model_id(doc) == "hf/adapter/lora"


# ---------------------------------------------------------------------------
# _extract_metrics
# ---------------------------------------------------------------------------

def test_extract_metrics_basic():
    from lm_eval.loggers.valichord_logger import _extract_metrics
    results = _make_results()
    metrics = _extract_metrics(results)
    keys = {(m["key"], m["filter"]) for m in metrics}
    assert ("arc_easy/acc", "none") in keys
    assert ("arc_easy/acc_norm", "none") in keys


def test_extract_metrics_skips_stderr_keys():
    from lm_eval.loggers.valichord_logger import _extract_metrics
    results = _make_results()
    metrics = _extract_metrics(results)
    assert all("stderr" not in m["key"] for m in metrics)


def test_extract_metrics_skips_non_numeric():
    from lm_eval.loggers.valichord_logger import _extract_metrics
    results = _make_results(tasks={"t": {"acc,none": "N/A", "real,none": 0.5}})
    metrics = _extract_metrics(results)
    assert len(metrics) == 1
    assert metrics[0]["key"] == "t/real"


def test_extract_metrics_skips_bool():
    from lm_eval.loggers.valichord_logger import _extract_metrics
    results = _make_results(tasks={"t": {"flag,none": True, "acc,none": 0.9}})
    metrics = _extract_metrics(results)
    assert all(m["key"] != "t/flag" for m in metrics)


def test_extract_metrics_stderr_attached():
    from lm_eval.loggers.valichord_logger import _extract_metrics
    results = _make_results()
    metrics = _extract_metrics(results)
    acc = next(m for m in metrics if m["key"] == "arc_easy/acc")
    assert acc.get("stderr") == pytest.approx(0.1)


def test_extract_metrics_multiple_tasks():
    from lm_eval.loggers.valichord_logger import _extract_metrics
    results = _make_results(tasks={
        "arc_easy": {"acc,none": 0.5, "acc_stderr,none": 0.1},
        "hellaswag": {"acc,none": 0.6, "acc_stderr,none": 0.05},
    })
    metrics = _extract_metrics(results)
    task_ids = {m["key"].split("/")[0] for m in metrics}
    assert task_ids == {"arc_easy", "hellaswag"}


# ---------------------------------------------------------------------------
# _extract_samples
# ---------------------------------------------------------------------------

def test_extract_samples_basic():
    from lm_eval.loggers.valichord_logger import _extract_samples
    samples = _make_samples()
    out = _extract_samples(samples, ["arc_easy"])
    assert len(out) == 3
    assert all(s["task_id"] == "arc_easy" for s in out)
    assert all("doc_id" in s for s in out)
    assert all("filtered_resps" in s for s in out)


def test_extract_samples_sorted_by_task():
    from lm_eval.loggers.valichord_logger import _extract_samples
    samples = _make_samples(["hellaswag", "arc_easy"])
    out = _extract_samples(samples, ["hellaswag", "arc_easy"])
    task_ids = [s["task_id"] for s in out]
    # arc_easy sorts before hellaswag
    assert task_ids[:3] == ["arc_easy", "arc_easy", "arc_easy"]
    assert task_ids[3:] == ["hellaswag", "hellaswag", "hellaswag"]


def test_extract_samples_unknown_task_skipped():
    from lm_eval.loggers.valichord_logger import _extract_samples
    samples = _make_samples(["arc_easy"])
    out = _extract_samples(samples, ["missing_task"])
    assert out == []


# ---------------------------------------------------------------------------
# ValiChordLogger lifecycle
# ---------------------------------------------------------------------------

def test_init_raises_without_valichord(monkeypatch):
    monkeypatch.delitem(sys.modules, "valichord_attestation", raising=False)
    # Force ImportError by temporarily removing the module
    with patch.dict(sys.modules, {"valichord_attestation": None}):
        from importlib import reload
        import lm_eval.loggers.valichord_logger as mod
        reload(mod)
        with pytest.raises(ImportError, match="valichord_attestation"):
            mod.ValiChordLogger({})


def test_post_init_stores_task_names(logger_cls):
    lg = logger_cls({})
    lg.post_init(_make_results())
    assert lg._task_names == ["arc_easy"]


def test_log_eval_result_is_noop(logger_cls, fake_valichord):
    calls = []
    original = fake_valichord.build_bundle
    fake_valichord.build_bundle = lambda **kw: (calls.append(kw), original(**kw))[1]

    lg = logger_cls({})
    lg.post_init(_make_results())
    lg.log_eval_result()  # bundle requires samples; should be a no-op

    assert len(calls) == 0


def test_log_eval_samples_builds_bundle_with_samples(logger_cls, fake_valichord):
    calls = []
    original = fake_valichord.build_bundle
    fake_valichord.build_bundle = lambda **kw: (calls.append(kw), original(**kw))[1]

    lg = logger_cls({})
    lg.post_init(_make_results())
    lg.log_eval_samples(_make_samples())

    assert len(calls) == 1
    assert len(calls[0]["samples"]) == 3
    assert calls[0]["samples_total"] == 3


def test_task_id_multi_task_sorted(logger_cls, fake_valichord):
    calls = []
    original = fake_valichord.build_bundle
    fake_valichord.build_bundle = lambda **kw: (calls.append(kw), original(**kw))[1]

    results = _make_results(tasks={
        "hellaswag": {"acc,none": 0.6, "acc_stderr,none": 0.05},
        "arc_easy": {"acc,none": 0.5, "acc_stderr,none": 0.1},
    })
    lg = logger_cls({})
    lg.post_init(results)
    lg.log_eval_samples(_make_samples(["arc_easy", "hellaswag"]))

    assert calls[0]["task_id"] == "arc_easy+hellaswag"


def test_task_id_override(logger_cls, fake_valichord):
    calls = []
    original = fake_valichord.build_bundle
    fake_valichord.build_bundle = lambda **kw: (calls.append(kw), original(**kw))[1]

    lg = logger_cls({"task_id": "my-custom-task"})
    lg.post_init(_make_results())
    lg.log_eval_samples(_make_samples())

    assert calls[0]["task_id"] == "my-custom-task"


def test_repo_commit_from_git_hash(logger_cls, fake_valichord):
    calls = []
    original = fake_valichord.build_bundle
    fake_valichord.build_bundle = lambda **kw: (calls.append(kw), original(**kw))[1]

    lg = logger_cls({})
    lg.post_init(_make_results(git_hash="deadbeef"))
    lg.log_eval_samples(_make_samples())

    assert calls[0]["repo_commit"] == "deadbeef"


def test_repo_commit_none_when_no_git_hash(logger_cls, fake_valichord):
    calls = []
    original = fake_valichord.build_bundle
    fake_valichord.build_bundle = lambda **kw: (calls.append(kw), original(**kw))[1]

    lg = logger_cls({})
    lg.post_init(_make_results(git_hash=None))
    lg.log_eval_samples(_make_samples())

    assert calls[0]["repo_commit"] is None


def test_no_metrics_skips_bundle(logger_cls, fake_valichord):
    calls = []
    original = fake_valichord.build_bundle
    fake_valichord.build_bundle = lambda **kw: (calls.append(kw), original(**kw))[1]

    results = _make_results(tasks={"t": {"not_a_metric": "string_value"}})
    lg = logger_cls({})
    lg.post_init(results)
    lg.log_eval_samples(_make_samples(["t"]))

    assert len(calls) == 0


def test_bundle_written_to_output_path(logger_cls, tmp_path):
    lg = logger_cls({"output_path": str(tmp_path)})
    lg.post_init(_make_results())
    lg.log_eval_samples(_make_samples())

    model_dir = tmp_path / "EleutherAI__pythia-70m"
    bundles = list(model_dir.glob("valichord_bundle_*.json"))
    assert len(bundles) == 1
    data = json.loads(bundles[0].read_text())
    assert data["fake"] is True


def test_log_eval_result_writes_no_bundle(logger_cls, tmp_path):
    lg = logger_cls({"output_path": str(tmp_path)})
    lg.post_init(_make_results())
    lg.log_eval_result()  # no-op — samples required

    model_dir = tmp_path / "EleutherAI__pythia-70m"
    bundles = list(model_dir.glob("valichord_bundle_*.json"))
    assert len(bundles) == 0


def test_no_output_path_does_not_write(logger_cls, tmp_path):
    lg = logger_cls({})
    lg.post_init(_make_results())
    lg.log_eval_samples(_make_samples())
    assert lg._bundle_path is None


def test_finish_is_noop(logger_cls):
    lg = logger_cls({})
    lg.finish()  # should not raise


def test_build_bundle_exception_does_not_propagate(logger_cls, fake_valichord):
    fake_valichord.build_bundle = MagicMock(side_effect=RuntimeError("boom"))
    lg = logger_cls({})
    lg.post_init(_make_results())
    lg.log_eval_samples(_make_samples())  # must not raise
