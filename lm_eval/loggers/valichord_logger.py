"""ValiChord attestation logger for lm-evaluation-harness.

Builds a cryptographically verifiable bundle from an evaluation run using the
``valichord_attestation`` library.  The bundle contains:

- A stable SHA-256 commitment over the full evaluation (RFC 8785 / JCS
  canonical encoding + pre-rounding to 6 decimal places).
- A Merkle tree over per-sample ``filtered_resps``, enabling selective
  disclosure: a verifier can spot-check any subset of samples without the
  researcher releasing the full sample set.
- A probabilistic challenge-response protocol for auditing.

The bundle is saved as ``valichord_bundle_<date_id>.json`` alongside the
``results_*.json`` artifact written by ``EvaluationTracker``.

Usage from the CLI::

    lm_eval --model hf --tasks arc_easy --output_path ./results \\
        --log_samples --valichord_args output_path=./results

Programmatic usage::

    valichord_logger = ValiChordLogger({"output_path": "./results"})
    valichord_logger.post_init(results)
    valichord_logger.log_eval_result()           # metrics-only bundle
    valichord_logger.log_eval_samples(samples)   # full Merkle bundle
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _derive_model_id(results: dict[str, Any]) -> str:
    """Extract a ``source/name`` model identifier from the results dict."""
    source = results.get("model_source")
    name = results.get("model_name")
    if source and name:
        return f"{source}/{name}"
    config = results.get("config") or {}
    raw_source = config.get("model")
    raw_args = config.get("model_args")
    if raw_source and raw_args:
        for prefix in ("peft", "delta", "pretrained", "model", "path", "engine"):
            if isinstance(raw_args, dict):
                val = raw_args.get(prefix)
                if isinstance(val, str) and val.strip():
                    return f"{raw_source}/{val.strip()}"
            if isinstance(raw_args, str) and f"{prefix}=" in raw_args:
                tail = raw_args.split(f"{prefix}=", 1)[1]
                val = tail.split(",", 1)[0].strip()
                if val:
                    return f"{raw_source}/{val}"
        return str(raw_source)
    return source or name or "unknown"


def _extract_metrics(results: dict[str, Any]) -> list[dict]:
    """Return a list of raw metric dicts suitable for ``build_bundle``."""
    entries: list[dict] = []
    for task_id, task_metrics in (results.get("results") or {}).items():
        if not isinstance(task_metrics, dict):
            continue
        for key, value in task_metrics.items():
            metric, sep, filter_key = key.partition(",")
            if not sep or metric.endswith("_stderr"):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            stderr = task_metrics.get(f"{metric}_stderr,{filter_key}")
            entry: dict = {
                "key": f"{task_id}/{metric}",
                "value": float(value),
                "filter": filter_key,
            }
            if isinstance(stderr, (int, float)) and not isinstance(stderr, bool):
                entry["stderr"] = float(stderr)
            entries.append(entry)
    return entries


def _extract_samples(
    samples: dict[str, list[dict[str, Any]]],
    task_names: list[str],
) -> list[dict]:
    """Convert lm-eval per-task sample dicts to valichord sample dicts."""
    out: list[dict] = []
    for task_id in sorted(task_names):
        for s in samples.get(task_id) or []:
            out.append(
                {
                    "doc_id": s.get("doc_id"),
                    "target": str(s.get("target", "")),
                    "filtered_resps": s.get("filtered_resps"),
                    "filter": s.get("filter", "none"),
                    "task_id": task_id,
                }
            )
    return out


class ValiChordLogger:
    def __init__(self, init_args: dict[str, Any] | None = None) -> None:
        """Attaches a ValiChord attestation step to an lm-eval run.

        Builds a ``valichord_attestation`` bundle at the end of evaluation and
        saves it as ``valichord_bundle_<date_id>.json`` in ``output_path``.

        Args:
            init_args: dict with optional keys:
                ``output_path`` — directory to write the bundle (defaults to
                    the current directory if omitted).
                ``task_id`` — override the auto-derived task identifier.

        Usage from the CLI: ``--valichord_args output_path=./results``
        """
        try:
            import valichord_attestation as _va  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "valichord_attestation is not installed. "
                "Install it with `pip install valichord_attestation` "
                "or `pip install lm_eval[valichord]`."
            ) from exc

        args = dict(init_args or {})
        self._output_path: Path | None = (
            Path(args.pop("output_path")) if "output_path" in args else None
        )
        self._task_id_override: str | None = args.pop("task_id", None)
        self._results: dict[str, Any] = {}
        self._task_names: list[str] = []
        self._bundle_path: Path | None = None

    def post_init(self, results: dict[str, Any]) -> None:
        self._results = results
        self._task_names = list(results.get("results", {}).keys())

    def log_eval_result(self) -> None:
        """No-op: the attestation bundle requires per-sample data.

        The bundle is written in ``log_eval_samples`` once ``filtered_resps``
        are available.  This method exists to satisfy the logger protocol used
        by ``wandb``/``trackio`` loggers; it intentionally does nothing here.
        Pair with ``--log_samples`` to produce a bundle.
        """

    def log_eval_samples(self, samples: dict[str, list[dict[str, Any]]]) -> None:
        """Build and save the attestation bundle with the full per-sample Merkle tree."""
        self._save_bundle(samples=samples)

    def finish(self) -> None:
        pass

    def _save_bundle(
        self, samples: dict[str, list[dict[str, Any]]] | None
    ) -> None:
        from valichord_attestation import build_bundle, hash_bundle

        raw_metrics = _extract_metrics(self._results)
        if not raw_metrics:
            logger.warning("ValiChordLogger: no scalar metrics found — bundle skipped")
            return

        model_id = _derive_model_id(self._results)
        task_id = self._task_id_override or "+".join(sorted(self._task_names))

        sample_dicts: list[dict] = []
        if samples:
            sample_dicts = _extract_samples(samples, self._task_names)

        repo_commit: str | None = self._results.get("git_hash") or None
        generated_at: str | None = self._results.get("date") or None

        try:
            bundle = build_bundle(
                model_id=model_id,
                task_id=task_id,
                raw_metrics=raw_metrics,
                samples=sample_dicts,
                samples_total=len(sample_dicts),
                repo_commit=repo_commit,
                generated_at=generated_at,
            )
        except Exception:
            logger.exception("ValiChordLogger: build_bundle failed")
            return

        bundle_hash = hash_bundle(bundle)
        logger.info("ValiChord bundle hash: %s", bundle_hash)

        if self._output_path:
            try:
                date_id = datetime.now().isoformat().replace(":", "-")
                out_dir = self._output_path / str(
                    self._results.get("model_name_sanitized")
                    or "unknown_model"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                bundle_path = out_dir / f"valichord_bundle_{date_id}.json"

                from valichord_attestation import bundle_to_dict

                bundle_path.write_text(
                    json.dumps(bundle_to_dict(bundle), indent=2, ensure_ascii=False)
                    + "\n",
                    encoding="utf-8",
                )
                self._bundle_path = bundle_path
                logger.info("ValiChord bundle written to %s", bundle_path)
            except Exception:
                logger.exception("ValiChordLogger: could not write bundle to disk")
