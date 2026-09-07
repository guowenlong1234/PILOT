#!/usr/bin/env python3
"""Validate episode identity and compare RGB-only navigation evaluations.

Pass episode JSON files, never the already-averaged stats_ckpt JSON. Globs
allow merging distributed rank outputs; duplicate episodes are rejected.
Example:
  python scripts/summarize_rgb_only_optimization.py \
    --baseline '/results/base/stats_ep_*.json' \
    --candidate 'alpha025=/results/alpha025/stats_ep_*.json' \
    --expected-ids data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz \
    --output-json /results/comparison.json --output-md /results/comparison.md
"""

import argparse
import glob
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def expected_ids(path):
    if not path:
        return None
    payload = read_json(path)
    if isinstance(payload, dict) and "episodes" in payload:
        payload = [item["episode_id"] for item in payload["episodes"]]
    elif isinstance(payload, dict) and "episode_ids" in payload:
        payload = payload["episode_ids"]
    elif isinstance(payload, dict):
        payload = list(payload)
    if not isinstance(payload, list):
        raise ValueError("Expected IDs must be a list, dataset, or episode mapping")
    result = [str(item) for item in payload]
    if len(result) != len(set(result)):
        raise ValueError("Expected episode IDs contain duplicates")
    return set(result)


def load_episodes(pattern, expected_count, expected):
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError("No files matched: {}".format(pattern))
    episodes = {}
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Empty/invalid episode result: {}".format(path))
        for episode_id, metrics in payload.items():
            episode_id = str(episode_id)
            if episode_id in episodes:
                raise ValueError("Duplicate episode {} across files".format(episode_id))
            if not isinstance(metrics, dict):
                raise ValueError("Use stats_ep files, not aggregate stats: {}".format(path))
            for name in ("success", "spl", "path_length"):
                value = metrics.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Invalid {} for episode {}".format(name, episode_id))
            if metrics["success"] not in (0, 1) or not 0 <= metrics["spl"] <= 1 or metrics["path_length"] < 0:
                raise ValueError("Out-of-range navigation metrics for {}".format(episode_id))
            episodes[episode_id] = metrics
    if len(episodes) != expected_count:
        raise ValueError("Expected {} episodes, got {} in {}".format(expected_count, len(episodes), pattern))
    if expected is not None and set(episodes) != expected:
        missing = sorted(expected - set(episodes))
        extra = sorted(set(episodes) - expected)
        raise ValueError("Episode set mismatch: missing={} extra={}".format(missing[:20], extra[:20]))
    digest = hashlib.sha256("\n".join(sorted(episodes)).encode("utf-8")).hexdigest()
    metrics = {}
    common_keys = set.intersection(*(set(item) for item in episodes.values()))
    for name in sorted(common_keys):
        values = [item[name] for item in episodes.values()]
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            metrics[name] = statistics.mean(values)
    return episodes, {"files": files, "episodes": len(episodes), "episode_ids_sha256": digest, "metrics": metrics}


def paired_summary(baseline, candidate):
    if set(baseline) != set(candidate):
        raise ValueError("Candidate and baseline do not contain identical episode IDs")
    ids = sorted(baseline)
    rescued = [key for key in ids if baseline[key]["success"] == 0 and candidate[key]["success"] == 1]
    lost = [key for key in ids if baseline[key]["success"] == 1 and candidate[key]["success"] == 0]
    both = [key for key in ids if baseline[key]["success"] == candidate[key]["success"] == 1]
    discordant = len(rescued) + len(lost)
    # Exact paired binary test (two-sided binomial form of McNemar).
    p_value = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(len(rescued), len(lost)) + 1)) / (2 ** discordant)) if discordant else 1.0
    result = {
        "rescued": len(rescued), "lost": len(lost), "net_successes": len(rescued) - len(lost),
        "both_successful": len(both), "rescued_ids": rescued, "lost_ids": lost,
        "mcnemar_exact_p_unadjusted": p_value,
        "delta_sr_percentage_points": 100 * (len(rescued) - len(lost)) / len(ids),
        "delta_spl_percentage_points": 100 * statistics.mean(candidate[key]["spl"] - baseline[key]["spl"] for key in ids),
    }
    result["both_successful_path_length"] = None if not both else {
        "baseline_m": statistics.mean(baseline[key]["path_length"] for key in both),
        "candidate_m": statistics.mean(candidate[key]["path_length"] for key in both),
        "delta_m": statistics.mean(candidate[key]["path_length"] - baseline[key]["path_length"] for key in both),
        "longer_count": sum(candidate[key]["path_length"] > baseline[key]["path_length"] + 1e-6 for key in both),
    }
    return result


def markdown(report):
    baseline = report["baseline"]
    lines = [
        "# RGB-only 导航实验汇总", "",
        "所有结果均已验证为相同的 {} 个 episode。".format(baseline["episodes"]), "",
        "| 模型 | SR (%) | SPL (%) | ΔSR (百分点) | ΔSPL (百分点) | 救回 / 损失 | 共同成功路线 Δ长度 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| 基线 | {:.4f} | {:.4f} | 0 | 0 | — | — |".format(100 * baseline["metrics"]["success"], 100 * baseline["metrics"]["spl"]),
    ]
    for candidate in report["candidates"]:
        metrics, paired = candidate["metrics"], candidate["paired"]
        common = paired["both_successful_path_length"]
        delta_length = "—" if common is None else "{:+.4f}".format(common["delta_m"])
        lines.append("| {} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} | {} / {} | {} |".format(
            candidate["label"].replace("|", "\\|"), 100 * metrics["success"], 100 * metrics["spl"],
            paired["delta_sr_percentage_points"], paired["delta_spl_percentage_points"], paired["rescued"], paired["lost"], delta_length))
    lines.extend(["", "注意：此汇总只验证结果完整性和配对指标；配置、模型来源、运行环境和随机种子仍须与实验清单核对。多次挑选最佳点有选择偏差，JSON 中的配对检验 p 值未校正多重比较。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="LABEL=GLOB")
    parser.add_argument("--expected-count", type=int, default=1839)
    parser.add_argument("--expected-ids", help="Dataset .json.gz, JSON ID list, or reference episode result")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")
    expected = expected_ids(args.expected_ids)
    if expected is not None and len(expected) != args.expected_count:
        parser.error("--expected-count differs from expected ID count")
    baseline, baseline_summary = load_episodes(args.baseline, args.expected_count, expected)
    report = {"format": "rgb_only_paired_navigation_v1", "baseline": baseline_summary, "candidates": []}
    labels = set()
    for specification in args.candidate:
        if "=" not in specification:
            parser.error("--candidate requires LABEL=GLOB")
        label, pattern = specification.split("=", 1)
        if not label or label in labels:
            parser.error("Candidate labels must be nonempty and unique")
        labels.add(label)
        candidate, summary = load_episodes(pattern, args.expected_count, set(baseline))
        summary.update(label=label, paired=paired_summary(baseline, candidate))
        report["candidates"].append(summary)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rendered = markdown(report)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
