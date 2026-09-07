#!/usr/bin/env python3
"""CPU-only final audit of the prescribed 12 evaluations and three trainings.

Run inside gwl-etpr1-rae using etpr1_rae. The server completion publisher
atomically supplies --train-manifest-dir/{A,B,C}.json and four frozen audits.
This tool does not fetch files, run models, alter checkpoints, or stop jobs.
Missing/running manifests are pending (exit 3, or poll with --wait); a failed
manifest or invalid completed artifact produces a failed JSON and exit 2.
Only all-complete, validated artifacts produce the final scientific report.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import time

from summarize_rgb_only_optimization import expected_ids, load_episodes, paired_summary, read_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data/logs/rgb_only_optimization_20260907"
DATASET = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz"
BASE_SHA = "1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61"
HISTORICAL_SR_PERCENT = 63.7303


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, payload):
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_text(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def cases():
    result = []
    for case in ("base", "legacy", "a0", "a025", "a05", "a1", "A", "B", "C"):
        for iteration in ([1000, 2000] if case in ("A", "B", "C") else [None]):
            source = "query_current" if case == "legacy" else "context_last"
            alpha = {"a0": 0.0, "a025": 0.25, "a05": 0.5}.get(case, 1.0)
            frozen, aligned = case in ("B", "C"), case == "C"
            name = f"rgbopt_{case}_alpha{alpha:g}_freeze{int(frozen)}_align{int(aligned)}_{source}"
            if iteration is not None:
                name += f"_iter{iteration}"
            result.append(dict(case=case, label=case + (f"_{iteration}" if iteration else ""),
                               name=name, iteration=iteration, alpha=alpha, source=source,
                               frozen=frozen, aligned=aligned, enabled=case not in ("base", "A")))
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def manifest_inventory(args):
    records, pending = {}, []
    publisher_path = args.train_manifest_dir / "publisher_status.json"
    if publisher_path.is_file():
        publisher = read_json(publisher_path)
        require(publisher.get("status") not in ("failed", "timed_out"),
                f"Training completion publisher failed: {publisher_path}; {publisher.get('errors')}")
    paths = [("eval/" + item["label"], args.root / "eval" / item["name"] / "manifest.json")
             for item in cases()]
    paths += [("train/" + branch, args.train_manifest_dir / (branch + ".json"))
              for branch in "ABC"]
    for key, path in paths:
        if not path.is_file():
            pending.append(dict(job=key, reason="missing_manifest", path=str(path)))
            continue
        payload = read_json(path)
        status = payload.get("status")
        require(status in ("running", "completed", "failed"), f"Invalid manifest status: {path}")
        require(status != "failed", f"Experiment failed: {path}; exit_code={payload.get('exit_code')}; "
                f"validation_error={payload.get('validation_error')}")
        if status == "running":
            pending.append(dict(job=key, reason="running", path=str(path)))
            continue
        require(payload.get("exit_code") == 0, f"Completed manifest has nonzero/missing exit code: {path}")
        records[key] = dict(path=str(path), payload=payload)
    for branch in "BC":
        for iteration in (1000, 2000):
            key = f"frozen/{branch}_{iteration}"
            path = args.train_manifest_dir / f"{branch}_{iteration}_frozen.json"
            if not path.is_file():
                pending.append(dict(job=key, reason="missing_frozen_audit", path=str(path)))
                continue
            payload = read_json(path)
            require(payload.get("ok") is True, f"Frozen checkpoint audit failed: {path}; {payload.get('errors')}")
            records[key] = dict(path=str(path), payload=payload)
    return records, pending


def override(record, key, expected):
    payload = record["payload"]
    opts, cmd = payload.get("overrides", {}), payload.get("command", [])
    require(key in opts and opts[key] == expected,
            f"Configuration mismatch {record['path']}: {key}={opts.get(key)!r}, expected {expected!r}")
    require(cmd.count(key) == 1 and cmd[cmd.index(key) + 1] == str(opts[key]),
            f"Command/overrides mismatch: {record['path']} {key}")


def source_diagnostics(result_dir, item):
    paths = list(result_dir.glob("source_pose_ckpt_*_val_unseen_r0_w1.json"))
    if not item["enabled"]:
        require(not paths, f"Unexpected world-model query diagnostics for {item['label']}")
        return None
    require(len(paths) == 1, f"Missing/ambiguous source diagnostics for {item['label']}")
    data = read_json(paths[0])
    require(data["condition_source_pose"] == item["source"], "Source diagnostic contract mismatch")
    totals = data["totals"]
    count = totals["queries"]
    require(isinstance(count, int) and count > 0, "RGB-enabled evaluation had no contextual queries")
    rates = {}
    for key in ("mismatch_queries", "yaw_mismatch_queries", "position_mismatch_queries"):
        value = totals[key]
        require(isinstance(value, int) and 0 <= value <= count, f"Invalid counter {key}")
        rates[key + "_fraction"] = value / count
    for key in ("yaw_abs_sum_rad", "position_sum_m", "max_yaw_rad", "max_position_m"):
        require(isinstance(totals[key], (int, float)) and math.isfinite(totals[key]) and totals[key] >= 0,
                "Invalid source pose statistic " + key)
    return dict(path=str(paths[0]), totals=totals, rates=rates,
                mean_abs_yaw_degrees=math.degrees(totals["yaw_abs_sum_rad"] / count),
                mean_position_gap_m=totals["position_sum_m"] / count,
                denominator="具备4帧上下文并进入条件组批的候选查询；可能仍在后续步骤被过滤。",
                interpretation="查询时位姿与最后上下文帧位姿的差异；不是episode占比，也不是修复后残余条件错误率。")


def audit_training(records):
    training = {}
    for branch in "ABC":
        record = records["train/" + branch]
        checks = {"MODEL.ACTIVE_LOOKAHEAD.enabled": False, "TASK_CONFIG.SEED": 100,
                  "IL.iters": 2000, "IL.log_every": 200, "IL.lr": 1e-5,
                  "IL.sample_ratio": 0.75, "IL.decay_interval": 3000,
                  "IL.sample_ratio_iteration_offset": 14200,
                  "IL.freeze_navigation_backbone": branch != "A",
                  "MODEL.RAENWM.enabled": branch != "A",
                  "MODEL.RAENWM.rgb_fusion_enabled": branch != "A",
                  "MODEL.RAENWM.rgb_fusion_align_navigation_cls": branch == "C",
                  "MODEL.RAENWM.condition_source_pose": "context_last"}
        for key, expected in checks.items():
            override(record, key, expected)
        payload = record["payload"]
        opts = payload["overrides"]
        batch = opts["GPU_NUMBERS"] * opts["IL.batch_size"] * opts["IL.gradient_accumulation_steps"]
        require(batch == 16, "Training global batch must be 16")
        require(payload["checkpoint_sha256"] == BASE_SHA, "Training did not start from approved base")
        validation = payload.get("validation", {})
        require(validation.get("iteration") == 2000 and validation.get("model_bytes", 0) > 0,
                "Training final checkpoint/state pair was not validated")
        require(validation.get("model") and validation.get("state") and validation.get("model_sha256"),
                "Training manifest lacks final artifact identity")
        training[branch] = dict(manifest=record["path"], source_commit=payload["commit"],
                                machine=payload["machine"], source_checkpoint=payload["checkpoint"],
                                source_checkpoint_sha256=payload["checkpoint_sha256"],
                                overrides=opts, validation=validation, log_file=payload["log_file"])
    return training


def audit(args, records):
    expected = expected_ids(args.dataset)
    require(len(expected) == 1839, "Original validation dataset must contain exactly 1839 unique episodes")
    training = audit_training(records)
    require(len({records["eval/" + item["label"]]["payload"]["machine"] for item in cases()}) == 1,
            "Final evaluations must all run on the same machine")
    episodes, results = {}, []
    for item in cases():
        record = records["eval/" + item["label"]]
        checks = {"MODEL.ACTIVE_LOOKAHEAD.enabled": False, "TASK_CONFIG.SEED": 100,
                  "GPU_NUMBERS": 1, "NUM_ENVIRONMENTS": 8, "EVAL.EPISODE_COUNT": -1,
                  "EVAL.fast_eval": False, "EVAL.SAVE_RESULTS": True,
                  "IL.freeze_navigation_backbone": item["frozen"],
                  "MODEL.RAENWM.enabled": item["enabled"],
                  "MODEL.RAENWM.rgb_fusion_enabled": item["enabled"],
                  "MODEL.RAENWM.rgb_fusion_alpha": item["alpha"],
                  "MODEL.RAENWM.rgb_fusion_align_navigation_cls": item["aligned"],
                  "MODEL.RAENWM.condition_source_pose": item["source"]}
        for key, value in checks.items():
            override(record, key, value)
        payload = record["payload"]
        result_dir = args.root / "eval" / item["name"] / "results" / item["name"] / "eval_results"
        ep, summary = load_episodes(str(result_dir / "stats_ep_ckpt_*_val_unseen_r0_w1.json"), 1839, expected)
        require(len(summary["files"]) == 1, "Expected exactly one single-rank episode file")
        aggregates = list(result_dir.glob("stats_ckpt_*_val_unseen.json"))
        require(len(aggregates) == 1, "Missing/ambiguous aggregate result")
        aggregate = read_json(aggregates[0])
        for key in ("success", "spl", "path_length"):
            require(abs(aggregate[key] - summary["metrics"][key]) < 1e-5,
                    f"Aggregate differs from per-episode {key}: {item['label']}")
        checkpoint, digest = payload["checkpoint"], payload["checkpoint_sha256"]
        require(isinstance(digest, str) and len(digest) == 64, "Missing checkpoint SHA256")
        if item["case"] == "base":
            require(digest == BASE_SHA, "Evaluation baseline SHA256 mismatch")
        if item["iteration"] == 2000:
            require(digest == training[item["case"]]["validation"]["model_sha256"],
                    "Evaluated final model differs from completed training model")
        frozen_audit = None
        if item["frozen"]:
            frozen_record = records[f"frozen/{item['case']}_{item['iteration']}"]
            frozen_audit = frozen_record["payload"]
            require(frozen_audit.get("ok") is True and frozen_audit.get("policy", {}).get("all_equal") is True,
                    "Frozen navigation policy changed")
            require(frozen_audit.get("base_sha256") == BASE_SHA and frozen_audit.get("checkpoint_sha256") == digest,
                    "Frozen audit evaluated a different checkpoint or base")
            require(frozen_audit.get("iteration") == item["iteration"], "Frozen audit iteration mismatch")
            saved = frozen_audit.get("saved_contract", {})
            require(saved.get("freeze_navigation_backbone") is True and
                    saved.get("align_navigation_cls") == item["aligned"] and
                    saved.get("condition_source_pose") == item["source"], "Frozen audit contract mismatch")
        episodes[item["label"]] = ep
        summary.update(item, manifest=record["path"], checkpoint=checkpoint, checkpoint_sha256=digest,
                       source_commit=payload["commit"], machine=payload["machine"], overrides=payload["overrides"],
                       frozen_checkpoint_audit=frozen_audit,
                       log_file=payload["log_file"], source_pose=source_diagnostics(result_dir, item))
        summary["effective_rgb_enabled"] = item["enabled"] and item["alpha"] > 0
        results.append(summary)
    hashes = {r["checkpoint_sha256"] for r in results if r["case"] in ("legacy", "a0", "a025", "a05", "a1")}
    require(len(hashes) == 1, "Strength/source ablations did not use the identical checkpoint")
    for result in results:
        result["paired_vs_base"] = paired_summary(episodes["base"], episodes[result["label"]])
        result["delta_sr_vs_historical_percentage_points"] = 100 * result["metrics"]["success"] - HISTORICAL_SR_PERCENT
        result["sr_exceeds_new_base"] = result["paired_vs_base"]["net_successes"] > 0
        result["sr_exceeds_historical"] = result["delta_sr_vs_historical_percentage_points"] > 0
    comparisons = []
    pairs = [("source_fix", "legacy", "a1")]
    pairs += [("alpha", "a0", label) for label in ("a025", "a05", "a1")]
    for iteration in (1000, 2000):
        pairs += [("continue_training", "base", f"A_{iteration}"),
                  ("frozen_rgb_vs_full_no_rgb", f"A_{iteration}", f"B_{iteration}"),
                  ("frozen_aligned_rgb_vs_full_no_rgb", f"A_{iteration}", f"C_{iteration}"),
                  ("alignment", f"B_{iteration}", f"C_{iteration}")]
    for group, left, right in pairs:
        comparisons.append(dict(group=group, reference=left, candidate=right,
                                paired=paired_summary(episodes[left], episodes[right])))
    qualified = [r for r in results if r["effective_rgb_enabled"] and r["source"] == "context_last"
                 and r["sr_exceeds_new_base"] and r["sr_exceeds_historical"]]
    qualified.sort(key=lambda r: (r["metrics"]["success"], r["metrics"]["spl"]), reverse=True)
    best = qualified[0] if qualified else None
    report = dict(format="rgb_only_optimization_final_v1", status="completed", finished_at=now(),
                  completed_evaluations=12, completed_trainings=3, completed_frozen_audits=4, all_e24_disabled=True,
                  dataset=str(args.dataset), expected_episodes=1839,
                  historical_sr_percent=HISTORICAL_SR_PERCENT, training=training, results=results,
                  comparisons=comparisons, goal_met_on_this_validation_set=bool(best),
                  qualifying_rgb_labels=[r["label"] for r in qualified],
                  recommended=None if best is None else dict(label=best["label"], checkpoint=best["checkpoint"],
                      checkpoint_sha256=best["checkpoint_sha256"], overrides=best["overrides"],
                      metrics=best["metrics"], paired_vs_base=best["paired_vs_base"]),
                  selection_rule="正确源合同、有效RGB且SR同时超过本轮基线与历史63.7303%；按SR、再SPL排序。",
                  limitations=["单个训练随机种子100；不代表跨种子稳定收益。",
                               "所有选择都发生在同一个val_unseen集合；固定1000/2000点仍有选优偏差。",
                               "配对检验p值未校正多重比较，不作为可靠提升或显著性的承诺。",
                               "B/C相对A同时改变冻结策略与RGB接入；不能把全部差值仅归因于世界模型。",
                               "alpha0仍执行世界模型查询，但最后有效注入为零，不属于开启有效RGB的目标方案。"])
    if args.historical_baseline:
        old_ep, old_summary = load_episodes(args.historical_baseline, 1839, expected)
        old_summary["paired_old_to_new_base"] = paired_summary(old_ep, episodes["base"])
        report["historical_baseline_recheck"] = old_summary
    elif (args.root / "audit/base_reproduction.json").is_file():
        reproduction = read_json(args.root / "audit/base_reproduction.json")
        report["historical_baseline_reproduction_artifact"] = dict(
            path=str(args.root / "audit/base_reproduction.json"), payload=reproduction,
            note="既有配对审计作为补充证据；本脚本本次未重新加载该审计所引用的旧episode文件。")
    return report


def render(report):
    rows = {row["label"]: row for row in report["results"]}
    best = report["recommended"]
    if best:
        paired = best["paired_vs_base"]
        outcome = (f"本轮找到了符合“关闭E24、有效RGB注入且SR超过两条基线”的配置：{best['label']}。"
                   f"SR为{100 * best['metrics']['success']:.4f}%，相对本轮基线{paired['delta_sr_percentage_points']:+.4f}个百分点，"
                   f"SPL变化{paired['delta_spl_percentage_points']:+.4f}个百分点。此结论限于本次验证集与训练种子。")
    else:
        outcome = "本轮三步实验已完成，但没有得到有效RGB、正确源合同且SR同时超过本轮基线与历史63.7303%的配置。不能宣称已获得最终导航性能提升。"
    lines = ["# RGB注入导航优化最终报告", "", outcome, "",
             "已验证3个训练完成、12个全量评测正常退出；每份评测都恰好覆盖原始验证集相同的1839个唯一episode。全部训练和评测的命令均明确关闭E24。B/C的1000/2000步共4份检查点通过冻结审计：全部保存的导航张量与基线严格相等，编码器来源元数据一致，审计SHA256与实际测评检查点一致。", "",
             "## 完整结果", "",
             "| 配置 | 有效RGB | SR (%) | SPL (%) | ΔSR vs新基线 | ΔSPL vs新基线 | 救回/损失 | 路径长(m) | 超新/旧基线SR |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in report["results"]:
        m, p = row["metrics"], row["paired_vs_base"]
        lines.append(f"| {row['label']} | {'是' if row['effective_rgb_enabled'] else '否'} | "
                     f"{100*m['success']:.4f} | {100*m['spl']:.4f} | {p['delta_sr_percentage_points']:+.4f} | "
                     f"{p['delta_spl_percentage_points']:+.4f} | {p['rescued']}/{p['lost']} | {m['path_length']:.4f} | "
                     f"{'是' if row['sr_exceeds_new_base'] else '否'}/{'是' if row['sr_exceeds_historical'] else '否'} |")
    lines += ["", "base为原始14200无世界模型基线；legacy为旧查询位姿合同的5200模型；a0/a025/a05/a1使用修复后的上下文源位姿，注入强度分别为0/0.25/0.5/1。A为无RGB全网续训；B冻结导航底座训练原注入模块；C再让预测经过同一冻结视觉适配层。后缀为新增训练步数。所有Δ单位均为百分点。", "",
              f"alpha0不算有效RGB开启。历史SR目标仍为63.7303%；本轮重新计算的基线为{100*rows['base']['metrics']['success']:.4f}%。"
              "两条基线分别比较，不以新基线替代用户原来的历史目标。相同权重在运行环境复测中若有差异，不能自动归因于源位姿修复，具体原因需另行定位。", "",
              "## 三步比较", "",
              "| 比较 | ΔSR (百分点) | ΔSPL (百分点) | 救回/损失 |",
              "|---|---:|---:|---:|"]
    for comp in report["comparisons"]:
        p = comp["paired"]
        lines.append(f"| {comp['reference']} → {comp['candidate']} | {p['delta_sr_percentage_points']:+.4f} | "
                     f"{p['delta_spl_percentage_points']:+.4f} | {p['rescued']}/{p['lost']} |")
    fixed = next(c for c in report["comparisons"] if c["group"] == "source_fix")["paired"]
    a0_delta = rows["a0"]["paired_vs_base"]["delta_sr_percentage_points"]
    injected_delta = next(c for c in report["comparisons"] if c["group"] == "alpha" and c["candidate"] == "a1")["paired"]["delta_sr_percentage_points"]
    lines += ["", f"源合同修复在同一个5200检查点上带来SR {fixed['delta_sr_percentage_points']:+.4f}个百分点、"
              f"SPL {fixed['delta_spl_percentage_points']:+.4f}个百分点。这是本次完整导航比较，不能仅凭一轮就把变化推广到所有模型。", "",
              f"5200模型关闭有效注入时，相对原始基线SR {a0_delta:+.4f}个百分点；在它自身基础上开启强度1，SR再变化{injected_delta:+.4f}个百分点。"
              "这两项分别反映当前模型离基线的距离与当前接入的边际作用；alpha0不构成相同数据的无注入训练对照，A分支才补上续训对照。", "",
              "A与B/C的差异同时包含冻结策略和RGB接入；B与C在同一步数的差异更直接检验特征空间对齐。"
              "所有分支均只训练一个种子；本报告不将这些描述性差值宣称为已确认的因果或统计显著提升。", "",
              "## 源位姿诊断", "",
              "分母是具备4帧上下文并进入条件组批的候选查询，部分候选可能在后续步骤被过滤。位置/朝向不同表示查询时位姿与缓存末帧位姿不同，"
              "不是episode占比，也不是修复后仍使用错误坐标系的比例。修复允许这两个位姿不同，并把条件转换到正确参考系。", "",
              "| 配置 | 查询分母 | 任意差异数量/比例 | 朝向差异数量/比例 | 位置差异数量/比例 | 平均朝向差(度) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        d = row["source_pose"]
        if d is None:
            continue
        t, rates = d["totals"], d["rates"]
        cells = [f"{t[key]}/{100*rates[key+'_fraction']:.3f}%" for key in
                 ("mismatch_queries", "yaw_mismatch_queries", "position_mismatch_queries")]
        lines.append(f"| {row['label']} | {t['queries']} | {' | '.join(cells)} | {d['mean_abs_yaw_degrees']:.4f} |")
    lines += ["", "## 模型与产物", ""]
    if best:
        lines += [f"建议保留并复核的配置：**{best['label']}**；检查点 `{best['checkpoint']}`。",
                  "完整推理开关保存在final_summary.json的recommended.overrides，复现时应整体使用，并将输出目录改为新的复测目录。", ""]
    for branch, training in report["training"].items():
        lines += [f"- {branch}训练机最终检查点：`{training['validation']['model']}`；训练状态：`{training['validation']['state']}`；清单：`{training['manifest']}`。"]
    lines += ["", "| 评测配置 | 实际检查点 | 结果文件 |", "|---|---|---|"]
    for row in report["results"]:
        lines.append(f"| {row['label']} | `{row['checkpoint']}` | `{row['files'][0]}` |")
    lines += ["", "## 结论与后续", ""]
    if best:
        p = best["paired_vs_base"]
        if p["delta_spl_percentage_points"] < 0:
            lines += ["该配置达到本次SR目标，但SPL仍低于本轮基线；应如实保留路线效率的代价，不能称作各项性能全面提升。", ""]
        lines += ["下一步使用相同固定配方补做独立训练种子，并报告全部预定检查点；避免继续在同一验证集反复筛选后只展示峰值。", ""]
    else:
        lines += ["当前实验没有验证目标成立。应先围绕配对结果中损失的原有成功路线，检查ghost预测质量与接入影响，"
                  "再比较保留纯观测主路、仅局部修正ghost的方案；不据此直接扩大同配方训练时长。", ""]
    lines += report["limitations"]
    if "historical_baseline_recheck" in report:
        p = report["historical_baseline_recheck"]["paired_old_to_new_base"]
        lines += ["", f"另核对既有基线逐episode结果：新复测相对旧结果SR {p['delta_sr_percentage_points']:+.4f}个百分点，"
                  f"SPL {p['delta_spl_percentage_points']:+.4f}个百分点，救回/损失{p['rescued']}/{p['lost']}。"]
    elif "historical_baseline_reproduction_artifact" in report:
        lines += ["", "旧基线与新基线的既有配对审计见 `audit/base_reproduction.json`；该补充文件已嵌入汇总JSON，"
                  "本次最终脚本未重新加载其引用的旧episode文件，不能把它当作本脚本重新完成的旧结果验证。"]
    lines += ["", f"生成时间：{report['finished_at']}。完整配置、源提交、SHA256、配对episode编号和未经多重比较校正的检验值见final_summary.json。", ""]
    return "\n".join(lines)


def render_collection(report):
    lines = [
        "# RGB-only 自动执行结果汇总", "",
        "三组训练、12次完整评测和4份冻结审计已全部通过验收。",
        "每份评测均覆盖相同的1839条任务，E24全部关闭。具体原因分析与模型选择等待用户后续发起。", "",
        "| 配置 | SR (%) | SPL (%) | 平均路径 (m) |",
        "|---|---:|---:|---:|",
    ]
    for row in report["results"]:
        metrics = row["metrics"]
        lines.append(f"| {row['label']} | {100*metrics['success']:.4f} | "
                     f"{100*metrics['spl']:.4f} | {metrics['path_length']:.4f} |")
    lines += ["", "完整指标、逐路线配对、源位姿统计、配置、检查点路径及校验值见同目录final_summary.json。",
              "此文件只记录执行结果，不提供原因判断、模型推荐或稳定提升结论。", "",
              f"完成时间：{report['finished_at']}。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-manifest-dir", type=Path,
                        help="Publisher output directory holding A/B/C.json and B/C_1000/2000_frozen.json")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--historical-baseline", help="Optional glob of historical baseline per-episode JSON")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--collect-only", action="store_true",
                        help="Validate and tabulate results; defer interpretation and model selection")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.train_manifest_dir = (args.train_manifest_dir or args.root / "audit/train_completion").resolve()
    summary_path, report_path = args.root / "final_summary.json", args.root / "final_report.md"
    try:
        while True:
            records, pending = manifest_inventory(args)
            if not pending:
                break
            status = dict(format="rgb_only_optimization_final_v1", status="pending", checked_at=now(),
                          completed_artifacts=len(records), expected_artifacts=19, pending=pending,
                          conclusion="实验尚未完成，不作最终导航性能结论。")
            write_json(summary_path, status)
            if report_path.exists():
                write_text(report_path, "# 实验尚未完成\n\n" + status["conclusion"] + "\n")
            print(json.dumps(status, ensure_ascii=False), flush=True)
            if not args.wait:
                return 3
            time.sleep(30)
        report = audit(args, records)
        if args.collect_only:
            for key in ("recommended", "qualifying_rgb_labels", "selection_rule",
                        "goal_met_on_this_validation_set"):
                report.pop(key, None)
            report["analysis_status"] = "deferred_until_user_request"
        write_text(report_path, render_collection(report) if args.collect_only else render(report))
        write_json(summary_path, report)
        print(json.dumps(dict(status="completed", goal_met=report.get("goal_met_on_this_validation_set"),
                              report=str(report_path), summary=str(summary_path)), ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = dict(format="rgb_only_optimization_final_v1", status="failed", checked_at=now(),
                       error_type=type(exc).__name__, error=str(exc),
                       conclusion="实验失败或产物校验不通过，不作最终导航性能结论。")
        write_json(summary_path, failure)
        if report_path.exists():
            write_text(report_path, "# 最终报告校验失败\n\n" + failure["conclusion"] + "\n\n" + str(exc) + "\n")
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
