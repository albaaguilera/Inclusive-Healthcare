from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learning.qpbrs_seeds import (
    DEFAULT_GROUP_ORDER,
    ExperimentConfig,
    PROJECT_ROOT,
    build_report,
    generate_figures,
    run_experiment,
)


@dataclass(frozen=True)
class ScaleSpec:
    label: str
    num_peh: int
    num_sw: int
    size: int


DEFAULT_SCALES: Tuple[ScaleSpec, ...] = (
    ScaleSpec("n4_sw8_sz5", num_peh=4, num_sw=8, size=5),
    ScaleSpec("n8_sw15_sz5", num_peh=8, num_sw=15, size=5),
    ScaleSpec("n8_sw15_sz7", num_peh=8, num_sw=15, size=7),
    ScaleSpec("n8_sw15_sz9", num_peh=8, num_sw=15, size=9),
    ScaleSpec("n16_sw20_sz7", num_peh=16, num_sw=20, size=7),
    ScaleSpec("n16_sw20_sz9", num_peh=16, num_sw=20, size=9),
    ScaleSpec("n16_sw25_sz9", num_peh=16, num_sw=25, size=9),
    ScaleSpec("n16_sw30_sz9", num_peh=16, num_sw=30, size=9),
    ScaleSpec("n25_sw25_sz8", num_peh=25, num_sw=25, size=8),
    ScaleSpec("n25_sw30_sz8", num_peh=25, num_sw=30, size=8),
    ScaleSpec("n25_sw35_sz8", num_peh=25, num_sw=35, size=8),
    ScaleSpec("n32_sw32_sz8", num_peh=32, num_sw=32, size=8),
    ScaleSpec("n32_sw40_sz8", num_peh=32, num_sw=40, size=8),
    ScaleSpec("n16_sw20_sz11", num_peh=16, num_sw=20, size=11),
    ScaleSpec("n16_sw25_sz11", num_peh=16, num_sw=25, size=11),
    ScaleSpec("n16_sw30_sz11", num_peh=16, num_sw=30, size=11),
)
BASELINE_LABEL = "n8_sw15_sz7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic PBRS/Q-learning experiments across multiple simulation scales "
            "and compare each scale to the baseline equilibrium."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--train-max-steps", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--eps-min", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=float, default=0.995)
    parser.add_argument("--pbrs-beta", type=float, default=0.02)
    parser.add_argument("--max-enc", type=int, default=10)
    parser.add_argument("--max-noneng", type=int, default=10)
    parser.add_argument("--healthy-threshold", type=float, default=3.0)
    parser.add_argument(
        "--scale-labels",
        nargs="+",
        default=None,
        help="Optional subset of scale labels to run, e.g. --scale-labels n8_sw15_sz7 n16_sw30_sz9",
    )
    parser.add_argument(
        "--outdir-name",
        default=None,
        help="Optional existing or custom run folder name under output/scalability for resuming a sweep.",
    )
    return parser.parse_args()


def cfg_for_scale(spec: ScaleSpec, args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        size=spec.size,
        num_peh=spec.num_peh,
        num_sw=spec.num_sw,
        episodes=args.episodes,
        train_max_steps=args.train_max_steps,
        eval_max_steps=args.eval_max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        pbrs_beta=args.pbrs_beta,
        max_enc=args.max_enc,
        max_noneng=args.max_noneng,
        healthy_threshold=args.healthy_threshold,
        seeds=tuple(args.seeds),
    )


def save_scale_outputs(scale_dir: Path, cfg: ExperimentConfig, outputs: Dict[str, pd.DataFrame]) -> None:
    scale_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(scale_dir / f"{name}.csv", index=False)

    report_text = build_report(
        cfg,
        outputs["training_summary_by_seed"],
        outputs["eval_summary_by_seed"],
        outputs["eval_summary_aggregated"],
        outputs["strategy_summary_aggregated"],
    )
    (scale_dir / "summary.md").write_text(report_text, encoding="utf-8")
    generate_figures(outputs, scale_dir)
    summary_payload = {
        "config": asdict(cfg),
        "generated_at": datetime.now().isoformat(),
        "output_files": sorted(str(path.relative_to(scale_dir)) for path in scale_dir.rglob("*") if path.is_file()),
    }
    (scale_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def get_metric_mean(aggregated_eval_df: pd.DataFrame, scenario: str, metric: str) -> float:
    block = aggregated_eval_df.loc[
        (aggregated_eval_df["scenario"] == scenario) & (aggregated_eval_df["metric"] == metric),
        "mean",
    ]
    return float(block.iloc[0]) if not block.empty else float("nan")


def get_metric_std(aggregated_eval_df: pd.DataFrame, scenario: str, metric: str) -> float:
    block = aggregated_eval_df.loc[
        (aggregated_eval_df["scenario"] == scenario) & (aggregated_eval_df["metric"] == metric),
        "std",
    ]
    return float(block.iloc[0]) if not block.empty else float("nan")


def paired_gap_stats(
    eval_summary_by_seed: pd.DataFrame,
    left_scenario: str,
    left_metric: str,
    right_scenario: str,
    right_metric: str | None = None,
) -> Tuple[float, float]:
    right_metric = right_metric or left_metric

    left_block = eval_summary_by_seed.loc[
        eval_summary_by_seed["scenario"] == left_scenario,
        ["seed", left_metric],
    ].rename(columns={left_metric: "left_value"})
    right_block = eval_summary_by_seed.loc[
        eval_summary_by_seed["scenario"] == right_scenario,
        ["seed", right_metric],
    ].rename(columns={right_metric: "right_value"})
    merged = left_block.merge(right_block, on="seed", how="inner")
    if merged.empty:
        return float("nan"), float("nan")

    gaps = merged["left_value"] - merged["right_value"]
    std = float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
    return float(gaps.mean()), std


def strategy_alignment(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    scenario: str,
    max_local_step: int = 5,
) -> float:
    cols = ["group", "local_step", "top_action"]
    cur = current.loc[(current["scenario"] == scenario) & (current["local_step"] < max_local_step), cols]
    ref = baseline.loc[(baseline["scenario"] == scenario) & (baseline["local_step"] < max_local_step), cols]
    merged = cur.merge(ref, on=["group", "local_step"], suffixes=("_cur", "_ref"))
    if merged.empty:
        return float("nan")
    return float((merged["top_action_cur"] == merged["top_action_ref"]).mean())


def scale_summary_row(
    spec: ScaleSpec,
    outputs: Dict[str, pd.DataFrame],
    baseline_strategy_df: pd.DataFrame,
) -> Dict[str, Any]:
    eval_by_seed = outputs["eval_summary_by_seed"]
    eval_agg = outputs["eval_summary_aggregated"]
    strategy_agg = outputs["strategy_summary_aggregated"]

    on_reward = get_metric_mean(eval_agg, "ON", "eval_total_reward")
    off_reward = get_metric_mean(eval_agg, "OFF", "eval_total_reward")
    on_reward_std = get_metric_std(eval_agg, "ON", "eval_total_reward")
    off_reward_std = get_metric_std(eval_agg, "OFF", "eval_total_reward")
    on_health = get_metric_mean(eval_agg, "ON", "final_mean_health")
    off_health = get_metric_mean(eval_agg, "OFF", "final_mean_health")
    on_health_std = get_metric_std(eval_agg, "ON", "final_mean_health")
    off_health_std = get_metric_std(eval_agg, "OFF", "final_mean_health")
    on_cost = get_metric_mean(eval_agg, "ON", "healthcare_spend")
    off_cost = get_metric_mean(eval_agg, "OFF", "healthcare_spend")
    on_cost_std = get_metric_std(eval_agg, "ON", "healthcare_spend")
    off_cost_std = get_metric_std(eval_agg, "OFF", "healthcare_spend")
    on_healthy = get_metric_mean(eval_agg, "ON", "final_share_healthy")
    off_healthy = get_metric_mean(eval_agg, "OFF", "final_share_healthy")
    on_healthy_std = get_metric_std(eval_agg, "ON", "final_share_healthy")
    off_healthy_std = get_metric_std(eval_agg, "OFF", "final_share_healthy")

    align_on = strategy_alignment(strategy_agg, baseline_strategy_df, scenario="ON")
    align_off = strategy_alignment(strategy_agg, baseline_strategy_df, scenario="OFF")

    reward_gap, reward_gap_std = paired_gap_stats(
        eval_by_seed, "ON", "eval_total_reward", "OFF", "eval_total_reward"
    )
    health_gap, health_gap_std = paired_gap_stats(
        eval_by_seed, "ON", "final_mean_health", "OFF", "final_mean_health"
    )
    healthy_gap, healthy_gap_std = paired_gap_stats(
        eval_by_seed, "ON", "final_share_healthy", "OFF", "final_share_healthy"
    )
    healthcare_advantage, healthcare_advantage_std = paired_gap_stats(
        eval_by_seed,
        "OFF",
        "healthcare_spend",
        "ON",
        "healthcare_spend",
    )

    reward_positive = reward_gap > 0
    health_positive = health_gap > 0
    cost_positive = healthcare_advantage > 0
    align_on_ok = bool(np.isfinite(align_on) and align_on >= 0.75)
    align_off_ok = bool(np.isfinite(align_off) and align_off >= 0.60)

    score_components = [
        float(reward_positive),
        float(health_positive),
        float(cost_positive),
        0.0 if not np.isfinite(align_on) else align_on,
        0.0 if not np.isfinite(align_off) else align_off,
    ]
    preservation_score = float(np.mean(score_components))
    preserved = reward_positive and health_positive and cost_positive and align_on_ok and align_off_ok

    return {
        "scale_label": spec.label,
        "num_peh": spec.num_peh,
        "num_sw": spec.num_sw,
        "size": spec.size,
        "on_reward_mean": on_reward,
        "on_reward_std": on_reward_std,
        "off_reward_mean": off_reward,
        "off_reward_std": off_reward_std,
        "reward_gap_on_minus_off": reward_gap,
        "reward_gap_on_minus_off_std": reward_gap_std,
        "on_health_mean": on_health,
        "on_health_std": on_health_std,
        "off_health_mean": off_health,
        "off_health_std": off_health_std,
        "health_gap_on_minus_off": health_gap,
        "health_gap_on_minus_off_std": health_gap_std,
        "healthy_gap_on_minus_off": healthy_gap,
        "healthy_gap_on_minus_off_std": healthy_gap_std,
        "on_healthcare_spend_mean": on_cost,
        "on_healthcare_spend_std": on_cost_std,
        "off_healthcare_spend_mean": off_cost,
        "off_healthcare_spend_std": off_cost_std,
        "healthcare_advantage_off_minus_on": healthcare_advantage,
        "healthcare_advantage_off_minus_on_std": healthcare_advantage_std,
        "on_healthy_share_std": on_healthy_std,
        "off_healthy_share_std": off_healthy_std,
        "strategy_alignment_on": align_on,
        "strategy_alignment_off": align_off,
        "equilibrium_preserved": preserved,
        "preservation_score": preservation_score,
    }


def plot_policy_gaps(scale_summary_df: pd.DataFrame, figures_dir: Path) -> None:
    labels = scale_summary_df["scale_label"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("reward_gap_on_minus_off", "reward_gap_on_minus_off_std", "Reward gap (ON - OFF)", "#1b9e77"),
        ("health_gap_on_minus_off", "health_gap_on_minus_off_std", "Health gap (ON - OFF)", "#7570b3"),
        ("healthcare_advantage_off_minus_on", "healthcare_advantage_off_minus_on_std", "Healthcare advantage (OFF - ON)", "#d95f02"),
    ]
    for ax, (metric, std_metric, title, color) in zip(axes, metrics):
        ax.bar(
            x,
            scale_summary_df[metric],
            yerr=scale_summary_df[std_metric],
            color=color,
            alpha=0.85,
            error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": "0.25"},
        )
        ax.axhline(0.0, color="0.3", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figures_dir / "policy_gaps_by_scale.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_strategy_alignment(scale_summary_df: pd.DataFrame, figures_dir: Path) -> None:
    labels = scale_summary_df["scale_label"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - width / 2, scale_summary_df["strategy_alignment_on"], width=width, color="#1b9e77", label="Policy ON")
    ax.bar(x + width / 2, scale_summary_df["strategy_alignment_off"], width=width, color="#d95f02", label="Policy OFF")
    ax.axhline(0.75, color="#1b9e77", linestyle="--", lw=1.0, alpha=0.8)
    ax.axhline(0.60, color="#d95f02", linestyle="--", lw=1.0, alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Match to baseline strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_title("Strategy alignment to baseline equilibrium")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "strategy_alignment_by_scale.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_preservation_score(scale_summary_df: pd.DataFrame, figures_dir: Path) -> None:
    labels = scale_summary_df["scale_label"].tolist()
    x = np.arange(len(labels))
    colors = ["#4daf4a" if keep else "#e41a1c" for keep in scale_summary_df["equilibrium_preserved"]]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x, scale_summary_df["preservation_score"], color=colors, alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Preservation score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_title("Equilibrium preservation across scales")
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figures_dir / "equilibrium_preservation_score.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_scalability_report(
    outdir: Path,
    scale_summary_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    preserved = scale_summary_df.loc[scale_summary_df["equilibrium_preserved"]].copy()
    not_preserved = scale_summary_df.loc[~scale_summary_df["equilibrium_preserved"]].copy()
    if not preserved.empty:
        largest = preserved.sort_values(["num_peh", "size", "num_sw"]).iloc[-1]
        largest_line = (
            f"Largest preserved scale under the heuristic criterion: `{largest['scale_label']}` "
            f"(N={int(largest['num_peh'])}, SW={int(largest['num_sw'])}, size={int(largest['size'])})."
        )
    else:
        largest_line = "No tested scale satisfied the heuristic preservation criterion."

    preserved_lines = (
        [f"- `{row.scale_label}`: score={row.preservation_score:.3f}" for row in preserved.itertuples()]
        if not preserved.empty
        else ["- None"]
    )
    broken_lines = (
        [f"- `{row.scale_label}`: score={row.preservation_score:.3f}" for row in not_preserved.itertuples()]
        if not not_preserved.empty
        else ["- None"]
    )

    lines = [
        "# Scalability summary",
        "",
        f"Seeds per scale: {list(args.seeds)}",
        f"Baseline scale: `{BASELINE_LABEL}`",
        "",
        "Heuristic preservation criterion:",
        "- Policy ON must outperform policy OFF on reward and health.",
        "- Policy ON must spend less healthcare budget than policy OFF.",
        "- Strategy alignment to the baseline must be at least 0.75 for policy ON and 0.60 for policy OFF.",
        "",
        largest_line,
        "",
        "## Preserved scales",
        *preserved_lines,
        "",
        "## Non-preserved scales",
        *broken_lines,
        "",
        "## Scale summary",
        "",
        "```csv",
        scale_summary_df.to_csv(index=False).strip(),
        "```",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    summary_json = {
        "generated_at": datetime.now().isoformat(),
        "baseline_label": BASELINE_LABEL,
        "seeds": list(args.seeds),
        "scales": [asdict(spec) for spec in DEFAULT_SCALES],
        "output_files": sorted(str(path.relative_to(outdir)) for path in outdir.rglob("*") if path.is_file()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_id = args.outdir_name or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    outdir = PROJECT_ROOT / "output" / "scalability" / run_id
    scales_dir = outdir / "scales"
    figures_dir = outdir / "figures"
    scales_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected_labels = set(args.scale_labels) if args.scale_labels else {spec.label for spec in DEFAULT_SCALES}
    spec_lookup = {spec.label: spec for spec in DEFAULT_SCALES}
    unknown = sorted(selected_labels.difference(spec_lookup))
    if unknown:
        raise ValueError(f"Unknown scale labels: {unknown}. Valid labels: {sorted(spec_lookup)}")

    for spec in DEFAULT_SCALES:
        if spec.label not in selected_labels:
            continue
        scale_dir = scales_dir / spec.label
        required_outputs = [
            scale_dir / "eval_summary_by_seed.csv",
            scale_dir / "eval_summary_aggregated.csv",
            scale_dir / "strategy_summary_aggregated.csv",
        ]
        if all(path.exists() for path in required_outputs):
            print(f"\nReusing cached scale {spec.label} (N={spec.num_peh}, SW={spec.num_sw}, size={spec.size})")
            continue
        print(f"\nRunning scale {spec.label} (N={spec.num_peh}, SW={spec.num_sw}, size={spec.size})")
        cfg = cfg_for_scale(spec, args)
        outputs = run_experiment(cfg)
        save_scale_outputs(scale_dir, cfg, outputs)
    baseline_dir = scales_dir / BASELINE_LABEL
    if not baseline_dir.exists():
        raise RuntimeError(f"Baseline scale `{BASELINE_LABEL}` is required in {scales_dir} before aggregation.")
    baseline_strategy_df = pd.read_csv(baseline_dir / "strategy_summary_aggregated.csv")

    scale_summary_rows: List[Dict[str, Any]] = []
    for spec in DEFAULT_SCALES:
        scale_dir = scales_dir / spec.label
        if not scale_dir.exists():
            continue
        eval_summary_by_seed = pd.read_csv(scale_dir / "eval_summary_by_seed.csv")
        eval_summary_aggregated = pd.read_csv(scale_dir / "eval_summary_aggregated.csv")
        strategy_summary_aggregated = pd.read_csv(scale_dir / "strategy_summary_aggregated.csv")
        outputs_stub = {
            "eval_summary_by_seed": eval_summary_by_seed,
            "eval_summary_aggregated": eval_summary_aggregated,
            "strategy_summary_aggregated": strategy_summary_aggregated,
        }
        scale_summary_rows.append(
            scale_summary_row(
                spec,
                outputs_stub,
                baseline_strategy_df,
            )
        )

    scale_summary_df = pd.DataFrame(scale_summary_rows).sort_values(["num_peh", "size", "num_sw"]).reset_index(drop=True)
    scale_summary_df.to_csv(outdir / "scalability_summary.csv", index=False)

    plot_policy_gaps(scale_summary_df, figures_dir)
    plot_strategy_alignment(scale_summary_df, figures_dir)
    plot_preservation_score(scale_summary_df, figures_dir)
    save_scalability_report(outdir, scale_summary_df, args)

    preserved = scale_summary_df.loc[scale_summary_df["equilibrium_preserved"]]
    print("\nScalability summary")
    print(scale_summary_df.to_string(index=False))
    if not preserved.empty:
        largest = preserved.sort_values(["num_peh", "size", "num_sw"]).iloc[-1]
        print(
            f"\nLargest preserved scale: {largest['scale_label']} "
            f"(N={int(largest['num_peh'])}, SW={int(largest['num_sw'])}, size={int(largest['size'])})"
        )
    else:
        print("\nNo tested scale satisfied the heuristic preservation criterion.")
    print(f"\nSaved scalability results to: {outdir}")


if __name__ == "__main__":
    main()
