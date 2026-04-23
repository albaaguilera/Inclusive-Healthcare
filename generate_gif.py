from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learning.qpbrs_seeds import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_GROUP_ORDER,
    ExperimentConfig,
    PROJECT_ROOT,
    _draw_cost_line,
    _draw_environment_panel,
    _plot_dumbbell_panel,
    build_env,
    initialize_capabilities,
    load_irl,
    load_profiles,
    reset_env,
    state_of,
    train_policy,
)
from learning.utils import action_mask_from_classify, group_key_from_initial, masked_argmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a policy evolution GIF using the current Figure 3 visual style."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-peh", type=int, default=16)
    parser.add_argument("--num-sw", type=int, default=20)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--train-max-steps", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=120)
    parser.add_argument("--snapshot-interval", type=int, default=8)
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
        "--output",
        default=str(PROJECT_ROOT / "output" / "paper_selected" / "policy_evolution_figure3_style_n16.gif"),
        help="Output GIF path.",
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        size=args.size,
        num_peh=args.num_peh,
        num_sw=args.num_sw,
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
        seeds=(args.seed,),
    )


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else float("nan")


def _single_stats(initial: List[float], final: List[float]) -> Dict[str, float]:
    return {
        "initial_mean": _safe_mean(initial),
        "initial_std": 0.0,
        "final_mean": _safe_mean(final),
        "final_std": 0.0,
    }


def build_snapshot_artifact(
    env,
    *,
    scenario: str,
    seed: int,
    initial_admin: Dict[str, str],
    initial_trust: Dict[str, str],
    initial_health: Dict[str, float],
    initial_cap_bh: Dict[str, float],
    initial_cap_af: Dict[str, float],
    init_health_budget: float,
    init_social_budget: float,
) -> Dict[str, Any]:
    bh_trace: Dict[str, List[float]] = {}
    af_trace: Dict[str, List[float]] = {}
    health_trace: Dict[str, List[float]] = {}
    admin_trace: Dict[str, List[int]] = {}

    for agent in env.possible_agents:
        peh = env.peh_agents[env.agent_name_mapping[agent]]
        caps = env.capabilities.get(agent, {}) or {}
        bh_trace[agent] = [float(initial_cap_bh[agent]), float(caps.get("Bodily Health", np.nan))]
        af_trace[agent] = [float(initial_cap_af[agent]), float(caps.get("Affiliation", np.nan))]
        health_trace[agent] = [float(initial_health[agent]), float(peh.health_state)]
        admin_trace[agent] = [int(initial_admin[agent] == "registered"), int(peh.administrative_state == "registered")]

    return {
        "scenario": scenario,
        "seed": seed,
        "env": copy.deepcopy(env),
        "bh_trace": bh_trace,
        "af_trace": af_trace,
        "health_trace": health_trace,
        "admin_trace": admin_trace,
        "init_admin": dict(initial_admin),
        "init_trust": dict(initial_trust),
        "init_health_budget": float(init_health_budget),
        "init_social_budget": float(init_social_budget),
    }


def build_snapshot_summary(cfg: ExperimentConfig, artifact: Dict[str, Any]) -> Dict[str, Any]:
    bh_init_all: List[float] = []
    bh_final_all: List[float] = []
    grouped_af_init = {group: [] for group in DEFAULT_GROUP_ORDER}
    grouped_af_final = {group: [] for group in DEFAULT_GROUP_ORDER}
    health_init: List[float] = []
    health_final: List[float] = []
    admin_init: List[int] = []
    admin_final: List[int] = []

    for agent, bh_trace in artifact["bh_trace"].items():
        group = group_key_from_initial(artifact["init_admin"], artifact["init_trust"], agent)
        bh_init_all.append(float(bh_trace[0]))
        bh_final_all.append(float(bh_trace[-1]))
        grouped_af_init[group].append(float(artifact["af_trace"][agent][0]))
        grouped_af_final[group].append(float(artifact["af_trace"][agent][-1]))
        health_init.append(float(artifact["health_trace"][agent][0]))
        health_final.append(float(artifact["health_trace"][agent][-1]))
        admin_init.append(int(artifact["admin_trace"][agent][0]))
        admin_final.append(int(artifact["admin_trace"][agent][-1]))

    env = artifact["env"]
    return {
        "capabilities": {
            "bodily_health_all": _single_stats(bh_init_all, bh_final_all),
            "affiliation_groups": {
                group: _single_stats(grouped_af_init[group], grouped_af_final[group]) for group in DEFAULT_GROUP_ORDER
            },
        },
        "functionings": {
            "Healthy": _single_stats(
                [1.0 if v >= cfg.healthy_threshold else 0.0 for v in health_init],
                [1.0 if v >= cfg.healthy_threshold else 0.0 for v in health_final],
            ),
            "Registered": _single_stats([float(v) for v in admin_init], [float(v) for v in admin_final]),
        },
        "costs": {
            "healthcare_spend": {
                "mean": float(artifact["init_health_budget"] - env.context.healthcare_budget),
                "std": 0.0,
            },
            "social_service_spend": {
                "mean": float(artifact["init_social_budget"] - env.context.social_service_budget),
                "std": 0.0,
            },
        },
    }


def collect_policy_snapshots(
    cfg: ExperimentConfig,
    profiles: List[Dict[str, Any]],
    q_tables: Dict[str, np.ndarray],
    *,
    policy_on: bool,
    seed: int,
    snapshot_interval: int,
) -> List[Dict[str, Any]]:
    phase = "ON" if policy_on else "OFF"
    eval_seed = seed * 100_000 + 50_000
    env = build_env(cfg, profiles, policy_on=policy_on, max_steps=cfg.eval_max_steps)
    reset_env(env, eval_seed, profiles)

    initial_admin: Dict[str, str] = {}
    initial_trust: Dict[str, str] = {}
    initial_health: Dict[str, float] = {}
    for agent in env.possible_agents:
        peh = env.peh_agents[env.agent_name_mapping[agent]]
        initial_admin[agent] = peh.administrative_state
        initial_trust[agent] = getattr(peh, "trust_type", "MODERATE_TRUST")
        initial_health[agent] = float(peh.health_state)

    init_health_budget = float(env.context.healthcare_budget)
    init_social_budget = float(env.context.social_service_budget)
    initial_cap_bh, initial_cap_af = initialize_capabilities(env)

    snapshots: List[Dict[str, Any]] = []

    def append_snapshot(step: int) -> None:
        artifact = build_snapshot_artifact(
            env,
            scenario=phase,
            seed=seed,
            initial_admin=initial_admin,
            initial_trust=initial_trust,
            initial_health=initial_health,
            initial_cap_bh=initial_cap_bh,
            initial_cap_af=initial_cap_af,
            init_health_budget=init_health_budget,
            init_social_budget=init_social_budget,
        )
        snapshots.append(
            {
                "scenario": phase,
                "step": step,
                "artifact": artifact,
                "summary": build_snapshot_summary(cfg, artifact),
            }
        )

    append_snapshot(0)
    total_steps = 0
    while env.agents and total_steps < cfg.eval_max_steps:
        total_steps += 1
        agent = env.agent_selection
        if env.dones.get(agent, False):
            env.step(None)
            continue

        state = state_of(env, agent, cfg)
        mask = action_mask_from_classify(env, agent)
        action = masked_argmax(q_tables[agent][state], mask)
        env.step(action)

        if total_steps % snapshot_interval == 0 or not env.agents or total_steps == cfg.eval_max_steps:
            append_snapshot(total_steps)

    return snapshots


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.uint8)


def render_frame(
    cfg: ExperimentConfig,
    on_snapshot: Dict[str, Any],
    off_snapshot: Dict[str, Any],
) -> np.ndarray:
    fig = plt.figure(figsize=(14.8, 10.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.28], hspace=0.24, wspace=0.30)
    xlim = (0.0, 1.03)

    for row_idx, payload in enumerate([on_snapshot, off_snapshot]):
        scenario = payload["scenario"]
        artifact = payload["artifact"]
        summary_stats = payload["summary"]
        final_color = "#1b9e77" if scenario == "ON" else "#d95f02"

        ax_grid = fig.add_subplot(gs[row_idx, 0])
        _draw_environment_panel(
            ax_grid,
            artifact,
            title=f"Policy {scenario} - step {payload['step']}",
            show_legend=(row_idx == 0),
            font_big=15.0,
            font_small=10.4,
        )

        right = gs[row_idx, 1].subgridspec(3, 1, height_ratios=[1.18, 0.92, 0.46], hspace=0.58)
        ax_caps = fig.add_subplot(right[0, 0])
        ax_fun = fig.add_subplot(right[1, 0])
        ax_cost = fig.add_subplot(right[2, 0])

        cap_rows = [
            ("BH all", summary_stats["capabilities"]["bodily_health_all"]),
            ("AF non-reg low", summary_stats["capabilities"]["affiliation_groups"]["NONREG_LOW"]),
            ("AF non-reg mod", summary_stats["capabilities"]["affiliation_groups"]["NONREG_MOD"]),
            ("AF reg low", summary_stats["capabilities"]["affiliation_groups"]["REG_LOW"]),
            ("AF reg mod", summary_stats["capabilities"]["affiliation_groups"]["REG_MOD"]),
        ]
        _plot_dumbbell_panel(
            ax_caps,
            cap_rows,
            title="Capabilities (agents' actions)",
            final_color=final_color,
            xlim=xlim,
            show_legend=True,
            font_med=13.8,
            font_small=11.0,
        )

        fun_rows = [
            ("Healthy", summary_stats["functionings"]["Healthy"]),
            ("Registered", summary_stats["functionings"]["Registered"]),
        ]
        _plot_dumbbell_panel(
            ax_fun,
            fun_rows,
            title="Functionings (agents' state)",
            final_color=final_color,
            xlim=xlim,
            show_legend=False,
            font_med=13.8,
            font_small=11.0,
        )
        ax_fun.set_xlabel("Population (%)", fontsize=11.0, labelpad=5)
        _draw_cost_line(
            ax_cost,
            summary_stats,
            scenario=scenario,
            font_med=12.2,
            font_small=11.0,
        )

    image = fig_to_rgb(fig)
    plt.close(fig)
    return image


def create_gif(cfg: ExperimentConfig, output_path: Path, *, snapshot_interval: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(cfg.num_peh)
    irl = load_irl(str(PROJECT_ROOT / "output" / "irl_calibration_results_raval.json"))

    q_tables_on, _, _ = train_policy(
        cfg,
        profiles,
        policy_on=True,
        seed=cfg.seeds[0],
        irl=irl,
        feature_cols=DEFAULT_FEATURE_COLS,
    )
    q_tables_off, _, _ = train_policy(
        cfg,
        profiles,
        policy_on=False,
        seed=cfg.seeds[0],
        irl=irl,
        feature_cols=DEFAULT_FEATURE_COLS,
    )

    on_snapshots = collect_policy_snapshots(
        cfg, profiles, q_tables_on, policy_on=True, seed=cfg.seeds[0], snapshot_interval=snapshot_interval
    )
    off_snapshots = collect_policy_snapshots(
        cfg, profiles, q_tables_off, policy_on=False, seed=cfg.seeds[0], snapshot_interval=snapshot_interval
    )

    frame_count = min(len(on_snapshots), len(off_snapshots))
    frames = [Image.fromarray(render_frame(cfg, on_snapshots[i], off_snapshots[i])) for i in range(frame_count)]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
        optimize=False,
    )

    meta = {
        "generated_at": datetime.now().isoformat(),
        "config": asdict(cfg),
        "snapshot_interval": snapshot_interval,
        "frame_count": frame_count,
        "output_gif": str(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    output_path = Path(args.output)
    result = create_gif(cfg, output_path, snapshot_interval=args.snapshot_interval)
    print(f"Saved Figure 3 style policy evolution GIF to: {result}")


if __name__ == "__main__":
    main()
