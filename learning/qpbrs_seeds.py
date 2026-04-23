from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from environment.context import Actions, Context, update_all_capability_scores
from environment.model import GridMAInequityEnv
from learning.utils import (
    a_label,
    action_mask_from_classify,
    cached_irl_potential,
    get_state,
    group_key,
    group_key_from_initial,
    irl_potential_from_env,
    load_irl,
    masked_argmax,
    plot_policy_summary_comparison,
    health_to_color,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_COLS = [
    "prev_encounters",
    "health_state",
    "homelessness_duration",
    "history_of_abuse",
    "trust_building",
    "age",
    "income",
]
DEFAULT_GROUP_ORDER = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]
SCENARIO_COLORS = {"ON": "#1b9e77", "OFF": "#d95f02"}
GROUP_COLORS = {
    "NONREG_LOW": "#d73027",
    "NONREG_MOD": "#fc8d59",
    "REG_LOW": "#1f9ac9",
    "REG_MOD": "#2c7fb8",
}
GROUP_LABELS = {
    "NONREG_LOW": "Non-registered + Low trust",
    "NONREG_MOD": "Non-registered + Moderate trust",
    "REG_LOW": "Registered + Low trust",
    "REG_MOD": "Registered + Moderate trust",
}


@dataclass
class ExperimentConfig:
    size: int = 7
    num_peh: int = 8
    num_sw: int = 15
    episodes: int = 400
    train_max_steps: int = 100
    eval_max_steps: int = 500
    alpha: float = 0.2
    gamma: float = 0.99
    epsilon: float = 0.1
    eps_min: float = 0.01
    eps_decay: float = 0.995
    pbrs_beta: float = 0.02
    max_enc: int = 10
    max_noneng: int = 10
    healthy_threshold: float = 3.0
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the inclusive healthcare PBRS/Q-learning simulation across multiple seeds "
            "and export mean/std summaries for policy ON and policy OFF."
        )
    )
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--num-peh", type=int, default=8)
    parser.add_argument("--num-sw", type=int, default=15)
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
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to sweep. Example: --seeds 0 1 2 3 4",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_profiles(num_peh: int) -> List[Dict[str, Any]]:
    profile_map = {
        4: PROJECT_ROOT / "output" / "peh_sample4.json",
        8: PROJECT_ROOT / "output" / "peh_sample8.json",
        16: PROJECT_ROOT / "output" / "peh_sample16.json",
    }
    if num_peh not in profile_map:
        profile_map[num_peh] = ensure_generated_profiles(num_peh)

    with profile_map[num_peh].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_generated_profiles(num_peh: int) -> Path:
    """Create a deterministic stratified PEH profile file for larger cohorts.

    The repository ships fixed cohorts for N=4, 8, and 16. For larger paper
    sweeps we keep reproducibility by deriving a balanced empirical cohort
    from the N=16 template and saving it once under output/peh_sample<N>.json.
    """
    if num_peh <= 0:
        raise ValueError("num_peh must be positive.")

    outpath = PROJECT_ROOT / "output" / f"peh_sample{num_peh}.json"
    if outpath.exists():
        return outpath

    template_path = PROJECT_ROOT / "output" / "peh_sample16.json"
    if not template_path.exists():
        raise ValueError(
            f"Unsupported num_peh={num_peh}; missing template file {template_path}."
        )

    with template_path.open("r", encoding="utf-8") as handle:
        template_profiles = json.load(handle)

    cells = [
        ("LOW_TRUST", "NON_REGISTERED"),
        ("MODERATE_TRUST", "NON_REGISTERED"),
        ("LOW_TRUST", "REGISTERED"),
        ("MODERATE_TRUST", "REGISTERED"),
    ]
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {cell: [] for cell in cells}
    for profile in template_profiles:
        key = (str(profile["trust_type"]).upper(), str(profile["admin_state"]).upper())
        if key in grouped:
            grouped[key].append(profile)

    if any(not grouped[cell] for cell in cells):
        raise ValueError("Cannot generate larger cohort because the N=16 template is not stratified.")

    per_cell = [num_peh // len(cells)] * len(cells)
    for idx in range(num_peh % len(cells)):
        per_cell[idx] += 1

    generated: List[Dict[str, Any]] = []
    next_person_id = 100_000 + num_peh * 1_000
    for cell, count in zip(cells, per_cell):
        base_profiles = grouped[cell]
        for idx in range(count):
            base = dict(base_profiles[idx % len(base_profiles)])
            replicate = idx // len(base_profiles)
            if replicate:
                base["person_id"] = next_person_id
                next_person_id += 1
                base["age"] = float(np.clip(float(base.get("age", 45.0)) + ((idx % 3) - 1), 18, 90))
                income = float(base.get("income", 400.0))
                base["income"] = max(0.0, income * (1.0 + 0.03 * ((idx % 5) - 2)))
                duration = float(base.get("homelessness_duration", 5.0))
                base["homelessness_duration"] = max(0.0, duration + ((idx % 4) - 1))
            base["health_state"] = 3.0
            generated.append(base)

    outpath.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    return outpath


def build_env(
    cfg: ExperimentConfig,
    profiles: List[Dict[str, Any]],
    *,
    policy_on: bool,
    max_steps: int,
) -> GridMAInequityEnv:
    ctx = Context(grid_size=cfg.size)
    ctx.set_scenario(policy_inclusive_healthcare=policy_on)
    return GridMAInequityEnv(
        context=ctx,
        render_mode=None,
        size=cfg.size,
        num_peh=len(profiles),
        num_social_agents=cfg.num_sw,
        peh_profiles=profiles,
        max_steps=max_steps,
    )


def reset_env(env: GridMAInequityEnv, seed: int, profiles: List[Dict[str, Any]]) -> None:
    set_global_seed(seed)
    env.reset(seed=seed, options={"peh_profiles": profiles})


def _admin_state_to_int(admin_state: str) -> int:
    return int(str(admin_state) == "registered")


def initialize_capabilities(env: GridMAInequityEnv) -> Tuple[Dict[str, float], Dict[str, float]]:
    for agent in env.possible_agents:
        peh = env.peh_agents[env.agent_name_mapping[agent]]
        possible, impossible = env._classify_actions(peh)
        env.current_possible_actions[agent] = possible
        env.current_impossible_actions[agent] = impossible
    update_all_capability_scores(env)

    init_cap_bh: Dict[str, float] = {}
    init_cap_af: Dict[str, float] = {}
    for agent in env.possible_agents:
        caps = env.capabilities.get(agent, {}) or {}
        init_cap_bh[agent] = float(caps.get("Bodily Health", np.nan))
        init_cap_af[agent] = float(caps.get("Affiliation", np.nan))
    return init_cap_bh, init_cap_af


def initialize_q_tables(
    env: GridMAInequityEnv,
    *,
    max_enc: int,
    max_noneng: int,
) -> Dict[str, np.ndarray]:
    n_health = int(
        (env.peh_agents[0].max_health - env.peh_agents[0].min_health) / env.peh_agents[0].health_step
    ) + 1
    q_tables: Dict[str, np.ndarray] = {}
    for agent in env.possible_agents:
        shape = (
            env.size,
            env.size,
            n_health,
            2,
            2,
            max_enc + 1,
            max_noneng + 1,
            env.action_space(agent).n,
        )
        q_tables[agent] = np.zeros(shape, dtype=float)
    return q_tables


def state_of(env: GridMAInequityEnv, agent: str, cfg: ExperimentConfig) -> Tuple[int, ...]:
    obs = env.observe(agent)
    peh = env.peh_agents[env.agent_name_mapping[agent]]
    return get_state(obs, peh, max_enc=cfg.max_enc, max_noneng=cfg.max_noneng)


def safe_best_next(q_values: np.ndarray, mask: np.ndarray) -> float:
    masked = np.where(mask == 1, q_values, -1e18)
    return float(np.max(masked))


def train_policy(
    cfg: ExperimentConfig,
    profiles: List[Dict[str, Any]],
    *,
    policy_on: bool,
    seed: int,
    irl: Dict[str, Any],
    feature_cols: Iterable[str],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, float]]:
    phase = "ON" if policy_on else "OFF"
    env = build_env(cfg, profiles, policy_on=policy_on, max_steps=cfg.train_max_steps)
    reset_env(env, seed * 100_000, profiles)
    q_tables = initialize_q_tables(env, max_enc=cfg.max_enc, max_noneng=cfg.max_noneng)
    explore_rng = np.random.default_rng(seed * 100_000 + 17)

    epsilon = cfg.epsilon
    training_rows: List[Dict[str, Any]] = []

    for episode in range(cfg.episodes):
        episode_seed = seed * 100_000 + episode
        cached_irl_potential.cache_clear()
        reset_env(env, episode_seed, profiles)

        obs = {agent: env.observe(agent) for agent in env.agents}
        state = {
            agent: get_state(
                obs[agent],
                env.peh_agents[env.agent_name_mapping[agent]],
                max_enc=cfg.max_enc,
                max_noneng=cfg.max_noneng,
            )
            for agent in env.agents
        }
        ep_returns = {agent: 0.0 for agent in env.agents}
        episode_groups = {
            agent: group_key(env.peh_agents[env.agent_name_mapping[agent]]) for agent in env.agents
        }

        for _ in range(cfg.train_max_steps):
            if not env.agents:
                break

            agent = env.agent_selection
            if env.dones.get(agent, False):
                env.step(None)
                continue

            mask = action_mask_from_classify(env, agent)
            feasible = np.flatnonzero(mask)
            if float(explore_rng.random()) < epsilon:
                action = int(explore_rng.choice(feasible)) if len(feasible) else int(explore_rng.integers(env.action_space(agent).n))
            else:
                action = masked_argmax(q_tables[agent][state[agent]], mask)

            phi_s = None
            if policy_on:
                phi_s = irl_potential_from_env(env, agent, irl, list(feature_cols))

            env.step(action)

            base_reward = float(env.rewards.get(agent, 0.0))
            shaped_reward = base_reward
            if policy_on and phi_s is not None:
                phi_sp = irl_potential_from_env(env, agent, irl, list(feature_cols))
                shaped_reward += cfg.pbrs_beta * (cfg.gamma * phi_sp - phi_s)

            next_state = state_of(env, agent, cfg)
            next_mask = action_mask_from_classify(env, agent)
            best_next = safe_best_next(q_tables[agent][next_state], next_mask)

            s = state[agent]
            q_tables[agent][s + (action,)] += cfg.alpha * (
                shaped_reward + cfg.gamma * best_next - q_tables[agent][s + (action,)]
            )

            state[agent] = next_state
            ep_returns[agent] += shaped_reward

        epsilon = max(cfg.eps_min, epsilon * cfg.eps_decay)

        row: Dict[str, Any] = {
            "scenario": phase,
            "seed": seed,
            "episode": episode,
            "epsilon": float(epsilon),
            "total_reward": float(sum(ep_returns.values())),
        }
        for group in DEFAULT_GROUP_ORDER:
            group_values = [ret for agent, ret in ep_returns.items() if episode_groups[agent] == group]
            row[f"mean_reward_{group.lower()}"] = float(np.mean(group_values)) if group_values else np.nan
        training_rows.append(row)

    training_df = pd.DataFrame(training_rows)
    summary = {
        "training_reward_mean": float(training_df["total_reward"].mean()),
        "training_reward_std_episode": float(training_df["total_reward"].std(ddof=1))
        if len(training_df) > 1
        else 0.0,
        "training_reward_last50_mean": float(training_df["total_reward"].tail(50).mean()),
        "training_reward_last50_std_episode": float(training_df["total_reward"].tail(50).std(ddof=1))
        if len(training_df.tail(50)) > 1
        else 0.0,
        "training_reward_best_episode": float(training_df["total_reward"].max()),
        "training_reward_final_episode": float(training_df["total_reward"].iloc[-1]),
    }
    return q_tables, training_df, summary


def evaluate_policy(
    cfg: ExperimentConfig,
    profiles: List[Dict[str, Any]],
    q_tables: Dict[str, np.ndarray],
    *,
    policy_on: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    phase = "ON" if policy_on else "OFF"
    eval_seed = seed * 100_000 + 50_000
    env = build_env(cfg, profiles, policy_on=policy_on, max_steps=cfg.eval_max_steps)
    reset_env(env, eval_seed, profiles)

    initial_admin = {}
    initial_trust = {}
    initial_health = {}
    for agent in env.possible_agents:
        peh = env.peh_agents[env.agent_name_mapping[agent]]
        initial_admin[agent] = peh.administrative_state
        initial_trust[agent] = getattr(peh, "trust_type", "MODERATE_TRUST")
        initial_health[agent] = float(peh.health_state)

    init_healthcare_budget = float(env.context.healthcare_budget)
    init_social_budget = float(env.context.social_service_budget)
    initial_cap_bh, initial_cap_af = initialize_capabilities(env)

    step_rows: List[Dict[str, Any]] = []
    agent_rewards = defaultdict(float)
    local_steps = defaultdict(int)

    total_steps = 0
    while env.agents and total_steps < cfg.eval_max_steps:
        total_steps += 1
        agent = env.agent_selection
        if env.dones.get(agent, False):
            env.step(None)
            continue

        peh = env.peh_agents[env.agent_name_mapping[agent]]
        state = get_state(
            env.observe(agent),
            peh,
            max_enc=cfg.max_enc,
            max_noneng=cfg.max_noneng,
        )
        mask = action_mask_from_classify(env, agent)
        action = masked_argmax(q_tables[agent][state], mask)
        env.step(action)

        reward = float(env.rewards.get(agent, 0.0))
        agent_rewards[agent] += reward
        caps = env.capabilities.get(agent, {}) or {}
        peh_after = env.peh_agents[env.agent_name_mapping[agent]]

        step_rows.append(
            {
                "scenario": phase,
                "seed": seed,
                "global_step": total_steps,
                "agent": agent,
                "group": group_key_from_initial(initial_admin, initial_trust, agent),
                "local_step": local_steps[agent],
                "action_idx": int(action),
                "action_name": Actions(int(action)).name,
                "reward": reward,
                "health_after": float(peh_after.health_state),
                "admin_after": peh_after.administrative_state,
                "trust_type": getattr(peh_after, "trust_type", "MODERATE_TRUST"),
                "engagement_counter": int(getattr(peh_after, "engagement_counter", 0)),
                "non_engagement_counter": int(getattr(peh_after, "non_engagement_counter", 0)),
                "cap_bh": float(caps.get("Bodily Health", np.nan)),
                "cap_af": float(caps.get("Affiliation", np.nan)),
                "healthcare_budget": float(env.context.healthcare_budget),
                "social_service_budget": float(env.context.social_service_budget),
            }
        )
        local_steps[agent] += 1

    agent_rows: List[Dict[str, Any]] = []
    for agent in env.possible_agents:
        peh = env.peh_agents[env.agent_name_mapping[agent]]
        caps = env.capabilities.get(agent, {}) or {}
        agent_rows.append(
            {
                "scenario": phase,
                "seed": seed,
                "agent": agent,
                "group": group_key_from_initial(initial_admin, initial_trust, agent),
                "initial_admin": initial_admin[agent],
                "initial_trust": initial_trust[agent],
                "initial_health": initial_health[agent],
                "final_admin": peh.administrative_state,
                "final_health": float(peh.health_state),
                "final_engagement_counter": int(getattr(peh, "engagement_counter", 0)),
                "final_non_engagement_counter": int(getattr(peh, "non_engagement_counter", 0)),
                "final_cap_bh": float(caps.get("Bodily Health", np.nan)),
                "final_cap_af": float(caps.get("Affiliation", np.nan)),
                "eval_return": float(agent_rewards.get(agent, 0.0)),
                "n_local_steps": int(local_steps.get(agent, 0)),
            }
        )

    step_df = pd.DataFrame(step_rows)
    agent_df = pd.DataFrame(agent_rows)

    if agent_df.empty:
        raise RuntimeError(f"No evaluation rows were produced for scenario={phase}, seed={seed}.")

    final_health = agent_df["final_health"]
    final_registered = (agent_df["final_admin"] == "registered").astype(float)
    final_healthy = (agent_df["final_health"] >= cfg.healthy_threshold).astype(float)

    summary = {
        "scenario": phase,
        "seed": seed,
        "eval_total_reward": float(agent_df["eval_return"].sum()),
        "eval_mean_reward_per_agent": float(agent_df["eval_return"].mean()),
        "eval_std_reward_per_agent": float(agent_df["eval_return"].std(ddof=1))
        if len(agent_df) > 1
        else 0.0,
        "final_mean_health": float(final_health.mean()),
        "final_std_health_agents": float(final_health.std(ddof=1)) if len(final_health) > 1 else 0.0,
        "final_share_registered": float(final_registered.mean()),
        "final_share_healthy": float(final_healthy.mean()),
        "final_mean_cap_bh": float(agent_df["final_cap_bh"].mean()),
        "final_mean_cap_af": float(agent_df["final_cap_af"].mean()),
        "final_mean_engagement_counter": float(agent_df["final_engagement_counter"].mean()),
        "final_mean_non_engagement_counter": float(agent_df["final_non_engagement_counter"].mean()),
        "healthcare_spend": float(init_healthcare_budget - env.context.healthcare_budget),
        "social_service_spend": float(init_social_budget - env.context.social_service_budget),
        "eval_total_steps": int(total_steps),
    }
    for group in DEFAULT_GROUP_ORDER:
        group_slice = agent_df.loc[agent_df["group"] == group, "eval_return"]
        summary[f"eval_mean_reward_{group.lower()}"] = float(group_slice.mean()) if not group_slice.empty else np.nan

    final_agents = agent_df.set_index("agent")
    bh_trace: Dict[str, List[float]] = {}
    af_trace: Dict[str, List[float]] = {}
    health_trace: Dict[str, List[float]] = {}
    admin_trace: Dict[str, List[int]] = {}

    for agent in env.possible_agents:
        row = final_agents.loc[agent]
        final_caps = env.capabilities.get(agent, {}) or {}
        final_bh = row["final_cap_bh"] if pd.notna(row["final_cap_bh"]) else final_caps.get("Bodily Health", np.nan)
        final_af = row["final_cap_af"] if pd.notna(row["final_cap_af"]) else final_caps.get("Affiliation", np.nan)

        bh_trace[agent] = [float(initial_cap_bh[agent]), float(final_bh)]
        af_trace[agent] = [float(initial_cap_af[agent]), float(final_af)]
        health_trace[agent] = [float(initial_health[agent]), float(row["final_health"])]
        admin_trace[agent] = [
            _admin_state_to_int(initial_admin[agent]),
            _admin_state_to_int(str(row["final_admin"])),
        ]

    plot_artifact = {
        "scenario": phase,
        "seed": seed,
        "env": env,
        "bh_trace": bh_trace,
        "af_trace": af_trace,
        "health_trace": health_trace,
        "admin_trace": admin_trace,
        "init_admin": dict(initial_admin),
        "init_trust": dict(initial_trust),
        "init_health_budget": init_healthcare_budget,
        "init_social_budget": init_social_budget,
    }
    return step_df, agent_df, summary, plot_artifact


def dominant_strategy_by_seed(step_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if step_df.empty:
        return pd.DataFrame(rows)

    grouped = step_df.groupby(["scenario", "seed", "group", "local_step"], dropna=False)
    for (scenario, seed, group, local_step), block in grouped:
        counts = block["action_name"].value_counts(dropna=False)
        if counts.empty:
            continue
        dominant_action = counts.index[0]
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "group": group,
                "local_step": int(local_step),
                "dominant_action": dominant_action,
                "dominant_action_idx": int(
                    block.loc[block["action_name"] == dominant_action, "action_idx"].iloc[0]
                ),
                "dominant_action_share": float(counts.iloc[0] / counts.sum()),
                "n_action_samples": int(counts.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_numeric_by_seed(
    df: pd.DataFrame,
    *,
    group_col: str = "scenario",
    value_cols: Iterable[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario, block in df.groupby(group_col):
        for metric in value_cols:
            series = block[metric].dropna()
            rows.append(
                {
                    group_col: scenario,
                    "metric": metric,
                    "mean": float(series.mean()) if not series.empty else np.nan,
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "min": float(series.min()) if not series.empty else np.nan,
                    "max": float(series.max()) if not series.empty else np.nan,
                    "n_seeds": int(series.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def summarize_strategy_across_seeds(strategy_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if strategy_df.empty:
        return pd.DataFrame(rows)

    for (scenario, group, local_step), block in strategy_df.groupby(["scenario", "group", "local_step"]):
        support = block["dominant_action"].value_counts()
        top_action = support.index[0]
        top_block = block.loc[block["dominant_action"] == top_action]
        rows.append(
            {
                "scenario": scenario,
                "group": group,
                "local_step": int(local_step),
                "top_action": top_action,
                "seed_support_fraction": float(support.iloc[0] / len(block)),
                "top_action_mean_share": float(top_block["dominant_action_share"].mean()),
                "top_action_std_share": float(top_block["dominant_action_share"].std(ddof=1))
                if len(top_block) > 1
                else 0.0,
                "n_seeds": int(len(block)),
            }
        )
    return pd.DataFrame(rows)


def paired_policy_difference(eval_summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in eval_summary_df.columns if col not in {"scenario", "seed"}]
    wide = eval_summary_df.pivot(index="seed", columns="scenario", values=metric_cols)
    rows: List[Dict[str, Any]] = []
    for metric in metric_cols:
        if (metric, "ON") not in wide.columns or (metric, "OFF") not in wide.columns:
            continue
        diff = (wide[(metric, "ON")] - wide[(metric, "OFF")]).dropna()
        rows.append(
            {
                "metric": metric,
                "on_minus_off_mean": float(diff.mean()) if not diff.empty else np.nan,
                "on_minus_off_std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                "min": float(diff.min()) if not diff.empty else np.nan,
                "max": float(diff.max()) if not diff.empty else np.nan,
                "n_seeds": int(diff.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def select_representative_artifacts(
    eval_summary_df: pd.DataFrame,
    eval_artifacts: Dict[str, Dict[int, Dict[str, Any]]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    representative: Dict[str, Dict[str, Any]] = {}
    representative_seeds: Dict[str, int] = {}
    metrics = [
        "eval_total_reward",
        "final_mean_health",
        "final_share_healthy",
        "final_share_registered",
        "healthcare_spend",
        "social_service_spend",
        "final_mean_cap_bh",
        "final_mean_cap_af",
    ]

    for scenario in ["ON", "OFF"]:
        block = eval_summary_df.loc[eval_summary_df["scenario"] == scenario].copy()
        if block.empty:
            continue

        scenario_metrics = [metric for metric in metrics if metric in block.columns]
        if not scenario_metrics:
            seed = int(block.sort_values("seed").iloc[0]["seed"])
        else:
            values = block[scenario_metrics].astype(float)
            center = values.mean(axis=0)
            scale = values.std(axis=0, ddof=1).replace(0.0, 1.0).fillna(1.0)
            distance = (((values - center) / scale) ** 2).sum(axis=1)
            ranked = block.assign(_distance=distance).sort_values(["_distance", "seed"])
            seed = int(ranked.iloc[0]["seed"])

        representative[scenario] = eval_artifacts[scenario][seed]
        representative_seeds[scenario] = seed

    return representative, representative_seeds


def _paired_mean_std(df: pd.DataFrame, initial_col: str, final_col: str) -> Dict[str, float]:
    if df.empty:
        return {
            "initial_mean": np.nan,
            "initial_std": 0.0,
            "final_mean": np.nan,
            "final_std": 0.0,
        }

    initial_series = df[initial_col].dropna()
    final_series = df[final_col].dropna()
    return {
        "initial_mean": float(initial_series.mean()) if not initial_series.empty else np.nan,
        "initial_std": float(initial_series.std(ddof=1)) if len(initial_series) > 1 else 0.0,
        "final_mean": float(final_series.mean()) if not final_series.empty else np.nan,
        "final_std": float(final_series.std(ddof=1)) if len(final_series) > 1 else 0.0,
    }


def build_policy_summary_across_seeds(
    cfg: ExperimentConfig,
    eval_summary_df: pd.DataFrame,
    eval_artifacts: Dict[str, Dict[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for scenario in ["ON", "OFF"]:
        scenario_artifacts = eval_artifacts.get(scenario, {})
        function_rows: List[Dict[str, Any]] = []
        capability_rows: List[Dict[str, Any]] = []

        for seed, artifact in scenario_artifacts.items():
            bh_init_all = []
            bh_final_all = []
            grouped_bh_init = {group: [] for group in DEFAULT_GROUP_ORDER}
            grouped_bh_final = {group: [] for group in DEFAULT_GROUP_ORDER}
            grouped_af_init = {group: [] for group in DEFAULT_GROUP_ORDER}
            grouped_af_final = {group: [] for group in DEFAULT_GROUP_ORDER}
            health_init = []
            health_final = []
            admin_init = []
            admin_final = []
            grouped_health_init = {group: [] for group in DEFAULT_GROUP_ORDER}
            grouped_health_final = {group: [] for group in DEFAULT_GROUP_ORDER}
            grouped_admin_init = {group: [] for group in DEFAULT_GROUP_ORDER}
            grouped_admin_final = {group: [] for group in DEFAULT_GROUP_ORDER}

            for agent, bh_trace in artifact["bh_trace"].items():
                group = group_key_from_initial(artifact["init_admin"], artifact["init_trust"], agent)
                bh_init_all.append(float(bh_trace[0]))
                bh_final_all.append(float(bh_trace[-1]))
                grouped_bh_init[group].append(float(bh_trace[0]))
                grouped_bh_final[group].append(float(bh_trace[-1]))

                af_trace = artifact["af_trace"][agent]
                grouped_af_init[group].append(float(af_trace[0]))
                grouped_af_final[group].append(float(af_trace[-1]))

                health_seq = artifact["health_trace"][agent]
                admin_seq = artifact["admin_trace"][agent]
                health_init.append(float(health_seq[0]))
                health_final.append(float(health_seq[-1]))
                admin_init.append(int(admin_seq[0]))
                admin_final.append(int(admin_seq[-1]))
                grouped_health_init[group].append(float(health_seq[0]))
                grouped_health_final[group].append(float(health_seq[-1]))
                grouped_admin_init[group].append(int(admin_seq[0]))
                grouped_admin_final[group].append(int(admin_seq[-1]))

            capability_rows.append(
                {
                    "seed": seed,
                    "capability": "Bodily Health",
                    "group": "ALL",
                    "initial": float(np.mean(bh_init_all)) if bh_init_all else np.nan,
                    "final": float(np.mean(bh_final_all)) if bh_final_all else np.nan,
                }
            )
            for group in DEFAULT_GROUP_ORDER:
                if grouped_bh_init[group]:
                    capability_rows.append(
                        {
                            "seed": seed,
                            "capability": "Bodily Health",
                            "group": group,
                            "initial": float(np.mean(grouped_bh_init[group])),
                            "final": float(np.mean(grouped_bh_final[group])),
                        }
                    )
                if not grouped_af_init[group]:
                    continue
                capability_rows.append(
                    {
                        "seed": seed,
                        "capability": "Affiliation",
                        "group": group,
                        "initial": float(np.mean(grouped_af_init[group])),
                        "final": float(np.mean(grouped_af_final[group])),
                    }
                )

            function_rows.extend(
                [
                    {
                        "seed": seed,
                        "metric": "Healthy",
                        "group": "ALL",
                        "initial": float(np.mean(np.array(health_init) >= cfg.healthy_threshold))
                        if health_init
                        else np.nan,
                        "final": float(np.mean(np.array(health_final) >= cfg.healthy_threshold))
                        if health_final
                        else np.nan,
                    },
                    {
                        "seed": seed,
                        "metric": "Registered",
                        "group": "ALL",
                        "initial": float(np.mean(admin_init)) if admin_init else np.nan,
                        "final": float(np.mean(admin_final)) if admin_final else np.nan,
                    },
                ]
            )
            for group in DEFAULT_GROUP_ORDER:
                if grouped_health_init[group]:
                    function_rows.append(
                        {
                            "seed": seed,
                            "metric": "Healthy",
                            "group": group,
                            "initial": float(np.mean(np.array(grouped_health_init[group]) >= cfg.healthy_threshold)),
                            "final": float(np.mean(np.array(grouped_health_final[group]) >= cfg.healthy_threshold)),
                        }
                    )
                if grouped_admin_init[group]:
                    function_rows.append(
                        {
                            "seed": seed,
                            "metric": "Registered",
                            "group": group,
                            "initial": float(np.mean(grouped_admin_init[group])),
                            "final": float(np.mean(grouped_admin_final[group])),
                        }
                    )

        capability_df = pd.DataFrame(capability_rows)
        function_df = pd.DataFrame(function_rows)
        scenario_eval = eval_summary_df.loc[eval_summary_df["scenario"] == scenario]

        bodily_health_groups: Dict[str, Dict[str, float]] = {}
        affiliation_groups: Dict[str, Dict[str, float]] = {}
        for group in DEFAULT_GROUP_ORDER:
            group_block = capability_df.loc[
                (capability_df["capability"] == "Bodily Health") & (capability_df["group"] == group)
            ]
            if not group_block.empty:
                bodily_health_groups[group] = _paired_mean_std(group_block, "initial", "final")
            group_block = capability_df.loc[
                (capability_df["capability"] == "Affiliation") & (capability_df["group"] == group)
            ]
            if group_block.empty:
                continue
            affiliation_groups[group] = _paired_mean_std(group_block, "initial", "final")

        functionings_by_group: Dict[str, Dict[str, Dict[str, float]]] = {"Healthy": {}, "Registered": {}}
        for metric in ["Healthy", "Registered"]:
            for group in DEFAULT_GROUP_ORDER:
                group_block = function_df.loc[
                    (function_df["metric"] == metric) & (function_df.get("group") == group)
                ]
                if group_block.empty:
                    continue
                functionings_by_group[metric][group] = _paired_mean_std(group_block, "initial", "final")

        summary[scenario] = {
            "capabilities": {
                "bodily_health_all": _paired_mean_std(
                    capability_df.loc[
                        (capability_df["capability"] == "Bodily Health") & (capability_df["group"] == "ALL")
                    ],
                    "initial",
                    "final",
                ),
                "bodily_health_groups": bodily_health_groups,
                "affiliation_groups": affiliation_groups,
            },
            "functionings": {
                metric: _paired_mean_std(
                    function_df.loc[(function_df["metric"] == metric) & (function_df["group"] == "ALL")],
                    "initial",
                    "final",
                )
                for metric in ["Healthy", "Registered"]
            },
            "functionings_by_group": functionings_by_group,
            "costs": {
                metric: {
                    "mean": float(series.mean()) if not series.empty else np.nan,
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                }
                for metric, series in {
                    "social_service_spend": scenario_eval["social_service_spend"].dropna(),
                    "healthcare_spend": scenario_eval["healthcare_spend"].dropna(),
                }.items()
            },
        }

    return summary


def build_report(
    cfg: ExperimentConfig,
    training_summary_df: pd.DataFrame,
    eval_summary_df: pd.DataFrame,
    aggregated_eval_df: pd.DataFrame,
    strategy_summary_df: pd.DataFrame,
) -> str:
    def fmt_metric(scenario: str, metric: str) -> str:
        block = aggregated_eval_df[
            (aggregated_eval_df["scenario"] == scenario) & (aggregated_eval_df["metric"] == metric)
        ]
        if block.empty:
            return "n/a"
        row = block.iloc[0]
        return f"{row['mean']:.4f} +/- {row['std']:.4f}"

    lines = [
        "# Multi-seed PBRS/Q-learning summary",
        "",
        f"Seeds: {list(cfg.seeds)}",
        f"Episodes per seed: {cfg.episodes}",
        "",
        "## Evaluation results",
        "",
        "| Metric | Policy ON | Policy OFF |",
        "| --- | --- | --- |",
    ]
    key_metrics = [
        "eval_total_reward",
        "eval_mean_reward_per_agent",
        "final_mean_health",
        "final_share_healthy",
        "final_share_registered",
        "final_mean_cap_bh",
        "final_mean_cap_af",
        "healthcare_spend",
        "social_service_spend",
        "eval_total_steps",
    ]
    for metric in key_metrics:
        lines.append(f"| `{metric}` | {fmt_metric('ON', metric)} | {fmt_metric('OFF', metric)} |")

    lines.extend(["", "## Training reward summary", ""])
    train_metrics = [
        "training_reward_mean",
        "training_reward_last50_mean",
        "training_reward_best_episode",
        "training_reward_final_episode",
    ]
    for metric in train_metrics:
        on_series = training_summary_df.loc[training_summary_df["scenario"] == "ON", metric].dropna()
        off_series = training_summary_df.loc[training_summary_df["scenario"] == "OFF", metric].dropna()
        on_text = f"{on_series.mean():.4f} +/- {on_series.std(ddof=1) if len(on_series) > 1 else 0.0:.4f}"
        off_text = f"{off_series.mean():.4f} +/- {off_series.std(ddof=1) if len(off_series) > 1 else 0.0:.4f}"
        lines.append(f"- `{metric}`: ON {on_text}; OFF {off_text}")

    lines.extend(["", "## Dominant greedy strategies", ""])
    preview = strategy_summary_df.loc[strategy_summary_df["local_step"] < 5].sort_values(
        ["scenario", "group", "local_step"]
    )
    for scenario in ["ON", "OFF"]:
        lines.append(f"### Policy {scenario}")
        scenario_block = preview.loc[preview["scenario"] == scenario]
        if scenario_block.empty:
            lines.append("- No strategy rows generated.")
            continue
        for group in DEFAULT_GROUP_ORDER:
            group_block = scenario_block.loc[scenario_block["group"] == group]
            if group_block.empty:
                continue
            strategy_bits = [
                f"t{int(row.local_step)}=`{row.top_action}` ({row.seed_support_fraction:.0%} seeds)"
                for row in group_block.itertuples()
            ]
            lines.append(f"- `{group}`: " + ", ".join(strategy_bits))
        lines.append("")

    lines.extend(["## Per-seed evaluation table", ""])
    lines.append("```csv")
    lines.append(eval_summary_df.to_csv(index=False).strip())
    lines.append("```")
    return "\n".join(lines)


def _savefig(path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _group_style(group: str) -> Dict[str, Any]:
    is_registered = group.startswith("REG")
    return {
        "color": GROUP_COLORS[group],
        "linestyle": "-" if is_registered else "--",
        "marker": "o" if is_registered else "x",
        "label": GROUP_LABELS[group],
    }


def plot_figure2_style(
    training_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    *,
    scenario: str,
    outdir: Path,
) -> None:
    metric_map = {
        "NONREG_LOW": "mean_reward_nonreg_low",
        "NONREG_MOD": "mean_reward_nonreg_mod",
        "REG_LOW": "mean_reward_reg_low",
        "REG_MOD": "mean_reward_reg_mod",
    }

    fig, (ax_rewards, ax_strategy) = plt.subplots(
        1,
        2,
        figsize=(12.8, 3.0),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(bottom=0.18, wspace=0.16)

    reward_block = training_df.loc[training_df["scenario"] == scenario].copy()
    for group in DEFAULT_GROUP_ORDER:
        metric = metric_map[group]
        style = _group_style(group)
        pivot = reward_block.pivot(index="episode", columns="seed", values=metric).sort_index()
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=1).fillna(0.0)
        ax_rewards.plot(
            mean.index,
            mean.values,
            color=style["color"],
            linestyle=style["linestyle"],
            lw=2.0,
            label=style["label"],
        )
        ax_rewards.fill_between(
            mean.index,
            (mean - std).values,
            (mean + std).values,
            color=style["color"],
            alpha=0.12,
        )

    ax_rewards.set_xlabel("Episode step", fontsize=12)
    ax_rewards.set_ylabel("Rewards", fontsize=12)
    ax_rewards.grid(True, axis="y", alpha=0.2)
    ax_rewards.tick_params(axis="both", labelsize=10)
    ax_rewards.legend(fontsize=8.0, loc="upper left", frameon=True)

    strategy_block = strategy_df.loc[strategy_df["scenario"] == scenario].copy()
    for group in DEFAULT_GROUP_ORDER:
        style = _group_style(group)
        group_block = strategy_block.loc[strategy_block["group"] == group].sort_values("local_step")
        if group_block.empty:
            continue

        agg = (
            group_block.groupby("local_step", as_index=False)
            .agg(
                mean_action_idx=("dominant_action_idx", "mean"),
                std_action_idx=("dominant_action_idx", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
                top_action=("dominant_action", lambda x: Counter(x).most_common(1)[0][0]),
            )
            .sort_values("local_step")
        )

        ax_strategy.errorbar(
            agg["local_step"],
            agg["mean_action_idx"],
            yerr=agg["std_action_idx"],
            fmt=style["marker"],
            color=style["color"],
            linestyle="none",
            capsize=3,
            elinewidth=1.0,
            markersize=6,
            markeredgewidth=1.2,
            markerfacecolor=(style["color"] if style["marker"] == "o" else "none"),
            label=style["label"],
        )

    num_actions = int(strategy_block["dominant_action_idx"].max()) + 1 if not strategy_block.empty else 5
    ax_strategy.set_yticks(range(num_actions))
    ax_strategy.set_yticklabels([a_label(i) for i in range(num_actions)], fontsize=12)
    ax_strategy.set_xlabel("Simulation step", fontsize=12)
    ax_strategy.set_ylabel("Actions", fontsize=12)
    ax_strategy.grid(True, axis="y", alpha=0.25)
    ax_strategy.tick_params(axis="x", labelsize=10)
    ax_strategy.legend(fontsize=8.0, loc="upper right", frameon=True)

    _savefig(outdir / f"figure2_policy_{scenario.lower()}_std.png")


def plot_figure2_rewards_only(
    training_df: pd.DataFrame,
    *,
    scenario: str,
    outpath: Path,
    figsize: Tuple[float, float] = (6.8, 3.2),
    font_med: float = 13,
    font_small: float = 11,
    legend_font: float = 10,
) -> None:
    metric_map = {
        "NONREG_LOW": "mean_reward_nonreg_low",
        "NONREG_MOD": "mean_reward_nonreg_mod",
        "REG_LOW": "mean_reward_reg_low",
        "REG_MOD": "mean_reward_reg_mod",
    }

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    reward_block = training_df.loc[training_df["scenario"] == scenario].copy()
    for group in DEFAULT_GROUP_ORDER:
        metric = metric_map[group]
        style = _group_style(group)
        pivot = reward_block.pivot(index="episode", columns="seed", values=metric).sort_index()
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=1).fillna(0.0)
        ax.plot(
            mean.index,
            mean.values,
            color=style["color"],
            linestyle=style["linestyle"],
            lw=2.2,
            label=style["label"],
        )
        ax.fill_between(
            mean.index,
            (mean - std).values,
            (mean + std).values,
            color=style["color"],
            alpha=0.12,
        )

    ax.set_xlabel("Episode step", fontsize=font_med)
    ax.set_ylabel("Rewards", fontsize=font_med)
    ax.grid(True, axis="y", alpha=0.22)
    ax.tick_params(axis="both", labelsize=font_small)
    ax.legend(fontsize=legend_font, loc="upper left", frameon=True)
    _savefig(outpath)


def plot_figure2_strategies_only(
    strategy_df: pd.DataFrame,
    *,
    scenario: str,
    outpath: Path,
    figsize: Tuple[float, float] = (6.2, 3.2),
    font_med: float = 13,
    font_small: float = 11,
    legend_font: float = 10,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    strategy_block = strategy_df.loc[strategy_df["scenario"] == scenario].copy()

    for group in DEFAULT_GROUP_ORDER:
        style = _group_style(group)
        group_block = strategy_block.loc[strategy_block["group"] == group].sort_values("local_step")
        if group_block.empty:
            continue

        agg = (
            group_block.groupby("local_step", as_index=False)
            .agg(
                mean_action_idx=("dominant_action_idx", "mean"),
                std_action_idx=("dominant_action_idx", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            )
            .sort_values("local_step")
        )

        ax.errorbar(
            agg["local_step"],
            agg["mean_action_idx"],
            yerr=agg["std_action_idx"],
            fmt=style["marker"],
            color=style["color"],
            linestyle="none",
            capsize=3.5,
            elinewidth=1.1,
            markersize=7,
            markeredgewidth=1.3,
            markerfacecolor=(style["color"] if style["marker"] == "o" else "none"),
            label=style["label"],
        )

    num_actions = int(strategy_block["dominant_action_idx"].max()) + 1 if not strategy_block.empty else 5
    ax.set_yticks(range(num_actions))
    ax.set_yticklabels([a_label(i) for i in range(num_actions)], fontsize=font_small)
    ax.set_xlabel("Simulation step", fontsize=font_med)
    ax.set_ylabel("Actions", fontsize=font_med)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=font_small)
    ax.legend(fontsize=legend_font, loc="upper right", frameon=True)
    _savefig(outpath)


def plot_training_rewards(training_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scenario in zip(axes, ["ON", "OFF"]):
        block = training_df.loc[training_df["scenario"] == scenario].copy()
        if block.empty:
            continue
        pivot = block.pivot(index="episode", columns="seed", values="total_reward").sort_index()
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1, ddof=1).fillna(0.0)

        ax.plot(mean.index, mean.values, color=SCENARIO_COLORS[scenario], lw=2.5, label=f"{scenario} mean")
        ax.fill_between(
            mean.index,
            (mean - std).values,
            (mean + std).values,
            color=SCENARIO_COLORS[scenario],
            alpha=0.2,
            label=f"{scenario} +/- 1 std",
        )
        ax.set_title(f"Policy {scenario}: training rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend()

    _savefig(outdir / "rewards_mean_std.png")


def plot_group_reward_summary(eval_summary_df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        "eval_mean_reward_nonreg_low",
        "eval_mean_reward_nonreg_mod",
        "eval_mean_reward_reg_low",
        "eval_mean_reward_reg_mod",
    ]
    labels = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, scenario in enumerate(["ON", "OFF"]):
        block = eval_summary_df.loc[eval_summary_df["scenario"] == scenario]
        means = [block[m].mean() for m in metrics]
        stds = [block[m].std(ddof=1) if len(block[m].dropna()) > 1 else 0.0 for m in metrics]
        offset = (-0.5 + idx) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=SCENARIO_COLORS[scenario],
            alpha=0.8,
            label=f"Policy {scenario}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Evaluation mean reward per group")
    ax.set_title("Grouped rewards across seeds")
    ax.legend()
    _savefig(outdir / "rewards_by_group_mean_std.png")


def plot_figure3_across_seeds(
    figure3_summary: Dict[str, Any],
    representative_artifacts: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    if "ON" not in representative_artifacts or "OFF" not in representative_artifacts:
        return

    art_on = representative_artifacts["ON"]
    art_off = representative_artifacts["OFF"]
    plot_policy_summary_comparison(
        env_on=art_on["env"],
        bh_on=art_on["bh_trace"],
        af_on=art_on["af_trace"],
        health_on=art_on["health_trace"],
        admin_on=art_on["admin_trace"],
        init_admin_on=art_on["init_admin"],
        init_trust_on=art_on["init_trust"],
        init_health_budget_on=art_on["init_health_budget"],
        init_social_budget_on=art_on["init_social_budget"],
        env_off=art_off["env"],
        bh_off=art_off["bh_trace"],
        af_off=art_off["af_trace"],
        health_off=art_off["health_trace"],
        admin_off=art_off["admin_trace"],
        init_admin_off=art_off["init_admin"],
        init_trust_off=art_off["init_trust"],
        init_health_budget_off=art_off["init_health_budget"],
        init_social_budget_off=art_off["init_social_budget"],
        summary_on=figure3_summary.get("ON"),
        summary_off=figure3_summary.get("OFF"),
        figsize=(16.8, 8.8),
        wspace=0.05,
        hspace=0.24,
        grid_width_ratio=0.95,
        right_width_ratio=1.85,
        title_on="Policy ON",
        title_off="Policy OFF",
        font_big=12,
        font_med=10.2,
        font_small=7.6,
        xlim=(0.0, 1.22),
        save_path=outdir / "figure3_across_seeds.png",
        show=False,
    )


def _draw_environment_panel(
    ax: plt.Axes,
    artifact: Dict[str, Any],
    *,
    title: str,
    show_legend: bool,
    font_big: float = 12,
    font_small: float = 8,
) -> None:
    env = artifact["env"]
    size = getattr(env, "size", 7)
    locs = getattr(getattr(env, "context", None), "locations", {}) or {}

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, size + 1))
    ax.set_yticks(np.arange(0, size + 1))
    ax.grid(True, color="0.25", linewidth=1.0, alpha=0.75)
    ax.tick_params(labelbottom=False, labelleft=False, length=2.5, color="0.55")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=font_big, pad=6)

    colour_map = {"PHC": "#d0d0ff", "ICU": "#7fa8ff", "SocialService": "#f0f0f0"}
    label_map = {"PHC": "PHC", "ICU": "ICU", "SocialService": "Social\nServices"}
    for name, info in locs.items():
        base = np.array(info["pos"])
        w, h = info.get("size", (1, 1))
        rect = plt.Rectangle(
            base,
            w,
            h,
            facecolor=colour_map.get(name, "#dddddd"),
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(base[0] + 0.08, base[1] + 0.34, label_map.get(name, name), fontsize=font_small, va="top", ha="left")

    if hasattr(env, "socserv_agents") and env.socserv_agents:
        n_sw = len(env.socserv_agents)
        for k, sw in enumerate(env.socserv_agents):
            x, y = sw.location
            ang = 2 * np.pi * (k / max(1, n_sw))
            dx = 0.08 * np.cos(ang)
            dy = 0.08 * np.sin(ang)
            ax.scatter(x + 0.5 + dx, y + 0.5 + dy, s=18, color="grey", edgecolors="none", zorder=2)

    for agent in env.possible_agents:
        idx = env.agent_name_mapping[agent]
        peh = env.peh_agents[idx]
        x, y = peh.location
        linestyle = "-" if peh.administrative_state == "registered" else (0, (3, 2))
        circle = plt.Circle(
            (x + 0.5, y + 0.5),
            radius=0.28,
            facecolor=health_to_color(peh.health_state, alpha=0.95),
            edgecolor="black",
            linewidth=1.45,
            linestyle=linestyle,
            zorder=3,
        )
        ax.add_patch(circle)

    if show_legend:
        handles = [
            Patch(facecolor=health_to_color(4.0, alpha=0.95), edgecolor="black", label="healthy"),
            Patch(facecolor=health_to_color(1.0, alpha=0.95), edgecolor="black", label="hospitalized"),
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="registered"),
            Line2D([0], [0], color="black", linestyle=(0, (3, 2)), linewidth=1.8, label="non-registered"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=font_small, frameon=True, borderpad=0.35, handlelength=2.1)


def _style_compact_axis(ax: plt.Axes, *, title: str, xlim: Tuple[float, float], font_med: float, font_small: float) -> None:
    ax.set_title(title, loc="left", fontsize=font_med, pad=6, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_facecolor("white")
    ax.grid(axis="x", color="0.90", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.9)
    ax.tick_params(axis="y", labelsize=font_small, length=0, pad=3)
    ax.tick_params(axis="x", labelsize=font_small - 0.2, colors="0.35", pad=3)


def _pct_pm_label(stats: Dict[str, float], prefix: str) -> str:
    return f"{prefix} {100.0 * stats['initial_mean']:.0f} +/- {100.0 * stats['initial_std']:.0f}%" if prefix == "I" else f"{prefix} {100.0 * stats['final_mean']:.0f} +/- {100.0 * stats['final_std']:.0f}%"


def _draw_cost_line(
    ax: plt.Axes,
    summary_stats: Dict[str, Any],
    *,
    scenario: str = "OFF",
    font_med: float,
    font_small: float | None = None,
) -> None:
    if font_small is None:
        font_small = font_med
    social = summary_stats["costs"]["social_service_spend"]
    health = summary_stats["costs"]["healthcare_spend"]
    social_text = rf"Social services = $-{social['mean']:.0f} \pm {social['std']:.0f}$ EUR"
    health_text = rf"Healthcare = $-{health['mean']:.0f} \pm {health['std']:.0f}$ EUR"
    health_color = "#c44e52" if scenario == "OFF" else "0.30"
    value_font = font_small * 1.08

    ax.axis("off")
    ax.text(0.00, 0.80, "Economic costs:", ha="left", va="center", fontsize=font_med, fontweight="bold", transform=ax.transAxes)
    ax.text(0.35, 0.43, social_text, ha="left", va="center", fontsize=value_font, color="0.30", transform=ax.transAxes)
    ax.text(0.35, 0.12, health_text, ha="left", va="center", fontsize=value_font, color=health_color, transform=ax.transAxes)


def _plot_splitbar_panel(
    ax: plt.Axes,
    rows: List[Tuple[str, Dict[str, float]]],
    *,
    title: str,
    final_color: str,
    xlim: Tuple[float, float],
    show_legend: bool,
    font_med: float,
    font_small: float,
) -> None:
    _style_compact_axis(ax, title=title, xlim=xlim, font_med=font_med, font_small=font_small)
    y_pos = np.arange(len(rows))[::-1].astype(float)

    init_handle = Patch(facecolor="white", edgecolor="0.45", hatch="////", label="Initial")
    final_handle = Patch(facecolor=final_color, edgecolor=final_color, alpha=0.85, label="Final")

    for y, (label, stats) in zip(y_pos, rows):
        ax.barh(
            y + 0.14,
            stats["initial_mean"],
            height=0.22,
            facecolor="white",
            edgecolor="0.45",
            linewidth=1.0,
            hatch="////",
            xerr=stats["initial_std"],
            error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": "0.45"},
            zorder=2,
        )
        ax.barh(
            y - 0.14,
            stats["final_mean"],
            height=0.22,
            color=final_color,
            edgecolor=final_color,
            alpha=0.85,
            xerr=stats["final_std"],
            error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": final_color},
            zorder=3,
        )
        ax.text(1.01, y + 0.14, f"I {100.0 * stats['initial_mean']:.0f} +/- {100.0 * stats['initial_std']:.0f}%", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=font_small - 0.2, color="0.45", clip_on=False)
        ax.text(1.01, y - 0.14, f"F {100.0 * stats['final_mean']:.0f} +/- {100.0 * stats['final_std']:.0f}%", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=font_small - 0.2, color=final_color, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _ in rows], fontsize=font_small)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if show_legend:
        ax.legend(handles=[init_handle, final_handle], fontsize=font_small, loc="lower right", frameon=True)


def _plot_groupcolor_panel(
    ax: plt.Axes,
    metric_groups: Dict[str, Dict[str, Dict[str, float]]],
    *,
    title: str,
    xlim: Tuple[float, float],
    show_group_legend: bool,
    font_med: float,
    font_small: float,
) -> None:
    _style_compact_axis(ax, title=title, xlim=xlim, font_med=font_med, font_small=font_small)
    metric_names = list(metric_groups.keys())
    centers = np.arange(len(metric_names))[::-1].astype(float)
    offsets = np.linspace(0.28, -0.28, len(DEFAULT_GROUP_ORDER))

    for center, metric_name in zip(centers, metric_names):
        stats_by_group = metric_groups[metric_name]
        for offset, group in zip(offsets, DEFAULT_GROUP_ORDER):
            stats = stats_by_group.get(group)
            if stats is None:
                continue
            color = GROUP_COLORS[group]
            ax.barh(
                center + offset + 0.035,
                stats["initial_mean"],
                height=0.06,
                facecolor="none",
                edgecolor=color,
                linewidth=1.1,
                hatch="////",
                xerr=stats["initial_std"],
                error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9, "ecolor": color},
                zorder=2,
            )
            ax.barh(
                center + offset - 0.035,
                stats["final_mean"],
                height=0.06,
                color=color,
                edgecolor=color,
                alpha=0.82,
                xerr=stats["final_std"],
                error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9, "ecolor": color},
                zorder=3,
            )

    ax.set_yticks(centers)
    ax.set_yticklabels(metric_names, fontsize=font_small)
    ax.set_ylim(-0.65, len(metric_names) - 0.35)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.text(0.995, 1.02, "outline = initial, filled = final", transform=ax.transAxes, ha="right", va="bottom", fontsize=font_small - 0.2, color="0.42")

    if show_group_legend:
        handles = [Line2D([0], [0], color=GROUP_COLORS[g], lw=4, label=GROUP_LABELS[g]) for g in DEFAULT_GROUP_ORDER]
        ax.legend(
            handles=handles,
            fontsize=font_small - 0.3,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.27),
            ncol=2,
            frameon=True,
            borderpad=0.35,
            columnspacing=1.2,
            handlelength=2.0,
        )


def _plot_overlay_panel(
    ax: plt.Axes,
    rows: List[Tuple[str, Dict[str, float]]],
    *,
    title: str,
    final_color: str,
    xlim: Tuple[float, float],
    show_legend: bool,
    font_med: float,
    font_small: float,
) -> None:
    _style_compact_axis(ax, title=title, xlim=xlim, font_med=font_med, font_small=font_small)
    y_pos = np.arange(len(rows))[::-1].astype(float)

    for y, (label, stats) in zip(y_pos, rows):
        ax.barh(
            y,
            stats["initial_mean"],
            height=0.36,
            facecolor="white",
            edgecolor="0.50",
            linewidth=1.0,
            hatch="////",
            xerr=stats["initial_std"],
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0, "ecolor": "0.50"},
            zorder=1,
        )
        ax.barh(
            y,
            stats["final_mean"],
            height=0.20,
            color=final_color,
            edgecolor=final_color,
            alpha=0.88,
            xerr=stats["final_std"],
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0, "ecolor": final_color},
            zorder=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _ in rows], fontsize=font_small)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=font_small)
    ax.text(0.995, 1.02, "initial = outline, final = filled overlay", transform=ax.transAxes, ha="right", va="bottom", fontsize=font_small - 0.2, color="0.42")

    if show_legend:
        handles = [
            Patch(facecolor="white", edgecolor="0.50", hatch="////", label="Initial"),
            Patch(facecolor=final_color, edgecolor=final_color, alpha=0.88, label="Final"),
        ]
        ax.legend(handles=handles, fontsize=font_small, loc="lower right", frameon=True)


def _plot_dumbbell_panel(
    ax: plt.Axes,
    rows: List[Tuple[str, Dict[str, float]]],
    *,
    title: str,
    final_color: str,
    xlim: Tuple[float, float],
    show_legend: bool,
    font_med: float,
    font_small: float,
) -> None:
    _style_compact_axis(ax, title=title, xlim=xlim, font_med=font_med, font_small=font_small)
    y_pos = np.arange(len(rows))[::-1].astype(float)

    for y, (label, stats) in zip(y_pos, rows):
        ax.hlines(y, stats["initial_mean"], stats["final_mean"], color="0.72", linewidth=2.0, zorder=1)
        ax.errorbar(
            stats["initial_mean"],
            y,
            xerr=stats["initial_std"],
            fmt="o",
            mfc="white",
            mec="0.45",
            ecolor="0.45",
            ms=8.0,
            mew=1.5,
            capsize=2.5,
            elinewidth=1.0,
            zorder=3,
        )
        ax.errorbar(
            stats["final_mean"],
            y,
            xerr=stats["final_std"],
            fmt="o",
            mfc=final_color,
            mec=final_color,
            ecolor=final_color,
            ms=5.6,
            mew=1.1,
            capsize=2.5,
            elinewidth=1.0,
            zorder=4,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _ in rows], fontsize=font_small)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=font_small)

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="0.45", markerfacecolor="white", markeredgecolor="0.45", lw=0, label="Initial"),
            Line2D([0], [0], marker="o", color=final_color, markerfacecolor=final_color, markeredgecolor=final_color, lw=0, label="Final"),
        ]
        ax.legend(
            handles=handles,
            fontsize=font_small,
            loc="center left",
            bbox_to_anchor=(0.02, 0.50),
            frameon=True,
            borderpad=0.35,
            handletextpad=0.5,
        )


def plot_figure3_splitbars(
    figure3_summary: Dict[str, Any],
    representative_artifacts: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    if "ON" not in representative_artifacts or "OFF" not in representative_artifacts:
        return

    fig = plt.figure(figsize=(16.0, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.75], hspace=0.24, wspace=0.12)
    xlim = (0.0, 1.05)

    for row_idx, scenario in enumerate(["ON", "OFF"]):
        artifact = representative_artifacts[scenario]
        summary_stats = figure3_summary[scenario]
        final_color = "#1b9e77" if scenario == "ON" else "#d95f02"

        ax_grid = fig.add_subplot(gs[row_idx, 0])
        _draw_environment_panel(ax_grid, artifact, title=f"Policy {scenario}", show_legend=(row_idx == 0))

        right = gs[row_idx, 1].subgridspec(3, 1, height_ratios=[1.18, 0.82, 0.18], hspace=0.52)
        ax_caps = fig.add_subplot(right[0, 0])
        ax_fun = fig.add_subplot(right[1, 0])
        ax_cost = fig.add_subplot(right[2, 0])

        cap_rows = [
            ("BH: all groups", summary_stats["capabilities"]["bodily_health_all"]),
            ("AF: non-reg + low", summary_stats["capabilities"]["affiliation_groups"]["NONREG_LOW"]),
            ("AF: non-reg + mod", summary_stats["capabilities"]["affiliation_groups"]["NONREG_MOD"]),
            ("AF: reg + low", summary_stats["capabilities"]["affiliation_groups"]["REG_LOW"]),
            ("AF: reg + mod", summary_stats["capabilities"]["affiliation_groups"]["REG_MOD"]),
        ]
        _plot_splitbar_panel(
            ax_caps,
            cap_rows,
            title="Capabilities (agents' actions)",
            final_color=final_color,
            xlim=xlim,
            show_legend=(row_idx == 0),
            font_med=10.3,
            font_small=7.8,
        )

        fun_rows = [
            ("Healthy", summary_stats["functionings"]["Healthy"]),
            ("Registered", summary_stats["functionings"]["Registered"]),
        ]
        _plot_splitbar_panel(
            ax_fun,
            fun_rows,
            title="Functionings (agents' state)",
            final_color=final_color,
            xlim=xlim,
            show_legend=False,
            font_med=10.3,
            font_small=7.8,
        )
        ax_fun.set_xlabel("Population (%)", fontsize=7.8)

        _draw_cost_line(ax_cost, summary_stats, font_med=10.1)

    _savefig(outdir / "figure3_across_seeds_splitbars.png")


def plot_figure3_groupcolors(
    figure3_summary: Dict[str, Any],
    representative_artifacts: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    if "ON" not in representative_artifacts or "OFF" not in representative_artifacts:
        return

    fig = plt.figure(figsize=(16.2, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.85], hspace=0.24, wspace=0.12)
    xlim = (0.0, 1.05)

    for row_idx, scenario in enumerate(["ON", "OFF"]):
        artifact = representative_artifacts[scenario]
        summary_stats = figure3_summary[scenario]

        ax_grid = fig.add_subplot(gs[row_idx, 0])
        _draw_environment_panel(ax_grid, artifact, title=f"Policy {scenario}", show_legend=(row_idx == 0))

        right = gs[row_idx, 1].subgridspec(3, 1, height_ratios=[1.18, 0.88, 0.18], hspace=0.52)
        ax_caps = fig.add_subplot(right[0, 0])
        ax_fun = fig.add_subplot(right[1, 0])
        ax_cost = fig.add_subplot(right[2, 0])

        _plot_groupcolor_panel(
            ax_caps,
            {
                "Bodily health": summary_stats["capabilities"]["bodily_health_groups"],
                "Affiliation": summary_stats["capabilities"]["affiliation_groups"],
            },
            title="Capabilities by group",
            xlim=xlim,
            show_group_legend=(row_idx == 0),
            font_med=10.2,
            font_small=7.6,
        )

        _plot_groupcolor_panel(
            ax_fun,
            {
                "Healthy": summary_stats["functionings_by_group"]["Healthy"],
                "Registered": summary_stats["functionings_by_group"]["Registered"],
            },
            title="Functionings by group",
            xlim=xlim,
            show_group_legend=False,
            font_med=10.2,
            font_small=7.6,
        )
        ax_fun.set_xlabel("Population (%)", fontsize=7.6)
        _draw_cost_line(ax_cost, summary_stats, font_med=10.0)

    _savefig(outdir / "figure3_across_seeds_groupcolors.png")


def plot_figure3_overlay(
    figure3_summary: Dict[str, Any],
    representative_artifacts: Dict[str, Dict[str, Any]],
    outdir: Path,
) -> None:
    if "ON" not in representative_artifacts or "OFF" not in representative_artifacts:
        return

    fig = plt.figure(figsize=(15.4, 8.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.68], hspace=0.24, wspace=0.12)
    xlim = (0.0, 1.05)

    for row_idx, scenario in enumerate(["ON", "OFF"]):
        artifact = representative_artifacts[scenario]
        summary_stats = figure3_summary[scenario]
        final_color = "#1b9e77" if scenario == "ON" else "#d95f02"

        ax_grid = fig.add_subplot(gs[row_idx, 0])
        _draw_environment_panel(ax_grid, artifact, title=f"Policy {scenario}", show_legend=(row_idx == 0))

        right = gs[row_idx, 1].subgridspec(3, 1, height_ratios=[1.12, 0.82, 0.18], hspace=0.46)
        ax_caps = fig.add_subplot(right[0, 0])
        ax_fun = fig.add_subplot(right[1, 0])
        ax_cost = fig.add_subplot(right[2, 0])

        cap_rows = [
            ("BH: all groups", summary_stats["capabilities"]["bodily_health_all"]),
            ("AF: non-reg + low", summary_stats["capabilities"]["affiliation_groups"]["NONREG_LOW"]),
            ("AF: non-reg + mod", summary_stats["capabilities"]["affiliation_groups"]["NONREG_MOD"]),
            ("AF: reg + low", summary_stats["capabilities"]["affiliation_groups"]["REG_LOW"]),
            ("AF: reg + mod", summary_stats["capabilities"]["affiliation_groups"]["REG_MOD"]),
        ]
        _plot_overlay_panel(
            ax_caps,
            cap_rows,
            title="Capabilities (agents' actions)",
            final_color=final_color,
            xlim=xlim,
            show_legend=(row_idx == 0),
            font_med=10.2,
            font_small=7.8,
        )

        fun_rows = [
            ("Healthy", summary_stats["functionings"]["Healthy"]),
            ("Registered", summary_stats["functionings"]["Registered"]),
        ]
        _plot_overlay_panel(
            ax_fun,
            fun_rows,
            title="Functionings (agents' state)",
            final_color=final_color,
            xlim=xlim,
            show_legend=False,
            font_med=10.2,
            font_small=7.8,
        )
        ax_fun.set_xlabel("Population (%)", fontsize=7.8)
        _draw_cost_line(ax_cost, summary_stats, font_med=10.0)

    _savefig(outdir / "figure3_across_seeds_overlay.png")


def plot_figure3_dumbbell(
    figure3_summary: Dict[str, Any],
    representative_artifacts: Dict[str, Dict[str, Any]],
    outdir: Path,
    *,
    filename: str = "figure3_across_seeds_dumbbell.png",
    figsize: Tuple[float, float] = (15.4, 8.0),
    font_big: float = 12,
    font_med: float = 10.2,
    font_small: float = 7.8,
) -> None:
    if "ON" not in representative_artifacts or "OFF" not in representative_artifacts:
        return

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.28], hspace=0.24, wspace=0.30)
    xlim = (0.0, 1.03)

    for row_idx, scenario in enumerate(["ON", "OFF"]):
        artifact = representative_artifacts[scenario]
        summary_stats = figure3_summary[scenario]
        final_color = "#1b9e77" if scenario == "ON" else "#d95f02"

        ax_grid = fig.add_subplot(gs[row_idx, 0])
        _draw_environment_panel(
            ax_grid,
            artifact,
            title=f"Policy {scenario}",
            show_legend=(row_idx == 0),
            font_big=font_big,
            font_small=font_small,
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
            font_med=font_med,
            font_small=font_small,
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
            font_med=font_med,
            font_small=font_small,
        )
        ax_fun.set_xlabel("Population (%)", fontsize=font_small, labelpad=5)
        _draw_cost_line(
            ax_cost,
            summary_stats,
            scenario=scenario,
            font_med=max(font_small, font_med - 0.2),
            font_small=font_small,
        )

    _savefig(outdir / filename)


def plot_eval_summary(aggregated_eval_df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        "eval_total_reward",
        "final_mean_health",
        "final_share_healthy",
        "healthcare_spend",
    ]
    pretty = {
        "eval_total_reward": "Eval total reward",
        "final_mean_health": "Final mean health",
        "final_share_healthy": "Final share healthy",
        "healthcare_spend": "Healthcare spend",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        block = aggregated_eval_df.loc[aggregated_eval_df["metric"] == metric].set_index("scenario")
        scenarios = ["ON", "OFF"]
        means = [block.loc[s, "mean"] for s in scenarios]
        stds = [block.loc[s, "std"] for s in scenarios]
        ax.bar(
            scenarios,
            means,
            yerr=stds,
            capsize=5,
            color=[SCENARIO_COLORS[s] for s in scenarios],
            alpha=0.85,
        )
        ax.set_title(pretty[metric])

    _savefig(outdir / "policy_on_off_summary_std.png")


def plot_strategy_consensus(strategy_summary_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, scenario in zip(axes, ["ON", "OFF"]):
        block = strategy_summary_df.loc[
            (strategy_summary_df["scenario"] == scenario) & (strategy_summary_df["local_step"] < 8)
        ].copy()
        if block.empty:
            continue
        for group in DEFAULT_GROUP_ORDER:
            group_block = block.loc[block["group"] == group].sort_values("local_step")
            if group_block.empty:
                continue
            ax.plot(
                group_block["local_step"],
                group_block["seed_support_fraction"],
                marker="o",
                lw=2,
                color=GROUP_COLORS[group],
                label=group,
            )
            for row in group_block.itertuples():
                ax.text(
                    row.local_step,
                    row.seed_support_fraction + 0.03,
                    row.top_action.replace("_", "\n"),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Seed support")
        ax.set_title(f"Policy {scenario}: dominant strategy consensus")
        ax.legend(ncol=4, fontsize=8)

    axes[-1].set_xlabel("Local step")
    _savefig(outdir / "strategy_consensus.png")


def generate_figures(outputs: Dict[str, Any], outdir: Path) -> None:
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_figure2_style(outputs["training_episodes"], outputs["strategy_by_seed"], scenario="OFF", outdir=figures_dir)
    plot_figure2_style(outputs["training_episodes"], outputs["strategy_by_seed"], scenario="ON", outdir=figures_dir)
    plot_figure3_across_seeds(outputs["figure3_summary"], outputs["representative_eval_artifacts"], figures_dir)
    plot_figure3_splitbars(outputs["figure3_summary"], outputs["representative_eval_artifacts"], figures_dir)
    plot_figure3_groupcolors(outputs["figure3_summary"], outputs["representative_eval_artifacts"], figures_dir)
    plot_figure3_overlay(outputs["figure3_summary"], outputs["representative_eval_artifacts"], figures_dir)
    plot_figure3_dumbbell(outputs["figure3_summary"], outputs["representative_eval_artifacts"], figures_dir)
    plot_training_rewards(outputs["training_episodes"], figures_dir)
    plot_group_reward_summary(outputs["eval_summary_by_seed"], figures_dir)
    plot_eval_summary(outputs["eval_summary_aggregated"], figures_dir)
    plot_strategy_consensus(outputs["strategy_summary_aggregated"], figures_dir)


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    profiles = load_profiles(cfg.num_peh)
    irl = load_irl(str(PROJECT_ROOT / "output" / "irl_calibration_results_raval.json"))

    all_training_rows: List[pd.DataFrame] = []
    all_training_summaries: List[Dict[str, Any]] = []
    all_eval_steps: List[pd.DataFrame] = []
    all_eval_agents: List[pd.DataFrame] = []
    all_eval_summaries: List[Dict[str, Any]] = []
    all_strategy_rows: List[pd.DataFrame] = []
    eval_artifacts: Dict[str, Dict[int, Dict[str, Any]]] = {"ON": {}, "OFF": {}}

    for seed in cfg.seeds:
        for policy_on in (True, False):
            q_tables, train_df, train_summary = train_policy(
                cfg,
                profiles,
                policy_on=policy_on,
                seed=seed,
                irl=irl,
                feature_cols=DEFAULT_FEATURE_COLS,
            )
            train_summary.update({"scenario": "ON" if policy_on else "OFF", "seed": seed})
            all_training_rows.append(train_df)
            all_training_summaries.append(train_summary)

            eval_steps_df, eval_agents_df, eval_summary, plot_artifact = evaluate_policy(
                cfg,
                profiles,
                q_tables,
                policy_on=policy_on,
                seed=seed,
            )
            all_eval_steps.append(eval_steps_df)
            all_eval_agents.append(eval_agents_df)
            all_eval_summaries.append(eval_summary)
            all_strategy_rows.append(dominant_strategy_by_seed(eval_steps_df))
            eval_artifacts["ON" if policy_on else "OFF"][seed] = plot_artifact

    training_df = pd.concat(all_training_rows, ignore_index=True)
    training_summary_df = pd.DataFrame(all_training_summaries).sort_values(["scenario", "seed"]).reset_index(drop=True)
    eval_steps_df = pd.concat(all_eval_steps, ignore_index=True)
    eval_agents_df = pd.concat(all_eval_agents, ignore_index=True)
    eval_summary_df = pd.DataFrame(all_eval_summaries).sort_values(["scenario", "seed"]).reset_index(drop=True)
    strategy_df = pd.concat(all_strategy_rows, ignore_index=True)

    training_metric_cols = [col for col in training_summary_df.columns if col not in {"scenario", "seed"}]
    eval_metric_cols = [col for col in eval_summary_df.columns if col not in {"scenario", "seed"}]

    aggregated_training_df = summarize_numeric_by_seed(
        training_summary_df,
        value_cols=training_metric_cols,
    )
    aggregated_eval_df = summarize_numeric_by_seed(
        eval_summary_df,
        value_cols=eval_metric_cols,
    )
    strategy_summary_df = summarize_strategy_across_seeds(strategy_df)
    policy_diff_df = paired_policy_difference(eval_summary_df)
    representative_artifacts, representative_seeds = select_representative_artifacts(
        eval_summary_df,
        eval_artifacts,
    )
    figure3_summary = build_policy_summary_across_seeds(
        cfg,
        eval_summary_df,
        eval_artifacts,
    )

    return {
        "training_episodes": training_df,
        "training_summary_by_seed": training_summary_df,
        "eval_steps": eval_steps_df,
        "eval_agents": eval_agents_df,
        "eval_summary_by_seed": eval_summary_df,
        "strategy_by_seed": strategy_df,
        "training_summary_aggregated": aggregated_training_df,
        "eval_summary_aggregated": aggregated_eval_df,
        "strategy_summary_aggregated": strategy_summary_df,
        "policy_difference": policy_diff_df,
        "representative_eval_artifacts": representative_artifacts,
        "representative_seeds": representative_seeds,
        "figure3_summary": figure3_summary,
    }


def save_outputs(cfg: ExperimentConfig, outputs: Dict[str, Any]) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = PROJECT_ROOT / "output" / "robustness_seeds" / f"run_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(outdir / f"{name}.csv", index=False)

    if "representative_seeds" in outputs:
        (outdir / "figure3_representative_seeds.json").write_text(
            json.dumps(outputs["representative_seeds"], indent=2),
            encoding="utf-8",
        )
    if "figure3_summary" in outputs:
        (outdir / "figure3_summary.json").write_text(
            json.dumps(outputs["figure3_summary"], indent=2),
            encoding="utf-8",
        )

    report_text = build_report(
        cfg,
        outputs["training_summary_by_seed"],
        outputs["eval_summary_by_seed"],
        outputs["eval_summary_aggregated"],
        outputs["strategy_summary_aggregated"],
    )
    (outdir / "summary.md").write_text(report_text, encoding="utf-8")

    summary_payload = {
        "config": asdict(cfg),
        "generated_at": datetime.now().isoformat(),
        "output_files": sorted(path.name for path in outdir.iterdir()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    generate_figures(outputs, outdir)

    summary_payload["output_files"] = sorted(
        str(path.relative_to(outdir)) for path in outdir.rglob("*") if path.is_file()
    )
    (outdir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return outdir


def print_terminal_summary(outputs: Dict[str, pd.DataFrame]) -> None:
    eval_summary = outputs["eval_summary_aggregated"]
    print("\nEvaluation mean +/- std across seeds")
    for scenario in ["ON", "OFF"]:
        block = eval_summary.loc[eval_summary["scenario"] == scenario].set_index("metric")
        if block.empty:
            continue
        print(f"\nPolicy {scenario}")
        for metric in [
            "eval_total_reward",
            "final_mean_health",
            "final_share_healthy",
            "final_share_registered",
            "healthcare_spend",
            "social_service_spend",
        ]:
            if metric in block.index:
                row = block.loc[metric]
                print(f"  {metric}: {row['mean']:.4f} +/- {row['std']:.4f}")


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
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
        seeds=tuple(args.seeds),
    )

    outputs = run_experiment(cfg)
    outdir = save_outputs(cfg, outputs)
    print_terminal_summary(outputs)
    print(f"\nSaved multi-seed results to: {outdir}")


if __name__ == "__main__":
    main()
