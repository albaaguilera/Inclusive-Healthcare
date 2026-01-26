# utils.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from environment.context import Context, ACTION_NAMES, Actions

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class EvalLogConfig:
    scenario_name: str
    policy_tag: str
    run_id: str
    outdir: str = "out_datasets"
    max_total_steps: int = 50_000

    # logging detail
    log_all_agents_each_step: bool = True   # snapshot de tots els PEH cada step global
    log_masks: bool = True                  # guarda mask (string) + action possible/impossible
    log_possible_impossible: bool = True
    log_sw: bool = True                     # snapshot SW cada step global

    # prints
    verbose_every: int = 0                  # 0 = silenciós; si 100 => print cada 100 steps

# ADDITIONAL HELPERS
feature_cols = [
    "prev_encounters",
    "health_state",
    "homelessness_duration",
    "history_of_abuse",
    "trust_building",
    "age",
    "income",
]

GROUP_ORDER = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]

# grouped label (a2,a4,a5) -> (bar{a1}, bar{a2}, a5)
GROUP_LABEL = r"$\bar{a}_1,\ \bar{a}_2,\ a_5$"
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))
# --- mida dels rombos i layout vertical (un sol lloc) ---
NODE_SCALE = 7.0  # 6–9 dóna bons resultats

LAYOUT_TIGHT = dict(
    y_group=0.76, y_a2=0.46, y_combo=0.34, y_a1=0.22,
    bh_y=0.93,    # ← dins el marc
    af_y=0.88,    # ← dins el marc
    ylim=(0.06, 1.02)
)
def load_irl(path="irl_calibration_results_raval.json"):
    with open(path, "r") as f:
        irl = json.load(f)
    irl["w_map"] = np.array(irl["w_map"], dtype=float)
    irl["X_mean"] = np.array(irl["X_mean"], dtype=float)
    irl["X_std"] = np.array(irl["X_std"], dtype=float)
    return irl

irl = load_irl("irl_calibration_results_raval.json")

def chebyshev_dist(a, b):
    return int(np.max(np.abs(np.array(a) - np.array(b))))
def plot_health_initial_final(health_state_trace):
    # take first and last recorded health per agent
    h0  = []
    hT  = []
    for ag, series in health_state_trace.items():
        if len(series) == 0:
            continue
        h0.append(series[0])
        hT.append(series[-1])

    h0 = np.array(h0, dtype=float)
    hT = np.array(hT, dtype=float)

    means = [h0.mean(), hT.mean()]
    stds  = [h0.std(),  hT.std()]

    plt.figure(figsize=(6,4))
    x = np.arange(2)
    plt.bar(x, means, yerr=stds, capsize=5, color=["tab:gray","tab:green"])
    plt.xticks(x, ["Initial", "Final"])
    plt.ylabel("Health state")
    #plt.title("Mean health (all agents)")
    plt.ylim(0.0, 4.2)
    plt.tight_layout()
    plt.show()

def plot_health_initial_final_by_group(health_state_trace, initial_admin_state, initial_trust_type):
    data = {g: {"h0": [], "hT": []} for g in GROUP_ORDER}

    for ag, series in health_state_trace.items():
        if len(series) == 0:
            continue
        g = group_of_agent(ag, initial_admin_state, initial_trust_type)  # <-- FIX
        data[g]["h0"].append(series[0])
        data[g]["hT"].append(series[-1])

    plt.figure(figsize=(8,4))
    width = 0.35
    x = np.arange(len(GROUP_ORDER))

    h0_means = [np.mean(data[g]["h0"]) if data[g]["h0"] else 0.0 for g in GROUP_ORDER]
    hT_means = [np.mean(data[g]["hT"]) if data[g]["hT"] else 0.0 for g in GROUP_ORDER]

    plt.bar(x - width/2, h0_means, width, label="Initial", color="tab:gray")
    plt.bar(x + width/2, hT_means, width, label="Final",  color="tab:green")

    plt.xticks(x, [GROUP_LABELS[g] for g in GROUP_ORDER], rotation=20, ha="right")
    plt.ylabel("Health state")
    #plt.title("Mean health by admin × trust")
    plt.ylim(0.0, 4.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_admin_initial_final(admin_state_trace):
    a0, aT = [], []
    for ag, series in admin_state_trace.items():
        if len(series) == 0:
            continue
        a0.append(series[0])
        aT.append(series[-1])

    a0 = np.array(a0, dtype=int)
    aT = np.array(aT, dtype=int)

    p0_reg = a0.mean()          # fraction registered
    pT_reg = aT.mean()

    plt.figure(figsize=(6,4))
    x = np.arange(2)
    plt.bar(x, [p0_reg, pT_reg], color=["tab:gray","tab:blue"])
    plt.xticks(x, ["Initial", "Final"])
    plt.ylim(0.0, 1.05)
    plt.ylabel("Share registered")
    #plt.title("Administrative state (all agents)")
    plt.tight_layout()
    plt.show()
def plot_admin_initial_final_by_group(admin_state_trace, initial_admin_state, initial_trust_type):
    data = {g: {"a0": [], "aT": []} for g in GROUP_ORDER}
    for ag, series in admin_state_trace.items():
        if len(series) == 0:
            continue
        g = group_of_agent(ag, initial_admin_state, initial_trust_type)  # <-- FIX
        data[g]["a0"].append(series[0])
        data[g]["aT"].append(series[-1])

    plt.figure(figsize=(8,4))
    width = 0.35
    x = np.arange(len(GROUP_ORDER))

    p0 = [np.mean(data[g]["a0"]) if data[g]["a0"] else 0.0 for g in GROUP_ORDER]
    pT = [np.mean(data[g]["aT"]) if data[g]["aT"] else 0.0 for g in GROUP_ORDER]

    plt.bar(x - width/2, p0, width, label="Initial", color="tab:gray")
    plt.bar(x + width/2, pT, width, label="Final",  color="tab:blue")

    plt.xticks(x, [GROUP_LABELS[g] for g in GROUP_ORDER], rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Share registered")
    #plt.title("Administrative state by admin × trust")
    plt.legend()
    plt.tight_layout()
    plt.show()
def _build_design_df(
    df: pd.DataFrame,
    feature_cols: List[str],
    fitted_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    Xdf = df[feature_cols].copy()

    if "gender" in Xdf.columns:
        Xdf["gender"] = pd.Categorical(Xdf["gender"].astype(str), categories=["female", "male", "non-binary"])
        Xdf = pd.get_dummies(Xdf, columns=["gender"], drop_first=True)

    Xdf = Xdf.fillna(0.0)

    if fitted_columns is not None:
        Xdf = Xdf.reindex(columns=fitted_columns, fill_value=0.0)

    return Xdf, list(Xdf.columns)
def build_exec_hist_from_steps(df_steps):
    """
    Reconstruct exec_hist from df_steps:
      exec_hist[agent] = [(t_local, action_idx, admin_state_after), ...]
    Requires df_steps columns: agent, action_idx, admin_state_after (or similar).
    """
    exec_hist = {}
    if df_steps is None or len(df_steps) == 0:
        return exec_hist

    # assegura ordre temporal
    sort_cols = [c for c in ["global_step", "t_global", "step"] if c in df_steps.columns]
    if sort_cols:
        df = df_steps.sort_values(sort_cols).copy()
    else:
        df = df_steps.copy()

    if "agent" not in df.columns:
        raise ValueError("df_steps must contain column 'agent'")
    if "action_idx" not in df.columns:
        raise ValueError("df_steps must contain column 'action_idx'")

    # local step per agent
    df["_t_local"] = df.groupby("agent").cumcount()

    admin_col = "admin_state_after" if "admin_state_after" in df.columns else None

    for ag, g in df.groupby("agent"):
        seq = []
        for _, row in g.iterrows():
            a_idx = int(row["action_idx"])
            tloc = int(row["_t_local"])
            admin = row[admin_col] if admin_col else None
            seq.append((tloc, a_idx, admin))
        exec_hist[ag] = seq

    return exec_hist

from matplotlib.lines import Line2D

def final_state_summary_figure(eval_env,
                               health_state_trace,
                               admin_state_trace,
                               initial_admin_state,
                               initial_trust_type):
    size = eval_env.size
    ctx  = eval_env.context

    fig, (ax_grid, ax_side) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    # reduce horizontal space between panels
    fig.subplots_adjust(wspace=0.08)

    # ---------- LEFT: GRID RENDER ----------
    ax = ax_grid
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, size+1))
    ax.set_yticks(np.arange(0, size+1))
    ax.grid(True, color="black", linewidth=1)
    ax.tick_params(labelbottom=False, labelleft=False)

    # static locations
    colour_map = {"PHC": "#d0d0ff", "ICU": "#7fa8ff", "SocialService": "#f0f0f0"}
    label_map  = {"PHC": "PHC", "ICU": "ICU", "SocialService": "Social\nServices"}

    for name, info in ctx.locations.items():
        base = np.array(info["pos"])
        w, h = info.get("size", (1, 1))
        col  = colour_map.get(name, "#dddddd")
        lab  = label_map.get(name, name[:3].upper())
        rect = plt.Rectangle(base, w, h, facecolor=col, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(base[0]+0.1, base[1]+0.4, lab,
                fontsize=10, va="top", ha="left")

    # PEH agents: large circles, color by final health,
    # linestyle dashed if final admin is non-registered
    for ag in eval_env.possible_agents:
        idx = eval_env.agent_name_mapping[ag]
        peh = eval_env.peh_agents[idx]
        x, y = peh.location

        # final health color
        face = health_to_color(peh.health_state)   # you already have this
        is_reg = (peh.administrative_state == "registered")
        grp = group_key_from_initial(initial_admin_state, initial_trust_type, ag)
        edge_col = GROUP_COLORS.get(grp, "black")
        ls = "-" if is_reg else "--"

        circ = plt.Circle((x+0.5, y+0.5),
                          radius=0.35,
                          facecolor=face,
                          edgecolor="black",
                          linewidth=2.2,
                          linestyle=ls)
        ax.add_patch(circ)
        group_handles = []

        #legend for groups
        # for g in ["NONREG_LOW","NONREG_MOD","REG_LOW","REG_MOD"]:
        #     group_handles.append(
        #         Line2D([0],[0],
        #             color=GROUP_COLORS[g],
        #             linestyle=GROUP_LS[g],
        #             linewidth=2.5,
        #             label=GROUP_LABELS[g])
        #     )
        # ax.legend(handles=group_handles, loc="upper left", fontsize=9, frameon=True)

    # Social workers: small grey dots (with jitter to avoid overlap)
    if hasattr(eval_env, "socserv_agents") and eval_env.socserv_agents:
        n_sw = len(eval_env.socserv_agents)

        for k, sw in enumerate(eval_env.socserv_agents):
            x, y = sw.location  # x=row, y=col (mateix criteri que PEH)

            jitter = 0.10  # puja/baixa si vols
            ang = 2 * np.pi * (k / max(1, n_sw))
            dx = jitter * np.cos(ang)
            dy = jitter * np.sin(ang)

            ax.scatter(
                x + 0.5 + dx,
                y + 0.5 + dy,
                s=30,
                color="grey",
                edgecolors="none",
                zorder=2,
            )

    ax.set_title("Final grid state")

    # ---------- RIGHT: SUMMARY PANEL ----------
    # Initial vs final health and admin (all agents)
    h0, hT, a0, aT = [], [], [], []
    for ag in eval_env.possible_agents:
        series_h = health_state_trace[ag]
        series_a = admin_state_trace[ag]
        if len(series_h) == 0:
            continue
        h0.append(series_h[0])
        hT.append(series_h[-1])
        a0.append(series_a[0])
        aT.append(series_a[-1])

    h0 = np.array(h0, float); hT = np.array(hT, float)
    a0 = np.array(a0, float); aT = np.array(aT, float)

    # bar positions
    ax = ax_side
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)

    # top: health
    x = np.array([0, 1])
    width = 0.35
    ax.bar(x - width/2, [h0.mean(), 0], width,
           color="tab:gray", label="Health initial")
    ax.bar(x + width/2, [0, hT.mean()], width,
           color="tab:green", label="Health final")

    # second set (admin share registered) on secondary y-axis
    ax2 = ax.twinx()
    p0_reg = a0.mean() if len(a0) else 0.0
    pT_reg = aT.mean() if len(aT) else 0.0
    ax2.plot([0,1], [p0_reg, pT_reg],
             color="tab:blue", marker="o",
             label="Share registered")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Initial", "Final"])
    ax.set_ylim(0, 4.2)
    ax.set_ylabel("Mean health")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Share registered")

    # build combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1+handles2, labels1+labels2, loc="upper center", fontsize=9)

    ax.set_title("Functionings summary")

    plt.tight_layout()
    plt.show()

def transform_raw_features_to_standardized(
    raw: Dict[str, Any],
    feature_cols: List[str],
    calib: Dict[str, Any]
) -> np.ndarray:
    one = pd.DataFrame([raw])
    Xdf, _ = _build_design_df(one, feature_cols, fitted_columns=calib["x_columns"])
    X = Xdf.to_numpy(dtype=float).reshape(-1)
    Xs = (X - calib["X_mean"]) / (calib["X_std"] + 1e-9)
    return Xs
from functools import lru_cache

@lru_cache(maxsize=None)
def cached_irl_potential(agent_id, prev_encounters, health_state, homelessness_duration, history_of_abuse, trust_building, age, income):
    raw = {
        "prev_encounters": float(prev_encounters),
        "health_state": float(health_state),
        "homelessness_duration": float(homelessness_duration),
        "history_of_abuse": float(history_of_abuse),
        "trust_building": float(trust_building),
        "age": float(age),
        "income": float(income),
    }
    Xs = transform_raw_features_to_standardized(raw, feature_cols, irl)
    w = np.array(irl["w_map"], dtype=float)
    z = float(w[0] + np.dot(w[1:], Xs))
    return z

def irl_potential_from_env(env, agent_id, irl, feature_cols):
    peh = env.peh_agents[env.agent_name_mapping[agent_id]]
    attrs = getattr(peh, "personal_attributes", {}) or {}
    prev_encounters = float(getattr(peh, "encounter_counter", 0))
    health_state = float(peh.health_state)
    homelessness_duration = float(attrs.get("homelessness_duration", 0))
    history_of_abuse = float(bool(attrs.get("history_of_abuse", False)))
    trust_building = float(getattr(peh, "non_engagement_counter", 0))
    age = float(attrs.get("age", 0))
    income = float(getattr(peh, "income", 0.0))
    return cached_irl_potential(
        agent_id,
        prev_encounters,
        health_state,
        homelessness_duration,
        history_of_abuse,
        trust_building,
        age,
        income
    )
def _ensure_ax(ax=None, figsize=(12,5), title=None):
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created = True
    else:
        fig = ax.figure
    if title:
        ax.set_title(title, pad=8)
    return fig, ax, created


def group_of_agent(
    ag: str,
    initial_admin_state: dict,
    initial_trust_type: dict,
) -> str:
    return group_key_from_initial(initial_admin_state, initial_trust_type, ag)

def get_state(obs, peh, max_enc=5, max_noneng=5):
    x, y = obs["agent"]
    h    = obs["health_state"]
    h_idx = int(round((h - peh.min_health) / peh.health_step))
    admin = int(peh.administrative_state == "non-registered")
    adj = int(obs.get("adjacent_to_social_agent", 0))

    enc = min(int(peh.encounter_counter), max_enc)
    noneng = min(int(peh.non_engagement_counter), max_noneng)

    return (x, y, h_idx, admin, adj, enc, noneng)

def _plot_split_rewards_4(ax, by_group, w=10):
    for k, series in by_group.items():
        ax.plot(series, color="#D9D9D9", alpha=0.9)

    styles = {
        "REG_LOW":    dict(ls="-",  label="REG+LOW"),
        "REG_MOD":    dict(ls="-",  label="REG+MOD"),
        "NONREG_LOW": dict(ls="--", label="NONREG+LOW"),
        "NONREG_MOD": dict(ls="--", label="NONREG+MOD"),
    }

    for k, series in by_group.items():
        if len(series) >= w:
            ma = moving_average(series, w)
            ax.plot(np.arange(len(ma)) + (w-1), ma, color="black", lw=2.2, **styles[k])

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("(B) Rewards by admin x trust")
    ax.legend(fontsize=12, ncol=2, loc="lower right")

def group_key(peh):
    admin = "REG" if peh.administrative_state == "registered" else "NONREG"
    trust = "LOW" if getattr(peh, "trust_type", "MODERATE_TRUST") == "LOW_TRUST" else "MOD"
    return f"{admin}_{trust}"

def moving_average(arr, w=10):
    return np.convolve(arr, np.ones(w)/w, mode="valid")

def _tuple_loc(x):
    try:
        return (int(x[0]), int(x[1]))
    except Exception:
        return None


def plot_rewards_by_group(episode_returns_by_group, w=10):
    """
    Plot training rewards by agent group (admin × trust).
    
    Args:
        episode_returns_by_group: dict with keys ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]
        w: window size for moving average (default 10)
    """
    order = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]
    plt.figure(figsize=(10,4))
    
    for k in order:
        series = episode_returns_by_group[k]
        col = GROUP_COLORS[k]
        ls = GROUP_LS[k]
        lab = GROUP_LABELS[k]
        
        # raw (transparent)
        plt.plot(series, color=col, alpha=0.18)
        
        # moving average (fort)
        if len(series) >= w:
            ma = moving_average(series, w)
            plt.plot(np.arange(len(ma)) + (w-1), ma, color=col, linestyle=ls, lw=2.5, label=lab)
    
    plt.xlabel("Episode step", fontsize=14)
    plt.ylabel("Rewards", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-1.5, 1.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# utils.py
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_optimal_strategy(
    exec_hist: dict,
    *,
    initial_admin_state: dict,
    initial_trust_type: dict,
    num_actions: int,
    engage_action_idx: int,
    group_order: list = None,
):
    """
    Mean greedy strategy (mode) by group over local step index.
    No globals: tot entra per paràmetres.
    """
    if group_order is None:
        group_order = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]

    by_group_time = defaultdict(lambda: defaultdict(list))
    engages_so_far = defaultdict(int)

    for ag, seq in exec_hist.items():
        g = group_of_agent(ag, initial_admin_state, initial_trust_type)
        for t_local, a_idx, _admin_status in seq:
            by_group_time[g][t_local].append((ag, int(a_idx)))

    plt.figure(figsize=(10, 4))

    for g in group_order:
        times = sorted(by_group_time[g].keys())
        if not times:
            continue

        col = GROUP_COLORS[g]
        label = GROUP_LABELS[g]
        seen_label = False

        for t in times:
            pairs = by_group_time[g][t]
            acts = [a for _, a in pairs]
            c = Counter(acts)
            a_mode, n_mode = c.most_common(1)[0]
            p_mode = n_mode / len(acts)

            for ag, a in pairs:
                if a == engage_action_idx:
                    engages_so_far[ag] += 1

            any_registered = any(engages_so_far[ag] >= 2 for ag, _ in pairs)
            marker = "o" if any_registered else "x"

            alpha = 0.15 + 0.85 * p_mode
            edge = "k" if marker == "o" else None
            plt.scatter(
                t, a_mode,
                color=col, marker=marker, s=90,
                alpha=alpha, edgecolors=edge,
                label=(label if not seen_label else None),
            )
            seen_label = True

        plt.plot(
            times,
            [Counter([a for _, a in by_group_time[g][t]]).most_common(1)[0][0] for t in times],
            color=col, alpha=0.35, linewidth=1.5
        )

    plt.yticks(range(num_actions), [a_label(i) for i in range(num_actions)], fontsize=14)
    plt.xlabel("Simulation step", fontsize=14)
    plt.ylabel("Actions", fontsize=14)
    #plt.title("Mean greedy strategy (mode) — admin × trust")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def final_state_comparison_figure(eval_env, health_state_trace, admin_state_trace, initial_admin_state,
                                  eval_env2, health_state_trace2, admin_state_trace2, initial_admin_state2):
    size = eval_env.size
    ctx  = eval_env.context

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10),
        gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    ax_grid1, ax_side1 = axes[0]
    ax_grid2, ax_side2 = axes[1]

    # ---------- helper to draw one scenario ----------
    def draw_panel(eval_env, health_state_trace, admin_state_trace,
                   initial_admin_state, ax_grid, ax_side, title_suffix):
        # grid
        ax = ax_grid
        ax.set_xlim(0, size); ax.set_ylim(0, size)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks(np.arange(0, size+1)); ax.set_yticks(np.arange(0, size+1))
        ax.grid(True, color="black", linewidth=1)
        ax.tick_params(labelbottom=False, labelleft=False)

        colour_map = {"PHC": "#d0d0ff", "ICU": "#7fa8ff", "SocialService": "#f0f0f0"}
        label_map  = {"PHC": "PHC", "ICU": "ICU", "SocialService": "Social\nServices"}

        for name, info in eval_env.context.locations.items():
            base = np.array(info["pos"])
            w, h = info.get("size", (1, 1))
            col  = colour_map.get(name, "#dddddd")
            lab  = label_map.get(name, name[:3].upper())
            rect = plt.Rectangle(base, w, h, facecolor=col, edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(base[0]+0.1, base[1]+0.4, lab, fontsize=10, va="top", ha="left")

        for ag in eval_env.possible_agents:
            idx = eval_env.agent_name_mapping[ag]
            peh = eval_env.peh_agents[idx]
            x, y = peh.location
            face = health_to_color(peh.health_state)
            is_reg = (peh.administrative_state == "registered")
            ls = "-" if is_reg else "--"
            circ = plt.Circle((x+0.5, y+0.5), radius=0.35,
                              facecolor=face, edgecolor="black",
                              linewidth=2.0, linestyle=ls)
            ax.add_patch(circ)

        if hasattr(eval_env, "socserv_agents"):
            for sw in eval_env.socserv_agents:
                x, y = sw.location
                ax.scatter(x+0.5, y+0.5, s=80, color="grey", edgecolors="none")

        ax.set_title(f"Final grid state{title_suffix}")

        # summary
        h0, hT, a0, aT = [], [], [], []
        for ag in eval_env.possible_agents:
            sh = health_state_trace[ag]
            sa = admin_state_trace[ag]
            if not sh:
                continue
            h0.append(sh[0]); hT.append(sh[-1])
            a0.append(sa[0]); aT.append(sa[-1])

        h0 = np.array(h0, float); hT = np.array(hT, float)
        a0 = np.array(a0, float); aT = np.array(aT, float)

        ax = ax_side
        ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.2)

        x = np.array([0, 1]); width = 0.35
        ax.bar(x - width/2, [h0.mean(), 0], width,
               color="tab:gray", label="Health initial")
        ax.bar(x + width/2, [0, hT.mean()], width,
               color="tab:green", label="Health final")

        ax2 = ax.twinx()
        p0_reg = a0.mean() if len(a0) else 0.0
        pT_reg = aT.mean() if len(aT) else 0.0
        ax2.plot([0, 1], [p0_reg, pT_reg],
                 color="tab:blue", marker="o",
                 label="Share registered")

        ax.set_xticks([0, 1]); ax.set_xticklabels(["Initial", "Final"])
        ax.set_ylim(0, 4.2); ax.set_ylabel("Mean health")
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("Share registered")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc="upper center", fontsize=8)
        ax.set_title(f"Functionings summary{title_suffix}")

    # top row: legal norm ON (universal_health=False)
    draw_panel(eval_env, health_state_trace, admin_state_trace,
               initial_admin_state, ax_grid1, ax_side1,
               title_suffix=" — legal norm ON")

    # bottom row: universal health (universal_health=True)
    draw_panel(eval_env2, health_state_trace2, admin_state_trace2,
               initial_admin_state2, ax_grid2, ax_side2,
               title_suffix=" — universal health")

    plt.tight_layout()
    plt.show()
import numpy as np

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    # si algun objecte no previst surt, que peti amb missatge clar
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def run_eval_and_log(
    env,
    q_tables,
    *,
    scenario_name: str,
    run_id: str,
    policy_tag: str,
    max_total_steps: int = 20000,
    verbose: bool = False,
):
    """
    Runs one eval episode in a PettingZoo AEC env and logs:
    - per-step: obs, state vars, chosen action, reward, shaping(optional if you pass it in), masks, possible/impossible actions
    - per-agent: init/final health/admin/trust, engagement/encounter counters
    - global: budgets + shelters over time
    Returns: (df_steps, df_agents, summary_dict)
    """

    step_rows = []
    # snapshot init per agent
    init_agents = {}

    # --- store init snapshot
    for ag in env.agents:
        idx = env.agent_name_mapping[ag]
        peh = env.peh_agents[idx]
        init_agents[ag] = dict(
            run_id=run_id,
            scenario=scenario_name,
            policy_tag=policy_tag,
            agent=ag,
            init_health=float(peh.health_state),
            init_admin=str(peh.administrative_state),
            init_trust=str(getattr(peh, "trust_type", "")),
            init_loc=_tuple_loc(peh.location),
            init_engagement=int(getattr(peh, "engagement_counter", 0)),
            init_encounters=int(getattr(peh, "encounter_counter", 0)),
            init_non_engagement=int(getattr(peh, "non_engagement_counter", 0)),
        )

    # --- safety cap (evita loops)
    total_steps = 0

    # IMPORTANT: assumeix que env ja està reset() i llest
    while env.agents:
        total_steps += 1
        if total_steps > max_total_steps:
            # truncation hard-stop
            for ag in list(env.agents):
                env.truncations[ag] = True
                env.dones[ag] = True
            env.agents = []
            break

        agent = env.agent_selection
        if agent is None:
            break

        idx = env.agent_name_mapping[agent]
        peh = env.peh_agents[idx]

        # --- pre info
        obs = env.observe(agent)
        pre_health = float(peh.health_state)
        pre_admin = str(peh.administrative_state)
        pre_loc = _tuple_loc(peh.location)
        pre_eng = int(getattr(peh, "engagement_counter", 0))
        pre_enc = int(getattr(peh, "encounter_counter", 0))
        pre_non = int(getattr(peh, "non_engagement_counter", 0))

        # --- mask + possible/impossible (usa el teu helper actual)
        mask = action_mask_from_classify(env, agent)
        possible = [Actions(i).name for i, m in enumerate(mask) if m == 1]
        impossible = [Actions(i).name for i, m in enumerate(mask) if m == 0]

        # --- choose action greedily with mask
        # IMPORTANT: si uses un altre selector (epsilon-greedy, etc), substitueix aquí
        state = get_state(obs, peh)
        action = masked_argmax(q_tables[agent][state], mask)

        # --- global context before step
        ctx_before = env.context.to_dict() if hasattr(env, "context") else {}
        shelters_before = getattr(env.context, "shelters_available", None)

        # --- STEP (només UNA vegada!)
        env.step(int(action))

        # --- post info
        post_obs = env.observe(agent) if agent in env.possible_agents else obs
        post_health = float(peh.health_state)
        post_admin = str(peh.administrative_state)
        post_loc = _tuple_loc(peh.location)
        post_eng = int(getattr(peh, "engagement_counter", 0))
        post_enc = int(getattr(peh, "encounter_counter", 0))
        post_non = int(getattr(peh, "non_engagement_counter", 0))

        r = float(env.rewards.get(agent, 0.0))
        done = bool(env.dones.get(agent, False))
        term = bool(env.terminations.get(agent, False))
        trunc = bool(env.truncations.get(agent, False))

        # --- capabilities/functionings snapshots (ja ho guardes a env.capabilities/env.functionings)
        caps = env.capabilities.get(agent, {})
        funs = env.functionings.get(agent, {})

        # --- social workers positions (per-step snapshot)
        sw_locs = []
        if hasattr(env, "socserv_agents"):
            sw_locs = [_tuple_loc(sw.location) for sw in env.socserv_agents]

        ctx_after = env.context.to_dict() if hasattr(env, "context") else {}
        shelters_after = getattr(env.context, "shelters_available", None)

        row = dict(
            run_id=run_id,
            scenario=scenario_name,
            policy_tag=policy_tag,

            t_global=int(getattr(env, "step_count", total_steps)),
            agent=agent,
            trust=str(getattr(peh, "trust_type", "")),

            pre_health=pre_health,
            post_health=post_health,
            pre_admin=pre_admin,
            post_admin=post_admin,
            pre_loc=pre_loc,
            post_loc=post_loc,

            action=int(action),
            action_name=Actions(int(action)).name,
            reward=r,

            done=done,
            term=term,
            trunc=trunc,

            pre_engagement=pre_eng,
            post_engagement=post_eng,
            pre_encounters=pre_enc,
            post_encounters=post_enc,
            pre_non_engagement=pre_non,
            post_non_engagement=post_non,

            mask=mask.astype(int).tolist(),
            possible_actions=possible,
            impossible_actions=impossible,

            capabilities=caps,
            functionings=funs,

            ctx_before=ctx_before,
            ctx_after=ctx_after,
            shelters_before=shelters_before,
            shelters_after=shelters_after,

            socserv_positions=sw_locs,
        )

        step_rows.append(row)

        if verbose:
            print(f"[{scenario_name}] t={row['t_global']:04d} {agent} a={row['action_name']} r={r:.3f} h:{pre_health}->{post_health} admin:{pre_admin}->{post_admin}")

    # --- final per-agent table
    agent_rows = []
    for ag in env.possible_agents:
        idx = env.agent_name_mapping[ag]
        peh = env.peh_agents[idx]
        base = init_agents.get(ag, dict(run_id=run_id, scenario=scenario_name, policy_tag=policy_tag, agent=ag))
        base.update(dict(
            final_health=float(peh.health_state),
            final_admin=str(peh.administrative_state),
            final_loc=_tuple_loc(peh.location),
            final_engagement=int(getattr(peh, "engagement_counter", 0)),
            final_encounters=int(getattr(peh, "encounter_counter", 0)),
            final_non_engagement=int(getattr(peh, "non_engagement_counter", 0)),
        ))
        agent_rows.append(base)

    summary = dict(
        run_id=run_id,
        scenario=scenario_name,
        policy_tag=policy_tag,
        final_context=env.context.to_dict() if hasattr(env, "context") else {},
        final_shelters=getattr(env.context, "shelters_available", None),
        n_steps=len(step_rows),
    )

    df_steps = pd.DataFrame(step_rows)
    df_agents = pd.DataFrame(agent_rows)
    return df_steps, df_agents, summary


def group_key_from_initial(initial_admin_state, initial_trust_type, ag):
    admin0 = initial_admin_state[ag]  # "registered" / "non-registered"
    trust0 = initial_trust_type[ag]   # "LOW_TRUST" / "MODERATE_TRUST"
    admin = "NONREG" if admin0 == "non-registered" else "REG"
    trust = "LOW" if trust0 == "LOW_TRUST" else "MOD"
    return f"{admin}_{trust}"

JITTER_Y = {
    "NONREG_LOW":  +0.10,
    "NONREG_MOD":  +0.03,
    "REG_LOW":     -0.03,
    "REG_MOD":     -0.10,
}

GROUP_COLORS = {
    "NONREG_MOD": "tab:orange",
    "NONREG_LOW": "tab:red",
    "REG_MOD":    "tab:cyan",
    "REG_LOW":    "tab:blue",
}

GROUP_LABELS = {
    "NONREG_MOD": "Non-registered + Moderate trust",
    "NONREG_LOW": "Non-registered + Low trust",
    "REG_MOD":    "Registered + Moderate trust",
    "REG_LOW":    "Registered + Low trust",
}

GROUP_LS = {
    "NONREG_MOD": "--",
    "NONREG_LOW": "--",
    "REG_MOD":    "-",
    "REG_LOW":    "-",
}


# Obtenir màscara d'accions (0/1)
def get_action_mask(env, agent, obs=None):
    n = env.action_space(agent).n
    if obs is not None and isinstance(obs, dict):
        if "action_mask" in obs and obs["action_mask"] is not None:
            m = np.array(obs["action_mask"]).astype(int).ravel()
            return m if m.size == n else np.ones(n, dtype=int)
        if "mask" in obs and obs["mask"] is not None:
            m = np.array(obs["mask"]).astype(int).ravel()
            return m if m.size == n else np.ones(n, dtype=int)
    if hasattr(env, "action_masks"):
        try:
            m = np.array(env.action_masks(agent)).astype(int).ravel()
            return m if m.size == n else np.ones(n, dtype=int)
        except Exception:
            pass
    return np.ones(n, dtype=int)
def action_mask_from_classify(env, agent):
    """Mask 1/0 a partir de _classify_actions (consistent amb el teu entorn)."""
    peh = env.peh_agents[env.agent_name_mapping[agent]]
    poss, _ = env._classify_actions(peh)
    n = env.action_space(agent).n
    mask = np.zeros(n, dtype=int)
    for act in poss:
        mask[act.value] = 1
    return mask

def masked_argmax(q_values, mask):
    """Argmax només entre accions possibles."""
    q = np.array(q_values, dtype=float).copy()
    q[mask == 0] = -1e9
    return int(np.argmax(q))

# Softer / pastel colors for health 1..4 (bad→good)
PASTEL_HEALTH = {
    1: (0.93, 0.36, 0.36),  # light red
    2: (1.00, 0.72, 0.72),  # light orange
    3: (0.90, 0.92, 0.55),  # light yellow-green
    4: (0.74, 0.90, 0.74),  # light green
}

def health_to_color(h, alpha=1.0):
    """Return a light/pastel RGB(A) color for health in [1..4]."""
    h = float(h)
    if h <= 1.0: base = PASTEL_HEALTH[1]
    elif h <= 2.0: base = PASTEL_HEALTH[2]
    elif h <= 3.0: base = PASTEL_HEALTH[3]
    else: base = PASTEL_HEALTH[4]
    # Matplotlib accepts RGB or RGBA; return RGBA only if alpha < 1
    return (*base, alpha) if alpha < 1 else base

# --- pretty TeX labels for actions ---
def a_label(i: int) -> str:
    # index: 0..4  ≡  a1..a5
    mapping = {
        0: r"$a_1$",
        1: r"$\bar{a}_1$",  # <- was a_2
        2: r"$a_2$",
        3: r"$\bar{a}_2$",  # <- was a_4
        4: r"$a_5$",
    }
    return mapping.get(i, rf"$a_{i+1}$")
# ------------------------------------------------------------
# Small helpers (masks / argmax)
# ------------------------------------------------------------
def _safe_action_space_n(env, agent: str) -> int:
    """
    Evita el warning de PettingZoo si env.action_space no està overridejat:
    - prova env.action_spaces dict
    - fallback env.action_space(agent).n
    """
    if hasattr(env, "action_spaces") and isinstance(getattr(env, "action_spaces"), dict):
        sp = env.action_spaces.get(agent, None)
        if sp is not None and hasattr(sp, "n"):
            return int(sp.n)
    return int(env.action_space(agent).n)


def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    """
    Argmax però ignorant accions amb mask=0.
    """
    q = np.asarray(q_values, dtype=float)
    m = np.asarray(mask, dtype=int)
    if q.shape[-1] != m.shape[-1]:
        # si hi ha desajust, millor no petar: assumeix tot possible
        return int(np.argmax(q))
    bad = (m == 0)
    if bad.all():
        return int(np.argmax(q))
    qq = q.copy()
    qq[bad] = -1e18
    return int(np.argmax(qq))


def action_mask_from_classify_env(env, agent: str, peh_obj=None) -> np.ndarray:
    """
    Construeix màscara 0/1 d'accions possibles.

    Prioritat:
    1) env.observe(agent) conté 'action_mask' o 'mask'
    2) env.action_masks(agent) si existeix
    3) env._classify_actions(peh) si existeix
    4) fallback: tot 1
    """
    n = _safe_action_space_n(env, agent)

    # (1) obs mask
    try:
        obs = env.observe(agent)
        if isinstance(obs, dict):
            for key in ("action_mask", "mask"):
                if key in obs and obs[key] is not None:
                    m = np.array(obs[key]).astype(int).ravel()
                    if m.size == n:
                        return m
    except Exception:
        pass

    # (2) env.action_masks
    if hasattr(env, "action_masks"):
        try:
            m = np.array(env.action_masks(agent)).astype(int).ravel()
            if m.size == n:
                return m
        except Exception:
            pass

    # (3) classify
    if hasattr(env, "_classify_actions") and peh_obj is not None:
        try:
            poss, _impos = env._classify_actions(peh_obj)
            m = np.zeros(n, dtype=int)
            for act in poss:
                # act pot ser enum amb .value
                v = int(getattr(act, "value", int(act)))
                if 0 <= v < n:
                    m[v] = 1
            return m
        except Exception:
            pass

    # (4) fallback
    return np.ones(n, dtype=int)


# ------------------------------------------------------------
# State extraction (PEH / SW)
# ------------------------------------------------------------
def _get_peh_obj(env, agent: str):
    idx = env.agent_name_mapping[agent]
    return env.peh_agents[idx]


def _group_key_from_initial(initial_admin: Dict[str, str],
                            initial_trust: Dict[str, str],
                            agent: str) -> str:
    admin = initial_admin.get(agent, "non-registered")
    trust = initial_trust.get(agent, "MODERATE_TRUST")
    admin_key = "REG" if admin == "registered" else "NONREG"
    trust_key = "LOW" if str(trust).upper().startswith("LOW") else "MOD"
    return f"{admin_key}_{trust_key}"


def _peh_snapshot(env, agent: str,
                  initial_admin: Dict[str, str],
                  initial_trust: Dict[str, str]) -> Dict[str, Any]:
    peh = _get_peh_obj(env, agent)
    x, y = tuple(peh.location) if hasattr(peh, "location") else (None, None)

    # --- capabilities (BH/AF) ---
    caps = {}
    if hasattr(env, "capabilities") and isinstance(env.capabilities, dict):
        caps = env.capabilities.get(agent, {}) or {}

    bh = caps.get("Bodily Health", np.nan)
    af = caps.get("Affiliation", np.nan)

    return {
        "agent": agent,
        "group": _group_key_from_initial(initial_admin, initial_trust, agent),
        "x": x, "y": y,
        "health": float(getattr(peh, "health_state", np.nan)),
        "admin": str(getattr(peh, "administrative_state", "")),
        "trust": str(getattr(peh, "trust_type", "")),
        "engagement_counter": int(getattr(peh, "engagement_counter", 0)),
        "non_engagement_counter": int(getattr(peh, "non_engagement_counter", 0)),

        # NEW:
        "cap_bh": float(bh) if bh is not None else np.nan,
        "cap_af": float(af) if af is not None else np.nan,
    }

def _sw_snapshot(env, sw_idx: int) -> Dict[str, Any]:
    sw = env.socserv_agents[sw_idx]
    x, y = tuple(sw.location) if hasattr(sw, "location") else (None, None)
    # assignment (si existeix)
    tgt = None
    if hasattr(env, "social_assignments"):
        try:
            tgt = int(env.social_assignments[sw_idx])
        except Exception:
            tgt = None
    return {
        "sw_id": sw_idx,
        "x": x, "y": y,
        "target_peh_idx": tgt,
    }


def _budgets_snapshot(env) -> Dict[str, Any]:
    ctx = env.context
    return {
        "healthcare_budget": float(getattr(ctx, "healthcare_budget", np.nan)),
        "social_service_budget": float(getattr(ctx, "social_service_budget", np.nan)),
    }


def _static_context_snapshot(env) -> Dict[str, Any]:
    """
    Cosetes que no canvien sovint: grid size, locations, etc.
    (ho guardem al meta.json)
    """
    ctx = env.context
    return {
        "grid_size": int(getattr(env, "size", -1)),
        "locations": getattr(ctx, "locations", {}),
    }


# ------------------------------------------------------------
# Main: run eval + rich log
# ------------------------------------------------------------
def run_eval_and_log_rich(
    env,
    *,
    q_tables: Dict[str, np.ndarray],
    get_state_fn,
    cfg: EvalLogConfig,
    reset_options: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executa una avaluació greedy (masked) i loga:
      - df_steps: 1 fila per step global (agent que actua)
      - df_agents: (si cfg.log_all_agents_each_step) snapshot de TOTS els agents per step global
      - df_sw: (si cfg.log_sw) snapshot de TOTS els SW per step global  -> dins meta["sw_rows"] per simplicitat
      - meta: info de run + inicials + resum final

    IMPORTANT: fa reset al principi sempre.
    """
    os.makedirs(cfg.outdir, exist_ok=True)

    # reset
    env.reset(options=reset_options or {})

    # inicials (admin/trust)
    initial_admin = {}
    initial_trust = {}
    for ag in env.possible_agents:
        peh = _get_peh_obj(env, ag)
        initial_admin[ag] = str(getattr(peh, "administrative_state", ""))
        initial_trust[ag] = str(getattr(peh, "trust_type", ""))

    # init obs/state per agents actius
    obs = {ag: env.observe(ag) for ag in env.agents}
    state = {ag: get_state_fn(obs[ag], _get_peh_obj(env, ag)) for ag in env.agents}

    step_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []
    sw_rows: List[Dict[str, Any]] = []

    # (clau!) log a t=0 abans de cap step
    t_global = 0
    if cfg.log_all_agents_each_step:
        b = _budgets_snapshot(env)
        for ag in env.possible_agents:
            row = {
                "run_id": cfg.run_id,
                "scenario": cfg.scenario_name,
                "policy_tag": cfg.policy_tag,
                "t_global": t_global,
                **b,
                **_peh_snapshot(env, ag, initial_admin, initial_trust),
            }
            agent_rows.append(row)

    if cfg.log_sw and hasattr(env, "socserv_agents"):
        b = _budgets_snapshot(env)
        for k in range(len(env.socserv_agents)):
            sw_rows.append({
                "run_id": cfg.run_id,
                "scenario": cfg.scenario_name,
                "policy_tag": cfg.policy_tag,
                "t_global": t_global,
                **b,
                **_sw_snapshot(env, k),
            })

    # loop
    total_steps = 0
    while env.agents and total_steps < cfg.max_total_steps:
        total_steps += 1
        t_global += 1

        agent = env.agent_selection

        # done -> step(None)
        if env.dones.get(agent, False):
            env.step(None)
            continue

        peh = _get_peh_obj(env, agent)

        # mask
        mask = action_mask_from_classify_env(env, agent, peh_obj=peh)

        # greedy action from Q
        s = state[agent]
        q = q_tables[agent][s]
        action = masked_argmax(q, mask)

        # extra info (possible/impossible)
        poss_names = None
        impos_names = None
        if cfg.log_possible_impossible and hasattr(env, "_classify_actions"):
            try:
                poss, impos = env._classify_actions(peh)
                poss_names = [getattr(a, "name", str(a)) for a in poss]
                impos_names = [getattr(a, "name", str(a)) for a in impos]
            except Exception:
                pass

        # step
        env.step(action)


        # reward & flags
        r = float(env.rewards.get(agent, 0.0))
        done = bool(env.dones.get(agent, False))
        term = bool(getattr(env, "terminations", {}).get(agent, False))
        trunc = bool(getattr(env, "truncations", {}).get(agent, False))

        # budgets after step
        b = _budgets_snapshot(env)

        # action name (si tens Actions enum al teu context, guarda també el nom)
        action_name = None
        try:
            # si és enum Actions, sovint tens env.context.Actions; aquí ho fem robust
            action_name = str(action)
        except Exception:
            action_name = None

        # df_steps row (1 per acció executada)
        row = {
            "run_id": cfg.run_id,
            "scenario": cfg.scenario_name,
            "policy_tag": cfg.policy_tag,
            "t_global": t_global,
            "agent": agent,
            "group": _group_key_from_initial(initial_admin, initial_trust, agent),
            "action": int(action),
            "action_name": action_name,
            "reward": r,
            "done": done,
            "term": term,
            "trunc": trunc,
            **b,
        }
        # caps_actor = {}
        # if hasattr(env, "capabilities") and isinstance(env.capabilities, dict):
        #     caps_actor = env.capabilities.get(agent, {}) or {}

        # row["actor_cap_bh"] = float(caps_actor.get("Bodily Health", np.nan))
        # row["actor_cap_af"] = float(caps_actor.get("Affiliation", np.nan))

        # afegeix estat PEH que ha actuat (després de step)
        row.update({f"actor_{k}": v for k, v in _peh_snapshot(env, agent, initial_admin, initial_trust).items()})
        # mask + poss/impos
        if cfg.log_masks:
            row["mask"] = ",".join(map(str, mask.tolist()))
        if cfg.log_possible_impossible:
            row["possible_actions"] = json.dumps(poss_names) if poss_names is not None else None
            row["impossible_actions"] = json.dumps(impos_names) if impos_names is not None else None

        step_rows.append(row)

        # snapshots de tots els agents (post-step)
        if cfg.log_all_agents_each_step:
            for ag in env.possible_agents:
                snap = {
                    "run_id": cfg.run_id,
                    "scenario": cfg.scenario_name,
                    "policy_tag": cfg.policy_tag,
                    "t_global": t_global,
                    **b,
                    **_peh_snapshot(env, ag, initial_admin, initial_trust),
                }
                agent_rows.append(snap)

        # snapshots SW (post-step)
        if cfg.log_sw and hasattr(env, "socserv_agents"):
            for k in range(len(env.socserv_agents)):
                sw_rows.append({
                    "run_id": cfg.run_id,
                    "scenario": cfg.scenario_name,
                    "policy_tag": cfg.policy_tag,
                    "t_global": t_global,
                    **b,
                    **_sw_snapshot(env, k),
                })

        # update state for that agent (post-step obs)
        try:
            new_obs = env.observe(agent)
            state[agent] = get_state_fn(new_obs, _get_peh_obj(env, agent))
        except Exception:
            pass

        if cfg.verbose_every and (total_steps % cfg.verbose_every == 0):
            print(f"[{cfg.policy_tag}] t={t_global} steps={total_steps} agent={agent} a={action} r={r:.3f}")

    # final summary
    final_states = {}
    for ag in env.possible_agents:
        peh = _get_peh_obj(env, ag)
        final_states[ag] = {
            "final_health": float(getattr(peh, "health_state", np.nan)),
            "final_admin": str(getattr(peh, "administrative_state", "")),
            "final_engagement_counter": int(getattr(peh, "engagement_counter", 0)),
            "final_non_engagement_counter": int(getattr(peh, "non_engagement_counter", 0)),
            "final_location": tuple(getattr(peh, "location", (None, None))),
        }

    meta = {
        "run_id": cfg.run_id,
        "scenario": cfg.scenario_name,
        "policy_tag": cfg.policy_tag,
        "total_steps": int(total_steps),
        "initial_admin": initial_admin,
        "initial_trust": initial_trust,
        "final_states": final_states,
        "static_context": _static_context_snapshot(env),
        "sw_rows_count": int(len(sw_rows)),
    }

    df_steps = pd.DataFrame(step_rows)
    df_agents = pd.DataFrame(agent_rows)
    df_sw = pd.DataFrame(sw_rows)

    # guardem df_sw dins meta també? millor separat en fitxer, però aquí el retornem via meta
    meta["_df_sw"] = df_sw  # <- només per conveniència; save_eval_artifacts el separarà

    return df_steps, df_agents, meta


# ------------------------------------------------------------
# Save / load artifacts
# ------------------------------------------------------------
def save_eval_artifacts(
    df_steps: pd.DataFrame,
    df_agents: pd.DataFrame,
    meta: Dict[str, Any],
    *,
    outdir: str,
    prefix: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # extreu df_sw si ve dins meta
    df_sw = meta.pop("_df_sw", None)

    steps_path = os.path.join(outdir, f"{prefix}_steps.csv")
    agents_path = os.path.join(outdir, f"{prefix}_agents.csv")
    meta_path = os.path.join(outdir, f"{prefix}_meta.json")
    sw_path = os.path.join(outdir, f"{prefix}_sw.csv")

    df_steps.to_csv(steps_path, index=False)
    df_agents.to_csv(agents_path, index=False)
    if df_sw is not None and isinstance(df_sw, pd.DataFrame):
        df_sw.to_csv(sw_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default = _json_default)

    print(f"✓ saved: {steps_path}")
    print(f"✓ saved: {agents_path}")
    if df_sw is not None:
        print(f"✓ saved: {sw_path}")
    print(f"✓ saved: {meta_path}")
import pandas as pd
import json
from pathlib import Path
import re
def load_eval_run(run_prefix: str, scenario_tag: str, data_dir=Path("out_datasets")):
    """
    scenario_tag: 'ON' o 'OFF'
    Espera fitxers:
      {run_prefix}_{scenario_tag}_steps.csv
      {run_prefix}_{scenario_tag}_agents.csv
      {run_prefix}_{scenario_tag}_meta.json
    """
    steps_path = data_dir / f"{run_prefix}_{scenario_tag}_steps.csv"
    agents_path = data_dir / f"{run_prefix}_{scenario_tag}_agents.csv"
    meta_path = data_dir / f"{run_prefix}_{scenario_tag}_meta.json"

    df_steps = pd.read_csv(steps_path) if steps_path.exists() else pd.DataFrame()
    df_agents = pd.read_csv(agents_path) if agents_path.exists() else pd.DataFrame()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    return df_steps, df_agents, meta


def load_eval_artifacts(outdir: str, prefix: str) -> Dict[str, Any]:
    steps_path = os.path.join(outdir, f"{prefix}_steps.csv")
    agents_path = os.path.join(outdir, f"{prefix}_agents.csv")
    sw_path = os.path.join(outdir, f"{prefix}_sw.csv")
    meta_path = os.path.join(outdir, f"{prefix}_meta.json")

    out = {
        "df_steps": pd.read_csv(steps_path) if os.path.exists(steps_path) else pd.DataFrame(),
        "df_agents": pd.read_csv(agents_path) if os.path.exists(agents_path) else pd.DataFrame(),
        "df_sw": pd.read_csv(sw_path) if os.path.exists(sw_path) else pd.DataFrame(),
        "meta": {},
    }
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            out["meta"] = json.load(f)
    return out
