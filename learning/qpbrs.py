# ─────────────────────────────────────────────────────────────
# SCRIPT: N AGENTS (REGISTERED / NON-REGISTERED and LOW / MODERATE TRUST)
# Q-learning + potential advice + avaluació greedy + plots
# + Capability Expansion agrupant a2,a4 i separant a1/a3 (dashed si impossibles)
# ─────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from environment.model import GridMAInequityEnv
from environment.context import Context, ACTION_NAMES, Actions

from learning.utils import (
    EvalLogConfig, run_eval_and_log, run_eval_and_log_rich, save_eval_artifacts, 
    _ensure_ax, group_of_agent, get_state, get_action_mask, _plot_split_rewards_4, 
    group_key, moving_average, _tuple_loc, cached_irl_potential, irl_potential_from_env,
    action_mask_from_classify, masked_argmax, plot_rewards_by_group, group_key_from_initial,
    a_label, health_to_color, final_state_comparison_figure, load_irl, chebyshev_dist, 
    plot_mean_optimal_strategy, transform_raw_features_to_standardized,
    plot_admin_initial_final, plot_admin_initial_final_by_group, plot_health_initial_final, plot_health_initial_final_by_group,
    final_state_summary_figure, plot_policy_summary_comparison, #save_results_summary,
    GROUP_COLORS, GROUP_LABEL, GROUP_LABELS, JITTER_Y, NODE_SCALE, LAYOUT_TIGHT, PASTEL_HEALTH
)

# ── paràmetres bàsics ────────────────────────────────────────
size = 7 # this should be fixed to 7 for the Raval scenario
num_peh = 8
num_sw = 15 # amb 8 PEH i 10 SW va bé (algun es pot quedar sense sw attention, 15SW ens assegurem que no), amb 16 PEH i 20 o 25SW ja no, politiques optimes es perden
#policy_inclusive_healthcare = False  # True: POLICY ON, False: POLICY OFF

import json
if num_peh == 4:
    with open("output/peh_sample4.json", "r") as f:
        profiles = json.load(f)
elif num_peh == 8:
    with open("output/peh_sample8.json", "r") as f:
        profiles = json.load(f)
elif num_peh == 16:
    with open("output/peh_sample16.json", "r") as f:
        profiles = json.load(f)
else:
    raise ValueError(f"num_peh={num_peh} no suportat. Usa 4, 8 o 16.")

def group_key_from_initial(initial_admin_state, initial_trust_type, ag):
    admin0 = initial_admin_state[ag]  # "registered" / "non-registered"
    trust0 = initial_trust_type[ag]   # "LOW_TRUST" / "MODERATE_TRUST"
    admin = "NONREG" if admin0 == "non-registered" else "REG"
    trust = "LOW" if trust0 == "LOW_TRUST" else "MOD"
    return f"{admin}_{trust}"

def health_to_color(h, alpha=1.0):
    """Return a light/pastel RGB(A) color for health in [1..4]."""
    h = float(h)
    if h <= 1.0: base = PASTEL_HEALTH[1]
    elif h <= 2.0: base = PASTEL_HEALTH[2]
    elif h <= 3.0: base = PASTEL_HEALTH[3]
    else: base = PASTEL_HEALTH[4]
    # Matplotlib accepts RGB or RGBA; return RGBA only if alpha < 1
    return (*base, alpha) if alpha < 1 else base


# SAVE ALL INFO IN DATASETS
import pandas as pd
import time, os 

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUTDIR = os.path.join("output", f"run_{RUN_ID}")
os.makedirs(OUTDIR, exist_ok=True)

FIGDIR = os.path.join(OUTDIR, "figures")
DATADIR = os.path.join(OUTDIR, "datasets")
MODELDIR = os.path.join(OUTDIR, "models")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

train_episode_rows_ON = []   # per episodi
eval_step_rows = []       # per step (avaluació)
eval_policyON_rows = []      # resum final per agent (avaluació)
eval_policyOFF_rows = []     # resum final per agent (avaluació)

# POTENTIAL-BASED ADVICE: 
irl = load_irl("output/irl_calibration_results_raval.json")
print(irl)
feature_cols = [
    "prev_encounters",
    "health_state",
    "homelessness_duration",
    "history_of_abuse",
    "trust_building",
    "age",
    "income",
]
# ============ POLICY ON TRAINING ============
print("\n" + "="*80)
print("TRAINING: POLICY INCLUSIVE HEALTHCARE = ON")
print("="*80)
# ── crea entorn ──────────────────────────────────────────────
# POLICY ON 
ctx = Context(grid_size=size)
ctx.set_scenario(policy_inclusive_healthcare=True)
env = GridMAInequityEnv(context=ctx, render_mode=None, size=size, num_peh=len(profiles), num_social_agents= num_sw, peh_profiles=profiles, max_steps = 150)
env.reset(options={"peh_profiles": profiles})# train always with the same profiles
#env.context.set_scenario(policy_inclusive_healthcare=True)

# DEBUG PRINTS
print(f"DEBUG: policy_inclusive_healthcare = {env.context.policy_inclusive_healthcare}")
if hasattr(env.peh_agents[0], 'can_access_healthcare'):
    print(f"DEBUG: Agent 0 can_access_healthcare = {env.peh_agents[0].can_access_healthcare}")


# ── Q-tables per agent ───────────────────────────────────────
MAX_ENC = 10
MAX_NONENG = 10

n_health = int((env.peh_agents[0].max_health - env.peh_agents[0].min_health) / env.peh_agents[0].health_step) + 1
q_tables_advice_ON = {}

for a in env.possible_agents:
    shape = (env.size, env.size, n_health, 2, 2, MAX_ENC+1, MAX_NONENG+1, env.action_space(a).n)
    q_tables_advice_ON[a] = np.zeros(shape)
# ── hiperparàmetres ──────────────────────────────────────────
episodes = 400
alpha, gamma = 0.2, 0.99
epsilon, eps_min, eps_decay = 0.1, 0.01, 0.995
max_steps = 100

episode_returns = []
episode_returns_by_group_ON = {
    "NONREG_LOW": [],
    "NONREG_MOD": [],
    "REG_LOW": [],
    "REG_MOD": [],
}

def group_of_agent(ag):
    admin0 = initial_admin_state[ag]              # "registered"/"non-registered"
    trust0 = initial_trust_type[ag]               # "LOW_TRUST"/"MODERATE_TRUST"
    admin = "NONREG" if admin0 == "non-registered" else "REG"
    trust = "LOW" if trust0 == "LOW_TRUST" else "MOD"
    return f"{admin}_{trust}"

# ── entrenament ──────────────────────────────────────────────
for ep in range(episodes):
    cached_irl_potential.cache_clear()   # reset cache each episode
    env.reset(options={"peh_profiles": profiles})# train always with the same profiles
    if ep == 0:
        print("\n[PBRS] potentials at episode 0")
        for a in env.agents:
            peh = env.peh_agents[env.agent_name_mapping[a]]
            print(a, getattr(peh,"trust_type","?"), "admin=", peh.administrative_state,
                "phi=", round(irl_potential_from_env(env, a, irl, feature_cols), 3))

    obs = {a: env.observe(a) for a in env.agents}
    state = {a: get_state(obs[a], env.peh_agents[env.agent_name_mapping[a]]) for a in env.agents}
    ep_ret = {a: 0.0 for a in env.agents}
    # snapshot del grup inicial (per episodi)
    init_group = {}
    for a in env.agents:
        peh = env.peh_agents[env.agent_name_mapping[a]]
        init_group[a] = group_key(peh)


    for _ in range(max_steps):
        agent = env.agent_selection
        if env.dones[agent]:
            env.step(None)
            if len(env.agents) == 0:
                break
            continue

        # ε-greedy
        # if np.random.rand() < epsilon:
        #     action = env.action_space(agent).sample()
        # else:
        #     action = int(np.argmax(q_tables_advice[agent][state[agent]]))

        mask = action_mask_from_classify(env, agent)
        feasible = np.flatnonzero(mask)

        # ε-greedy però només entre possibles
        if np.random.rand() < epsilon:
            if len(feasible) > 0:
                action = int(np.random.choice(feasible))
            else:
                action = env.action_space(agent).sample()  # fallback (no hauria de passar)
        else:
            action = masked_argmax(q_tables_advice_ON[agent][state[agent]], mask)

        ## POTENTIAL-BASED ADVICE
        # --- PBRS (potential-based reward shaping) ---
        beta = 0.02  # prova 0.05–0.15
        phi_s = irl_potential_from_env(env, agent, irl, feature_cols)

        # PAS (només una vegada!)
        env.step(action)

        base_r = env.rewards[agent]
        phi_sp = irl_potential_from_env(env, agent, irl, feature_cols)
        shaping = beta * (gamma * phi_sp - phi_s)
        individualreward = base_r + shaping

        new_obs = env.observe(agent)
        next_state = get_state(new_obs, env.peh_agents[env.agent_name_mapping[agent]])

        # Q-update (amb reward shaped)
        next_mask = action_mask_from_classify(env, agent)

        best_next = np.max(np.where(next_mask == 1,
                                    q_tables_advice_ON[agent][next_state],
                                    -1e9))
        s = state[agent]; a = action

        q_tables_advice_ON[agent][s + (a,)] += alpha * (
            individualreward + gamma * best_next - q_tables_advice_ON[agent][s + (a,)]
        )

        state[agent] = next_state
        ep_ret[agent] += individualreward

        if len(env.agents) == 0:
            break

    # bookkeeping
    episode_returns.append(sum(ep_ret.values()))
    epsilon = max(eps_min, epsilon * eps_decay)

    # rewards per grup (admin x trust) segons estat inicial de l'episodi
    group_totals = {k: 0.0 for k in episode_returns_by_group_ON.keys()}
    group_counts = {k: 0   for k in episode_returns_by_group_ON.keys()}

    for agent, ret in ep_ret.items():
        g = init_group[agent]
        group_totals[g] += ret
        group_counts[g] += 1

    for k in episode_returns_by_group_ON.keys():
        if group_counts[k] > 0:
            episode_returns_by_group_ON[k].append(group_totals[k] / group_counts[k])  # MEAN
        else:
            episode_returns_by_group_ON[k].append(0.0)
    
    train_episode_rows_ON.append({
            "run_id": RUN_ID,
            "scenario": "ON",
            "episode": ep,
            "epsilon": float(epsilon),
            "total_reward": float(sum(ep_ret.values())),
            "mean_reward_nonreg_low": float(episode_returns_by_group_ON["NONREG_LOW"][-1]),
            "mean_reward_nonreg_mod": float(episode_returns_by_group_ON["NONREG_MOD"][-1]),
            "mean_reward_reg_low": float(episode_returns_by_group_ON["REG_LOW"][-1]),
            "mean_reward_reg_mod": float(episode_returns_by_group_ON["REG_MOD"][-1]),
        })

plot_rewards_by_group(episode_returns_by_group_ON, w=10)

# POLICY ON
# ─────────────────────────────────────────────────────────────
#  GREEDY EVALUATION — CAPABILITIES & POLICY
# ─────────────────────────────────────────────────────────────
from collections import Counter
ctx = Context(grid_size=size)
ctx.set_scenario(policy_inclusive_healthcare=True)
eval_env = GridMAInequityEnv(context=ctx, render_mode=None, size=size, num_peh=len(profiles),num_social_agents= num_sw, peh_profiles=profiles)
eval_env.reset(options={"peh_profiles": profiles})
env.context.set_scenario(policy_inclusive_healthcare=True)

print(f"\nDEBUG [EVAL ON]: policy = {eval_env.context.policy_inclusive_healthcare}")

print("\n=== INIT CHECK (EVAL) ===")
# quants social agents assignats a cada PEH
assign_counts = Counter(eval_env.social_assignments.tolist())
for ag in eval_env.possible_agents:
    idx = eval_env.agent_name_mapping[ag]
    peh = eval_env.peh_agents[idx]
    print(f"{ag} admin={peh.administrative_state} trust={getattr(peh,'trust_type','?')} "
          f"loc={tuple(peh.location)} health={peh.health_state:.2f} "
          f"assigned_social_agents={assign_counts.get(idx,0)}")

nonreg_step_labels = []   # passos globals on actua el non-reg
global_step = 0
DEBUG_CAPS = False  # posa True per debug de capabilities

# tracking
bh_trace = {ag: [] for ag in eval_env.agents}
af_trace = {ag: [] for ag in eval_env.agents}
exec_hist = {ag: [] for ag in eval_env.agents}  # (t_local, action_idx, admin_status)
step_clock = {ag: 0 for ag in eval_env.agents}
health_state_trace = {ag: [] for ag in eval_env.possible_agents}
admin_state_trace = {ag: [] for ag in eval_env.possible_agents}

healthcare_budget_trace = []
social_service_budget_trace = []

initial_admin_state = {}
initial_trust_type = {}

for ag in eval_env.possible_agents:
    idx = eval_env.agent_name_mapping[ag]
    peh_agent = eval_env.peh_agents[idx]
    initial_admin_state[ag] = peh_agent.administrative_state
    initial_trust_type[ag] = getattr(peh_agent, "trust_type", "MODERATE_TRUST")

healthcare_budget_trace.append(eval_env.context.healthcare_budget)
social_service_budget_trace.append(eval_env.context.social_service_budget)

# màscares de capabilities (ens quedem amb el non-registered)
action_masks_trace = {ag: [] for ag in eval_env.possible_agents}

# sèries alineades al torn del non-registered (per evitar zeros)
bh_turns = []
af_turns = []
health_turns = []

# Troba un non-registered
nonreg_agent_id = None
for ag in eval_env.possible_agents:
    if initial_admin_state[ag] == "non-registered":
        nonreg_agent_id = ag
        break
reg_agent_id = next((ag for ag in eval_env.possible_agents if initial_admin_state[ag]=="registered"), None)
reg_bh_turns, reg_af_turns, reg_health_turns = [], [], []
reg_step_labels = []

# bucle d’avaluació greedy
obs = {ag: eval_env.observe(ag) for ag in eval_env.agents}
state = {
    ag: get_state(obs[ag], eval_env.peh_agents[eval_env.agent_name_mapping[ag]])
    for ag in eval_env.agents
}
print("Initial types:")
for ag in eval_env.possible_agents:
    idx = eval_env.agent_name_mapping[ag]
    peh = eval_env.peh_agents[idx]
    print(ag, peh.administrative_state, getattr(peh, "trust_type", "?"))
print("Agents active at start:", eval_env.agents)

MAX_EVAL_GLOBAL_STEPS = 500
eval_steps = 0
init_health_budget_on = float(eval_env.context.healthcare_budget)
init_social_budget_on = float(eval_env.context.social_service_budget)
epsilon = 0.1
while eval_env.agents and eval_steps < MAX_EVAL_GLOBAL_STEPS:
    eval_steps += 1
    agent = eval_env.agent_selection
    idx = eval_env.agent_name_mapping[agent]
    peh_agent = eval_env.peh_agents[idx]

    if eval_env.dones[agent]:
        eval_env.step(None)
        continue

    # captura màscara just abans de decidir (només per al non-registered)
    current_obs = eval_env.observe(agent)
    DEBUG_CAPS = True  # posa False per silenciar

    if agent == nonreg_agent_id:
        poss, impos = eval_env._classify_actions(peh_agent)
        n = eval_env.action_space(agent).n
        mask = np.zeros(n, dtype=int)
        for act in poss:
            mask[act.value] = 1
        action_masks_trace[agent].append(mask)
        if DEBUG_CAPS:
            print(f"[caps] tloc={step_clock[agent]:03d} {agent} | "
                f"possible={[a.name for a in poss]} | "
                f"impossible={[a.name for a in impos]} | mask={mask.tolist()}")
            
    # --- capability mask for the REGISTERED agent (mirror of non-reg) ---
    if agent == reg_agent_id:
        poss_r, impos_r = eval_env._classify_actions(peh_agent)
        n = eval_env.action_space(agent).n
        mask_r = np.zeros(n, dtype=int)
        for act in poss_r:
            mask_r[act.value] = 1
        action_masks_trace[agent].append(mask_r)
        if DEBUG_CAPS:
            print(f"[caps REG] tloc={step_clock[agent]:03d} {agent} | "
                f"possible={[a.name for a in poss_r]} | "
                f"impossible={[a.name for a in impos_r]} | "
                f"mask={mask_r.tolist()}")


    # acció greedy amb Q apresa
    # action = int(np.argmax(q_tables_advice[agent][state[agent]]))
    # eval_env.step(action)
    mask = action_mask_from_classify(eval_env, agent)
    action = masked_argmax(q_tables_advice_ON[agent][state[agent]], mask)
    if mask[action] == 0:
        print(f"[WARN] {agent} ha triat acció IMPOSSIBLE: {Actions(action).name} | mask={mask.tolist()}")

    eval_env.step(action)

    # prints per comprobar
    new_h = eval_env.peh_agents[eval_env.agent_name_mapping[agent]].health_state
    new_admin = eval_env.peh_agents[eval_env.agent_name_mapping[agent]].administrative_state
    r = eval_env.rewards[agent]
    d = eval_env.dones.get(agent, False)
    t = eval_env.terminations.get(agent, False)
    tr = eval_env.truncations.get(agent, False)

    print(f"[EVAL] step={eval_steps:04d} ag={agent} grp={group_key_from_initial(initial_admin_state, initial_trust_type, agent)} "
        f"a={Actions(action).name} h={new_h:.2f} admin={new_admin} r={r:.3f} done={d} term={t} trunc={tr}")

    eval_step_rows.append({
        "run_id": RUN_ID,
        "step": eval_steps,
        "agent": agent,
        "action": Actions(action).name,
        "health": new_h,
        "administrative_state": new_admin,
        "reward": r,
        "done": d,
        "termination": t,
        "truncation": tr,
    })
    # historial d'execució
    exec_hist[agent].append((step_clock[agent], action, peh_agent.administrative_state))
    step_clock[agent] += 1

    # capabilities del mateix agent que actua
    caps = eval_env.capabilities[agent]
    bh_trace[agent].append(caps.get("Bodily Health", 0.0))
    af_trace[agent].append(caps.get("Affiliation", 0.0))
    caps = eval_env.capabilities.get(agent, {}) if hasattr(eval_env, "capabilities") else {}
    eval_step_rows[-1]["cap_bh"] = float(caps.get("Bodily Health", np.nan))
    eval_step_rows[-1]["cap_af"] = float(caps.get("Affiliation", np.nan))

    if agent == nonreg_agent_id:
        bh_turns.append(bh_trace[agent][-1])
        af_turns.append(af_trace[agent][-1])
        health_turns.append(peh_agent.health_state)
    if agent == reg_agent_id:
        reg_bh_turns.append(bh_trace[agent][-1])
        reg_af_turns.append(af_trace[agent][-1])
        reg_health_turns.append(peh_agent.health_state)

    # funcionaments centrals i admin per TOTS els agents existents
    for ag in eval_env.agents:
        aobj = eval_env.peh_agents[eval_env.agent_name_mapping[ag]]
        health_state_trace[ag].append(aobj.health_state)
        admin_state_trace[ag].append(1 if aobj.administrative_state == "registered" else 0)

    # pressupostos de context
    healthcare_budget_trace.append(eval_env.context.healthcare_budget)
    social_service_budget_trace.append(eval_env.context.social_service_budget)

    # update state per l'agent actual
    obs_agent = eval_env.observe(agent)
    state[agent] = get_state(obs_agent, peh_agent)
    

run_id = f"run_{int(time.time())}"

# ─────────────────────────────────────────────────────────────
#  Estratègia òptima per step (sense "apply_for_shelter")
# ─────────────────────────────────────────────────────────────
ACTION_LUT = list(ACTION_NAMES.values())
colors  = {ag: col for ag, col in zip(exec_hist.keys(),
                                      plt.rcParams['axes.prop_cycle'].by_key()['color'])}
markers = {"registered": "o", "non-registered": "x"}

apply_for_shelter_idx = Actions.APPLY_AND_GET_SHELTER.value


plt.figure(figsize=(10, 4))

# Per evitar llegenda duplicada
seen = set()

for ag, seq in exec_hist.items():
    grp = group_key_from_initial(initial_admin_state, initial_trust_type, ag)
    col = GROUP_COLORS[grp]

    admin0 = initial_admin_state[ag]  # "registered" / "non-registered"
    marker = markers[admin0]          # "o" / "x"
    jy = JITTER_Y.get(grp, 0.0)

    for t, a_idx, _status in seq:
        if a_idx == apply_for_shelter_idx:
            continue

        # label només el primer cop que apareix el grup
        label = None
        if grp not in seen:
            label = GROUP_LABELS[grp]
            seen.add(grp)

        y = a_idx + jy

        if marker == "o":
            plt.scatter(t, y, color=col, marker=marker, s=90, edgecolors="k", label=label)
        else:
            plt.scatter(t, y, color=col, marker=marker, s=90, label=label)

num_actions = env.action_space(env.possible_agents[0]).n
plt.yticks(range(num_actions), [a_label(i) for i in range(num_actions)], fontsize=14)
plt.xlabel("Simulation step", fontsize=14)
plt.grid(True, axis="y", alpha=.3)
plt.legend(ncol=1, fontsize=13)
plt.tight_layout()
plt.show()

#### MEAN OPTIMAL STRATEGY 

from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

GROUP_ORDER = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]

num_actions = env.action_space(env.possible_agents[0]).n
plot_mean_optimal_strategy(
    exec_hist,
    initial_admin_state=initial_admin_state,
    initial_trust_type=initial_trust_type,
    num_actions=num_actions,
    engage_action_idx=Actions.ENGAGE_SOCIAL_SERVICES.value
)

# ─────────────────────────────────────────────────────────────
#  Capabilities BH / AF per agent (linies: reg= contínua, non-reg= discontínua)
# ─────────────────────────────────────────────────────────────
BH_COLOR = "tab:blue"
AF_COLOR = "tab:orange"

plt.figure(figsize=(10, 4))
for ag in bh_trace.keys():
    is_registered = (initial_admin_state[ag] == "registered")
    ls = "-" if is_registered else "--"
    plt.plot(bh_trace[ag], color=BH_COLOR, linestyle=ls)
    plt.plot(af_trace[ag], color=AF_COLOR, linestyle=ls)

plt.xlabel("Simulation step", fontsize=14)
plt.ylabel("Capability metric", fontsize=14)
custom_lines = [
    Line2D([0], [0], color='black', linestyle='-', label='Registered PEH'),
    Line2D([0], [0], color='black', linestyle='--', label='Non-registered PEH'),
    Line2D([0], [0], color=BH_COLOR, linestyle='-', label='Bodily health'),
    Line2D([0], [0], color=AF_COLOR, linestyle='-', label='Affiliation')
]
plt.legend(handles=custom_lines, fontsize=13, ncol=2); plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────
#  Functionings centrals (Health/Registration)
# ─────────────────────────────────────────────────────────────
def plot_central_functionings(health_state_trace, admin_state_trace, initial_admin_state):
    plt.figure(figsize=(10, 4))
    colors = {"health": "tab:blue", "admin": "tab:orange"}
    for ag in health_state_trace.keys():
        is_registered = (initial_admin_state[ag] == "registered")
        ls = "-" if is_registered else "--"
        plt.plot(np.array(health_state_trace[ag]) / 4, color=colors["health"], linestyle=ls)
        plt.plot(admin_state_trace[ag], color=colors["admin"], linestyle=ls)

    plt.xlabel("Simulation Step", fontsize=14)
    plt.ylabel("Functioning metric", fontsize=14)
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', label='Registered PEH'),
        Line2D([0], [0], color='black', linestyle='--', label='Non-registered PEH'),
        Line2D([0], [0], color=colors["health"], linestyle='-', label='Health'),
        Line2D([0], [0], color=colors["admin"], linestyle='-', label='Registration'),
    ]
    plt.legend(handles=custom_lines, fontsize=13, ncol=2)
    plt.tight_layout(); plt.show()

#plot_central_functionings(health_state_trace, admin_state_trace, initial_admin_state)

# plot_health_initial_final(health_state_trace)

# plot_health_initial_final_by_group(health_state_trace, initial_admin_state, initial_trust_type)

# plot_admin_initial_final(admin_state_trace)

# plot_admin_initial_final_by_group(admin_state_trace, initial_admin_state, initial_trust_type)

# final_state_summary_figure(eval_env, health_state_trace, admin_state_trace, initial_admin_state, initial_trust_type)

# =====================  SAVE OVERALL RESULTS IN OUTPUT  =====================
import os
from matplotlib.patches import Circle

def _plot_learning_curve(ax, ep_returns, w=10):
    # línia “raw” gris clar
    ax.plot(ep_returns, color="#D9D9D9", alpha=1.0)
    # mitjana mòbil NEGRA
    if len(ep_returns) >= w:
        ma = moving_average(ep_returns, w)
        ax.plot(np.arange(len(ma))+(w-1), ma, lw=2.2, color="black")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Aggregated reward")
    ax.set_title("(A) Learning curve")

def _plot_split_rewards(ax, reg, nonreg, w=10):
 # raw de tots dos: gris clar
    ax.plot(nonreg, color="#D9D9D9", alpha=1.0)
    ax.plot(reg,    color="#D9D9D9", alpha=1.0)
    # suau NEGRE: registered línia contínua, non-registered discontínua
    if len(nonreg) >= w:
        ax.plot(moving_average(nonreg, w), color="black", linestyle="--", lw=2.2)
    if len(reg) >= w:
        ax.plot(moving_average(reg, w),    color="black", linestyle="-",  lw=2.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("(B) Rewards by admin status")
    # llegenda “estètica”
    from matplotlib.lines import Line2D
    custom = [
        Line2D([0],[0], color="black", ls="-",  lw=2.2, label="Registered (smoothed)"),
        Line2D([0],[0], color="black", ls="--", lw=2.2, label="Non-registered (smoothed)")
    ]
    ax.legend(handles=custom, fontsize=9, loc="lower right")

def _plot_optimal_strategy(ax, exec_hist, num_actions, apply_for_shelter_idx,
                          initial_admin_state, initial_trust_type, exclude_idxs=None):
    if exclude_idxs is None:
        exclude_idxs = []
    # ensure apply_for_shelter_idx is excluded
    exclude = set(exclude_idxs) | {apply_for_shelter_idx}

    # both use 'o' marker; non-registered will get a dashed circular outline
    MARKER = "o"
    seen = set()


    for ag, seq in exec_hist.items():
        admin0 = initial_admin_state[ag]              # <- estat inicial
        grp = group_key_from_initial(initial_admin_state, initial_trust_type, ag)
        col = GROUP_COLORS[grp]

        for (t_global, a_idx, _status_ignored) in seq:
            if a_idx in exclude:
                continue

            label = None
            if grp not in seen:
                label = GROUP_LABELS[grp]
                seen.add(grp)

            jitter = 0.0

            y = a_idx + jitter
            # draw filled circle marker for both
            # for registered keep a solid black edge from scatter; for non-registered hide scatter edge and draw dashed patch
            if admin0 == "registered":
                ax.scatter(t_global, y, color=col, marker=MARKER, s=160, edgecolors="k", label=label, zorder=5)
            else:
                # draw filled marker without edge
                ax.scatter(t_global, y, color=col, marker=MARKER, s=160, edgecolors="none", label=label, zorder=5)
                # overlay a dashed circle as outline (data coordinates)
                circ = Circle((t_global, y), radius=0.25, facecolor="none",
                              edgecolor="k", linewidth=1.6, linestyle="--", zorder=6)
                ax.add_patch(circ)

    ax.set_yticks(range(num_actions))
    ax.set_yticklabels([a_label(i) for i in range(num_actions)], fontsize=14)
    ax.set_xlabel("Simulation step", fontsize=14)
    ax.grid(True, axis="y", alpha=.3)
    ax.legend(ncol=1, fontsize=13)
    plt.tight_layout()

def _plot_capability_traces(ax, bh_trace, af_trace, initial_admin_state):
    BH_COLOR = "tab:blue"; AF_COLOR = "tab:orange"
    for ag in bh_trace.keys():
        ls = "-" if initial_admin_state[ag] == "registered" else "--"
        ax.plot(bh_trace[ag], color=BH_COLOR, linestyle=ls)
        ax.plot(af_trace[ag], color=AF_COLOR, linestyle=ls)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Capability metric")
    ax.set_title("(D) Capabilities over time")
    custom = [
        Line2D([0],[0], color='black', ls='-', label='Registered'),
        Line2D([0],[0], color='black', ls='--', label='Non-registered'),
        Line2D([0],[0], color=BH_COLOR, ls='-', label='BH'),
        Line2D([0],[0], color=AF_COLOR, ls='-', label='AF'),
    ]
    ax.legend(handles=custom, fontsize=8, ncol=2)

def _plot_functionings(ax, health_state_trace, admin_state_trace, initial_admin_state):
    colors = {"health":"tab:blue","admin":"tab:orange"}
    for ag in health_state_trace.keys():
        ls = "-" if initial_admin_state[ag] == "registered" else "--"
        ax.plot(np.array(health_state_trace[ag])/4, color=colors["health"], linestyle=ls)
        ax.plot(admin_state_trace[ag], color=colors["admin"], linestyle=ls)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Functioning metric")
    ax.set_title("(E) Functionings over time")
    custom = [
        Line2D([0],[0], color='black', ls='-', label='Registered'),
        Line2D([0],[0], color='black', ls='--', label='Non-registered'),
        Line2D([0],[0], color=colors["health"], ls='-', label='Health'),
        Line2D([0],[0], color=colors["admin"], ls='-', label='Registration'),
    ]
    ax.legend(handles=custom, fontsize=8, ncol=2)

#save_results_summary()
# ========================================================================
# ============ POLICY OFF TRAINING ============
print("\n" + "="*80)
print("TRAINING: POLICY INCLUSIVE HEALTHCARE = OFF")
print("="*80)
policy_inclusive_healthcare = False  # desactiva la política
train_episodes_rows_OFF = []

# ── crea entorn ──────────────────────────────────────────────
ctx2 = Context(grid_size=size)
ctx2.set_scenario(policy_inclusive_healthcare=policy_inclusive_healthcare)
env2 = GridMAInequityEnv(context=ctx2, render_mode=None, size=size, num_peh=len(profiles), num_social_agents= num_sw, peh_profiles=profiles, max_steps = 300)
env2.reset(options={"peh_profiles": profiles})# train always with the same profiles
#env.context.set_scenario(policy_inclusive_healthcare=policy_inclusive_healthcare)

# DEBUG PRINTS
print(f"DEBUG: policy_inclusive_healthcare = {env2.context.policy_inclusive_healthcare}")
if hasattr(env2.peh_agents[0], 'can_access_healthcare'):
    print(f"DEBUG: Agent 0 can_access_healthcare = {env2.peh_agents[0].can_access_healthcare}")

# ── Q-tables per agent ───────────────────────────────────────

MAX_ENC = 10
MAX_NONENG = 10

n_health = int((env2.peh_agents[0].max_health - env2.peh_agents[0].min_health) / env2.peh_agents[0].health_step) + 1

q_tables_advice_OFF = {}

for a in env2.possible_agents:
    shape = (env2.size, env2.size, n_health, 2, 2, MAX_ENC+1, MAX_NONENG+1, env2.action_space(a).n)
    q_tables_advice_OFF[a] = np.zeros(shape)

# ── hiperparàmetres ──────────────────────────────────────────

episode_returns_OFF = []
episode_returns_by_group_OFF = {
    "NONREG_LOW": [],
    "NONREG_MOD": [],
    "REG_LOW": [],
    "REG_MOD": [],
}
epsilon = 0.1
# ── entrenament ──────────────────────────────────────────────
for ep in range(episodes):
    env2.reset(options={"peh_profiles": profiles})# train always with the same profiles
    obs = {a: env2.observe(a) for a in env2.agents}
    state = {a: get_state(obs[a], env2.peh_agents[env2.agent_name_mapping[a]]) for a in env2.agents}
    ep_ret = {a: 0.0 for a in env2.agents}
    # snapshot del grup inicial (per episodi)
    init_group = {}
    for a in env2.agents:
        peh = env2.peh_agents[env2.agent_name_mapping[a]]
        init_group[a] = group_key(peh)

    for _ in range(max_steps):
        agent = env2.agent_selection
        if env2.dones[agent]:
            env2.step(None)
            if len(env2.agents) == 0:
                break
            continue

        mask = action_mask_from_classify(env2, agent)
        feasible = np.flatnonzero(mask)

        # ε-greedy però només entre possibles
        if np.random.rand() < epsilon:
            if len(feasible) > 0:
                action = int(np.random.choice(feasible))
            else:
                action = env2.action_space(agent).sample()  # fallback (no hauria de passar)
        else:
            action = masked_argmax(q_tables_advice_OFF[agent][state[agent]], mask)

        # pas
        env2.step(action)

        # nou estat i reward
        new_obs = env2.observe(agent)
        individualreward = env2.rewards[agent]
        next_state = get_state(new_obs, env2.peh_agents[env2.agent_name_mapping[agent]])

        # Q-update
        s = state[agent]; a = action
        # best_next = np.max(q_tables_advice[agent][next_state])
        # q_tables_advice[agent][s + (a,)] += alpha * (individualreward + gamma * best_next - q_tables_advice[agent][s + (a,)])
        next_mask = action_mask_from_classify(env2, agent)

        best_next = np.max(np.where(next_mask == 1,
                                    q_tables_advice_OFF[agent][next_state],
                                    -1e9))
        q_tables_advice_OFF[agent][s + (a,)] += alpha * (
            individualreward + gamma * best_next - q_tables_advice_OFF[agent][s + (a,)]
        )
        state[agent] = next_state
        ep_ret[agent] += individualreward

        if len(env2.agents) == 0:
            break

    # bookkeeping
    episode_returns_OFF.append(sum(ep_ret.values()))
    epsilon = max(eps_min, epsilon * eps_decay)

    # rewards per grup (admin x trust) segons estat inicial de l'episodi
    # rewards per grup (admin x trust) segons estat inicial de l'episodi
    group_totals = {k: 0.0 for k in episode_returns_by_group_OFF.keys()}
    group_counts = {k: 0   for k in episode_returns_by_group_OFF.keys()}

    for agent, ret in ep_ret.items():
        g = init_group[agent]
        group_totals[g] += ret
        group_counts[g] += 1

    for k in episode_returns_by_group_OFF.keys():
        if group_counts[k] > 0:
            episode_returns_by_group_OFF[k].append(group_totals[k] / group_counts[k])  # MEAN
        else:
            episode_returns_by_group_OFF[k].append(0.0)
    
    train_episodes_rows_OFF.append({
        "run_id": RUN_ID,
        "scenario": "OFF",
        "episode": ep,
        "epsilon": float(epsilon),
        "total_reward": float(sum(ep_ret.values())),
        "mean_reward_nonreg_low": float(episode_returns_by_group_OFF["NONREG_LOW"][-1]),
        "mean_reward_nonreg_mod": float(episode_returns_by_group_OFF["NONREG_MOD"][-1]),
        "mean_reward_reg_low": float(episode_returns_by_group_OFF["REG_LOW"][-1]),
        "mean_reward_reg_mod": float(episode_returns_by_group_OFF["REG_MOD"][-1]),
    })
plot_rewards_by_group(episode_returns_by_group_OFF)

# ─────────────────────────────────────────────────────────────
#  GREEDY EVALUATION — CAPABILITIES & POLICY
# ────────────────────────────────────────────────────────────

ctx2 = Context(grid_size=size)
ctx2.set_scenario(policy_inclusive_healthcare=False)  # universal health = policy OFF
eval_env2 = GridMAInequityEnv(context=ctx2, render_mode=None, size=size, num_peh=len(profiles),num_social_agents= num_sw, peh_profiles=profiles)
eval_env2.reset(options={"peh_profiles": profiles})
ctx2.set_scenario(policy_inclusive_healthcare=False)

print(f"DEBUG [EVAL OFF]: policy = {eval_env2.context.policy_inclusive_healthcare}")
from collections import Counter

print("\n=== INIT CHECK (EVAL2) ===")
# quants social agents assignats a cada PEH
assign_counts2 = Counter(eval_env2.social_assignments.tolist())
for ag in eval_env2.possible_agents:
    idx = eval_env2.agent_name_mapping[ag]
    peh = eval_env2.peh_agents[idx]
    print(f"{ag} admin={peh.administrative_state} trust={getattr(peh,'trust_type','?')} "
          f"loc={tuple(peh.location)} health={peh.health_state:.2f} "
          f"assigned_social_agents={assign_counts2.get(idx,0)}")

nonreg_step_labels2 = []   # passos globals on actua el non-reg
global_step2 = 0
DEBUG_CAPS2 = False  # posa True per debug de capabilities

# tracking
bh_trace2 = {ag: [] for ag in eval_env2.agents}
af_trace2 = {ag: [] for ag in eval_env2.agents}
exec_hist2 = {ag: [] for ag in eval_env2.agents}  # (t_local, action_idx, admin_status)
step_clock2 = {ag: 0 for ag in eval_env2.agents}
health_state_trace2 = {ag: [] for ag in eval_env2.possible_agents}
admin_state_trace2 = {ag: [] for ag in eval_env2.possible_agents}

healthcare_budget_trace2 = []
social_service_budget_trace2 = []

initial_admin_state2 = {}
initial_trust_type2 = {}

for ag in eval_env2.possible_agents:
    idx = eval_env2.agent_name_mapping[ag]
    peh_agent = eval_env2.peh_agents[idx]
    initial_admin_state2[ag] = peh_agent.administrative_state
    initial_trust_type2[ag] = getattr(peh_agent, "trust_type", "MODERATE_TRUST")

healthcare_budget_trace2.append(eval_env2.context.healthcare_budget)
social_service_budget_trace2.append(eval_env2.context.social_service_budget)

# màscares de capabilities (ens quedem amb el non-registered)
action_masks_trace2 = {ag: [] for ag in eval_env2.possible_agents}

# sèries alineades al torn del non-registered (per evitar zeros)
bh_turns2 = []
af_turns2 = []
health_turns2 = []

# Troba un non-registered
nonreg_agent_id2 = None
for ag in eval_env2.possible_agents:
    if initial_admin_state2[ag] == "non-registered":
        nonreg_agent_id2 = ag
        break
reg_agent_id2 = next((ag for ag in eval_env2.possible_agents if initial_admin_state2[ag]=="registered"), None)
reg_bh_turns2, reg_af_turns2, reg_health_turns2 = [], [], []
reg_step_labels2 = []

# bucle d'avaluació greedy
obs2 = {ag: eval_env2.observe(ag) for ag in eval_env2.agents}
state2 = {
    ag: get_state(obs2[ag], eval_env2.peh_agents[eval_env2.agent_name_mapping[ag]])
    for ag in eval_env2.agents
}
print("Initial types:")
for ag in eval_env2.possible_agents:
    idx = eval_env2.agent_name_mapping[ag]
    peh = eval_env2.peh_agents[idx]
    print(ag, peh.administrative_state, getattr(peh, "trust_type", "?"))
print("Agents active at start:", eval_env2.agents)

MAX_EVAL_GLOBAL_STEPS2 = 1000
eval_steps2 = 0
init_health_budget_off = float(eval_env2.context.healthcare_budget)
init_social_budget_off = float(eval_env2.context.social_service_budget)

while eval_env2.agents and eval_steps2 < MAX_EVAL_GLOBAL_STEPS2:
    eval_steps2 += 1
    agent = eval_env2.agent_selection
    idx = eval_env2.agent_name_mapping[agent]
    peh_agent = eval_env2.peh_agents[idx]

    if eval_env2.dones[agent]:
        eval_env2.step(None)
        continue

    # captura màscara just abans de decidir (només per al non-registered)
    current_obs2 = eval_env2.observe(agent)
    DEBUG_CAPS2 = True  # posa False per silenciar

    if agent == nonreg_agent_id2:
        poss, impos = eval_env2._classify_actions(peh_agent)
        n = eval_env2.action_space(agent).n
        mask = np.zeros(n, dtype=int)
        for act in poss:
            mask[act.value] = 1
        action_masks_trace2[agent].append(mask)
        if DEBUG_CAPS2:
            print(f"[caps] tloc={step_clock2[agent]:03d} {agent} | "
                f"possible={[a.name for a in poss]} | "
                f"impossible={[a.name for a in impos]} | mask={mask.tolist()}")
            
    # --- capability mask for the REGISTERED agent (mirror of non-reg) ---
    if agent == reg_agent_id2:
        poss_r, impos_r = eval_env2._classify_actions(peh_agent)
        n = eval_env2.action_space(agent).n
        mask_r = np.zeros(n, dtype=int)
        for act in poss_r:
            mask_r[act.value] = 1
        action_masks_trace2[agent].append(mask_r)
        if DEBUG_CAPS2:
            print(f"[caps REG] tloc={step_clock2[agent]:03d} {agent} | "
                f"possible={[a.name for a in poss_r]} | "
                f"impossible={[a.name for a in impos_r]} | "
                f"mask={mask_r.tolist()}")


    # acció greedy amb Q apresa
    # action = int(np.argmax(q_tables_advice[agent][state2[agent]]))
    # eval_env2.step(action)
    mask = action_mask_from_classify(eval_env2, agent)
    action = masked_argmax(q_tables_advice_OFF[agent][state2[agent]], mask)
    if mask[action] == 0:
        print(f"[WARN] {agent} ha triat acció IMPOSSIBLE: {Actions(action).name} | mask={mask.tolist()}")

    eval_env2.step(action)

    # prints per comprobar
    new_h = eval_env2.peh_agents[eval_env2.agent_name_mapping[agent]].health_state
    new_admin = eval_env2.peh_agents[eval_env2.agent_name_mapping[agent]].administrative_state
    r = eval_env2.rewards[agent]
    d = eval_env2.dones.get(agent, False)
    t = eval_env2.terminations.get(agent, False)
    tr = eval_env2.truncations.get(agent, False)

    print(f"[EVAL2] step={eval_steps2:04d} ag={agent} grp={group_key_from_initial(initial_admin_state2, initial_trust_type2, agent)} "
        f"a={Actions(action).name} h={new_h:.2f} admin={new_admin} r={r:.3f} done={d} term={t} trunc={tr}")


    # historial d'execució
    exec_hist2[agent].append((step_clock2[agent], action, peh_agent.administrative_state))
    step_clock2[agent] += 1

    # capabilities del mateix agent que actua
    caps = eval_env2.capabilities[agent]
    bh_trace2[agent].append(caps.get("Bodily Health", 0.0))
    af_trace2[agent].append(caps.get("Affiliation", 0.0))
    if agent == nonreg_agent_id2:
        bh_turns2.append(bh_trace2[agent][-1])
        af_turns2.append(af_trace2[agent][-1])
        health_turns2.append(peh_agent.health_state)
    if agent == reg_agent_id2:
        reg_bh_turns2.append(bh_trace2[agent][-1])
        reg_af_turns2.append(af_trace2[agent][-1])
        reg_health_turns2.append(peh_agent.health_state)

    # funcionaments centrals i admin per TOTS els agents existents
    for ag in eval_env2.agents:
        aobj = eval_env2.peh_agents[eval_env2.agent_name_mapping[ag]]
        health_state_trace2[ag].append(aobj.health_state)
        admin_state_trace2[ag].append(1 if aobj.administrative_state == "registered" else 0)

    # pressupostos de context
    healthcare_budget_trace2.append(eval_env2.context.healthcare_budget)
    social_service_budget_trace2.append(eval_env2.context.social_service_budget)

    # update state per l'agent actual
    obs_agent2 = eval_env2.observe(agent)
    state2[agent] = get_state(obs_agent2, peh_agent)

print("\n=== FINAL STATES (EVAL2) ===")
for ag in eval_env2.possible_agents:
    idx = eval_env2.agent_name_mapping[ag]
    peh = eval_env2.peh_agents[idx]
    last_action = exec_hist2[ag][-1][1] if len(exec_hist2[ag]) else None
    print(f"{ag} init={group_key_from_initial(initial_admin_state2, initial_trust_type2, ag)} "
          f"final_h={peh.health_state:.2f} final_admin={peh.administrative_state} "
          f"done={eval_env2.dones.get(ag)} term={eval_env2.terminations.get(ag)} trunc={eval_env2.truncations.get(ag)} "
          f"last_action={(Actions(last_action).name if last_action is not None else None)}")

num_actions2 = eval_env2.action_space(eval_env2.possible_agents[0]).n
plot_mean_optimal_strategy(
    exec_hist2,
    initial_admin_state=initial_admin_state2,
    initial_trust_type=initial_trust_type2,
    num_actions=num_actions2,
    engage_action_idx=Actions.ENGAGE_SOCIAL_SERVICES.value,
)
plot_health_initial_final(health_state_trace2)
plot_health_initial_final_by_group(health_state_trace2, initial_admin_state2, initial_trust_type2)
plot_admin_initial_final(admin_state_trace2)
plot_admin_initial_final_by_group(admin_state_trace2, initial_admin_state2, initial_trust_type2)
final_state_summary_figure(eval_env2, health_state_trace2, admin_state_trace2, initial_admin_state2, initial_trust_type2)

# IMPORTANT: crea un env nou per log (no reutilitzis eval_env ja consumit)
ctx_log = Context(grid_size=size)
ctx_log.set_scenario(policy_inclusive_healthcare=False)
eval_env_log = GridMAInequityEnv(
    context=ctx_log, render_mode=None, size=size,
    num_peh=len(profiles), num_social_agents=num_sw, peh_profiles=profiles
)
eval_env_log.reset(options={"peh_profiles": profiles})

# (A) Guarda la policy per l'escenari OFF (universal health)
eval_policyOFF = {
    "policy_tag": "greedy_masked",
    "q_tables_ref": "q_tables_advice_OFF",
    "notes": "Universal health evaluation policy (masked greedy over Q)",
}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# assumes you already import these from utils:
# - group_key_from_initial
# - GROUP_COLORS, GROUP_LABELS, GROUP_LS
# - health_to_color

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Path

# Necessites aquestes globals (com ja tens):
# GROUP_COLORS, GROUP_LABELS, health_to_color
soft_green = health_to_color(4.0, alpha=0.85)   # like Functionings good
soft_red   = health_to_color(1.0, alpha=0.95)   # bad


plot_policy_summary_comparison(
    env_on=eval_env,
    bh_on=bh_trace,
    af_on=af_trace,
    health_on=health_state_trace,
    admin_on=admin_state_trace,
    init_admin_on=initial_admin_state,
    init_trust_on=initial_trust_type,
    init_health_budget_on=init_health_budget_on,
    init_social_budget_on=init_social_budget_on,

    env_off=eval_env2,
    bh_off=bh_trace2,
    af_off=af_trace2,
    health_off=health_state_trace2,
    admin_off=admin_state_trace2,
    init_admin_off=initial_admin_state2,
    init_trust_off=initial_trust_type2,
    init_health_budget_off=init_health_budget_off,
    init_social_budget_off=init_social_budget_off,

    healthy_threshold=3.0,
    figsize=(18, 12),
)

#final_state_comparison_figure(eval_env, health_state_trace, admin_state_trace, initial_admin_state,
                             # eval_env2, health_state_trace2, admin_state_trace2, initial_admin_state2)

pd.DataFrame(train_episode_rows_ON).to_csv(os.path.join(DATADIR, "training_episodes_ON.csv"), index=False)
pd.DataFrame(train_episodes_rows_OFF).to_csv(os.path.join(DATADIR, "training_episodes_OFF.csv"), index=False)
pd.DataFrame(eval_step_rows).to_csv(os.path.join(DATADIR, "eval_steps.csv"), index=False)
pd.DataFrame(eval_policyON_rows).to_csv(os.path.join(DATADIR, "eval_policyON_agents.csv"), index=False)
pd.DataFrame(eval_policyOFF_rows).to_csv(os.path.join(DATADIR, "eval_policyOFF_agents.csv"), index=False)

print("Saved datasets to:", DATADIR)

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUTDIR = os.path.join("out_models", f"run_{RUN_ID}")
os.makedirs(OUTDIR, exist_ok=True)

np.save(os.path.join(MODELDIR, "q_tables_advice_ON.npy"),  q_tables_advice_ON,  allow_pickle=True)
np.save(os.path.join(MODELDIR, "q_tables_advice_OFF.npy"), q_tables_advice_OFF, allow_pickle=True)

print("Saved Q-tables to:", MODELDIR)

OUTDIR = "out_datasets"
run_id = "run_001"   # o timestamp

# -------------------------
# EVAL POLICY ON (ex: legal norm ON)
# -------------------------
ctx_on = Context(grid_size=size)
ctx_on.set_scenario(policy_inclusive_healthcare=True)   # <-- segons el teu criteri "ON"

eval_env_on = GridMAInequityEnv(
    context=ctx_on, render_mode=None, size=size,
    num_peh=len(profiles), num_social_agents=num_sw, peh_profiles=profiles
)

cfg_on = EvalLogConfig(
    scenario_name="legal_norm_ON",
    policy_tag="eval_policyON",
    run_id=run_id,
    outdir=OUTDIR,
    max_total_steps=50_000,
    log_all_agents_each_step=True,
    log_masks=True,
    log_possible_impossible=True,
    log_sw=True,
    verbose_every=0
)

df_steps_on, df_agents_on, meta_on = run_eval_and_log_rich(
    eval_env_on,
    q_tables=q_tables_advice_ON,            # <-- posa el teu Q-table "ON"
    get_state_fn=get_state,
    cfg=cfg_on,
    reset_options={"peh_profiles": profiles},
)

save_eval_artifacts(df_steps_on, df_agents_on, meta_on, outdir=OUTDIR, prefix=f"{run_id}_ON")

# -------------------------
# EVAL POLICY OFF 
# -------------------------
ctx_off = Context(grid_size=size)
ctx_off.set_scenario(policy_inclusive_healthcare=False)   # <-- "OFF" 

eval_env_off = GridMAInequityEnv(
    context=ctx_off, render_mode=None, size=size,
    num_peh=len(profiles), num_social_agents=num_sw, peh_profiles=profiles
)

cfg_off = EvalLogConfig(
    scenario_name="policy_inclusive_healthcare",
    policy_tag="eval_policyOFF",
    run_id=run_id,
    outdir=OUTDIR,
    max_total_steps=50_000,
    log_all_agents_each_step=True,
    log_masks=True,
    log_possible_impossible=True,
    log_sw=True,
    verbose_every=0
)

df_steps_off, df_agents_off, meta_off = run_eval_and_log_rich(
    eval_env_off,
    q_tables=q_tables_advice_OFF,           # <-- posa el teu Q-table "OFF"
    get_state_fn=get_state,
    cfg=cfg_off,
    reset_options={"peh_profiles": profiles},
)

save_eval_artifacts(df_steps_off, df_agents_off, meta_off, outdir=OUTDIR, prefix=f"{run_id}_OFF")

print("✓ Logged rich datasets:", run_id)

