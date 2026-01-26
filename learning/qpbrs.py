# ─────────────────────────────────────────────────────────────
# SCRIPT: N AGENTS (REGISTERED / NON-REGISTERED and LOW / MODERATE TRUST)
# Q-learning + potential advice + avaluació greedy + plots
# + Capability Expansion agrupant a2,a4 i separant a1/a3 (dashed si impossibles)
# ─────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from CASimulation.environment.model import GridMAInequityEnv
from environment.context import Context, ACTION_NAMES, Actions

from learning.utils import (
    EvalLogConfig, run_eval_and_log, run_eval_and_log_rich, save_eval_artifacts, 
    _ensure_ax, group_of_agent, get_state, get_action_mask, _plot_split_rewards_4, 
    group_key, moving_average, _tuple_loc, cached_irl_potential, irl_potential_from_env,
    action_mask_from_classify, masked_argmax, plot_rewards_by_group, group_key_from_initial,
    a_label, health_to_color, final_state_comparison_figure, load_irl, chebyshev_dist, 
    plot_mean_optimal_strategy, transform_raw_features_to_standardized,
    plot_admin_initial_final, plot_admin_initial_final_by_group, plot_health_initial_final, plot_health_initial_final_by_group,
    final_state_summary_figure, 
    GROUP_COLORS, GROUP_LABEL, GROUP_LABELS, JITTER_Y, NODE_SCALE, LAYOUT_TIGHT, PASTEL_HEALTH
)

# ── paràmetres bàsics ────────────────────────────────────────
size = 7 # this should be fixed to 7 for the Raval scenario
num_peh = 8
num_sw = 15 # amb 8 PEH i 10 SW va bé (algun es pot quedar sense sw attention, 15SW ens assegurem que no), amb 16 PEH i 20 o 25SW ja no, politiques optimes es perden
universal_health = False  # True: legal norm OFF, False: legal norm ON

import json
if num_peh == 4:
    with open("peh_sample4.json", "r") as f:
        profiles = json.load(f)
elif num_peh == 8:
    with open("peh_sample8.json", "r") as f:
        profiles = json.load(f)
elif num_peh == 16:
    with open("peh_sample16.json", "r") as f:
        profiles = json.load(f)
else:
    raise ValueError(f"num_peh={num_peh} no suportat. Usa 4, 8 o 16.")

# extra things


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

# Non-registered -> dashed, Registered -> solid
GROUP_LS = {
    "NONREG_MOD": "--",
    "NONREG_LOW": "--",
    "REG_MOD":    "-.",
    "REG_LOW":    "-.",
}

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

train_episode_rows = []   # per episodi
eval_step_rows = []       # per step (avaluació)
eval_policyON_rows = []      # resum final per agent (avaluació)
eval_policyOFF_rows = []     # resum final per agent (avaluació)

# POTENTIAL-BASED ADVICE: 

irl = load_irl("irl_calibration_results_raval.json")
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
# ── crea entorn ──────────────────────────────────────────────
ctx = Context(grid_size=size)
ctx.set_scenario(universal_health=universal_health)
env = GridMAInequityEnv(context=ctx, render_mode=None, size=size, num_peh=len(profiles), num_social_agents= num_sw, peh_profiles=profiles, max_steps = 150)
env.reset(options={"peh_profiles": profiles})# train always with the same profiles

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
episode_returns_by_group = {
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

        # NO PBRS
        # env.step(action)
        # # nou estat i reward
        # new_obs = env.observe(agent)
        # individualreward = env.rewards[agent]
        # next_state = get_state(new_obs, env.peh_agents[env.agent_name_mapping[agent]])
        # # Q-update
        # s = state[agent]; a = action
        # # best_next = np.max(q_tables_advice[agent][next_state])
        # # q_tables_advice[agent][s + (a,)] += alpha * (individualreward + gamma * best_next - q_tables_advice[agent][s + (a,)])
        # next_mask = action_mask_from_classify(env, agent)
        # best_next = np.max(np.where(next_mask == 1,
        #                             q_tables_advice[agent][next_state],
        #                             -1e9))
        # q_tables_advice[agent][s + (a,)] += alpha * (
        #     individualreward + gamma * best_next - q_tables_advice[agent][s + (a,)]
        # )
        # state[agent] = next_state
        # ep_ret[agent] += individualreward

        if len(env.agents) == 0:
            break

    # bookkeeping
    episode_returns.append(sum(ep_ret.values()))
    epsilon = max(eps_min, epsilon * eps_decay)

    # rewards per grup (admin x trust) segons estat inicial de l'episodi
    group_totals = {k: 0.0 for k in episode_returns_by_group.keys()}
    group_counts = {k: 0   for k in episode_returns_by_group.keys()}

    for agent, ret in ep_ret.items():
        g = init_group[agent]
        group_totals[g] += ret
        group_counts[g] += 1

    for k in episode_returns_by_group.keys():
        if group_counts[k] > 0:
            episode_returns_by_group[k].append(group_totals[k] / group_counts[k])  # MEAN
        else:
            episode_returns_by_group[k].append(0.0)
    
    train_episode_rows.append({
            "run_id": RUN_ID,
            "episode": ep,
            "epsilon": float(epsilon),
            "total_reward": float(sum(ep_ret.values())),
            "mean_reward_nonreg_low": float(episode_returns_by_group["NONREG_LOW"][-1]),
            "mean_reward_nonreg_mod": float(episode_returns_by_group["NONREG_MOD"][-1]),
            "mean_reward_reg_low": float(episode_returns_by_group["REG_LOW"][-1]),
            "mean_reward_reg_mod": float(episode_returns_by_group["REG_MOD"][-1]),
        })

plot_rewards_by_group(episode_returns_by_group, w=10)


# POLICY ON
# ─────────────────────────────────────────────────────────────
#  GREEDY EVALUATION — CAPABILITIES & POLICY
# ─────────────────────────────────────────────────────────────
from collections import Counter
ctx = Context(grid_size=size)
ctx.set_scenario(universal_health=universal_health)
eval_env = GridMAInequityEnv(context=ctx, render_mode=None, size=size, num_peh=len(profiles),num_social_agents= num_sw, peh_profiles=profiles)
eval_env.reset(options={"peh_profiles": profiles})


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

# IMPORTANT: crea un env nou per log (no reutilitzis eval_env ja consumit)
ctx_log = Context(grid_size=size)
ctx_log.set_scenario(universal_health=False)
eval_env_log = GridMAInequityEnv(
    context=ctx_log, render_mode=None, size=size,
    num_peh=len(profiles), num_social_agents=num_sw, peh_profiles=profiles
)
eval_env_log.reset(options={"peh_profiles": profiles})


# (B) Corre eval + log dataset
df_steps_ON, df_agents_ON, summary_ON = run_eval_and_log(
    eval_env,
    q_tables_advice_ON,               # <-- el Q-table que estàs usant per legal norm ON
    scenario_name="legal_norm_ON",
    run_id=run_id,
    policy_tag="eval_policyON",
    max_total_steps=50000,
    verbose=False,
)

# (C) Desa fitxers (CSV/JSON) per al teu notebook
df_steps_ON.to_csv(f"dataset_steps_legal_norm_ON_{run_id}.csv", index=False)
df_agents_ON.to_csv(f"dataset_agents_legal_norm_ON_{run_id}.csv", index=False)

print("✓ Saved datasets for legal_norm_ON:", run_id)
for ag in eval_env.possible_agents:
    eval_policyON_rows.append({
        "run_id": RUN_ID,
        "agent": ag,
        "final_health": float(health_state_trace[ag][-1]) if len(health_state_trace[ag]) > 0 else None,
        "final_administrative_state": (
            "registered" if (len(admin_state_trace[ag]) > 0 and admin_state_trace[ag][-1] == 1)
            else ("non-registered" if len(admin_state_trace[ag]) > 0 else None)
        ),
        "total_steps": int(len(health_state_trace[ag])),
    })

print("\n=== FINAL STATES (EVAL) ===")
for ag in eval_env.possible_agents:
    idx = eval_env.agent_name_mapping[ag]
    peh = eval_env.peh_agents[idx]
    last_action = exec_hist[ag][-1][1] if len(exec_hist[ag]) else None
    print(f"{ag} init={group_key_from_initial(initial_admin_state, initial_trust_type, ag)} "
          f"final_h={peh.health_state:.2f} final_admin={peh.administrative_state} "
          f"done={eval_env.dones.get(ag)} term={eval_env.terminations.get(ag)} trunc={eval_env.truncations.get(ag)} "
          f"last_action={(Actions(last_action).name if last_action is not None else None)}")

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

# ─────────────────────────────────────────────────────────────
#  Capability expansion — agrupat i amb color per salut
# ─────────────────────────────────────────────────────────────

from matplotlib.colors import LinearSegmentedColormap, Normalize
def capability_expansion_plot(
    masks_over_time, bh_series, af_series, health_series,
    *, conditional_indices=(0, 2),  # (a1, a2)
    step_labels=None,               # llista de passos globals del non-reg
    node_scale=NODE_SCALE,
    layout=LAYOUT_TIGHT,
    show_side_labels=True,
    title= "(F) Capability expansion (non-registered)",
    ax=None,
    add_health_cbar=False,
    cbar_pos=None
):
    """
    Dibuixa l'expansió de capacitats amb rombos:
      - grup superior: \bar{a1}, \bar{a2} (NO a5)
      - a1/a2: B/N si impossible, acolorit si possible; es separen quan difereixen
      - BH/AF per sobre del marc (espai constant)
    """  
    T = len(masks_over_time)
    fig, ax, created = _ensure_ax(ax, title=title)
    ax.set_title(title, pad=18)  # ← una mica més d’aire
    if T == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        if created: plt.show()
        return ax

    x = np.arange(T)
    m = np.array([np.array(msk, dtype=int) for msk in masks_over_time], dtype=int)
    a1_pos = m[:, conditional_indices[0]]
    a2_pos = m[:, conditional_indices[1]]
        # etiquetes BH/AF a l’esquerra, fora del marc
    if show_side_labels:
        ax.text(-0.03, layout["bh_y"], "Bodily Health",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, clip_on=False)
        ax.text(-0.03, layout["af_y"], "Affiliation",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, clip_on=False)
    # mida (area en pt^2)
    def node_size(k):  # k ∈ {1,2}
        base = {1: 260, 2: 360}[int(k)]
        return base * float(node_scale)

    y_group = layout["y_group"]; y_a2 = layout["y_a2"]
    y_combo = layout["y_combo"]; y_a1 = layout["y_a1"]

    # rang vertical normalitzat (espai fix)
    ax.set_ylim(*layout["ylim"])

    GROUP_LABEL = r"$\bar{a}_1,\ \bar{a}_2$"
    COMBO_LABEL = r"$a_1,a_2$"

    for t in range(T):
        # inside capability_expansion_plot loop
        face = health_to_color(health_series[t] if t < len(health_series) else 2.5, alpha=0.95)
            

        # ─ top group (\bar{a1}, \bar{a2})
        ax.scatter(x[t], y_group, s=node_size(2), marker='D',
                   facecolors=face, edgecolors='black', linewidths=1.8)
        ax.text(x[t], y_group, GROUP_LABEL, ha="center", va="center", fontsize=13)

        # ─ BH/AF números per sobre del marc (no s'encavalquen)
        if t < len(bh_series):
            ax.text(x[t], layout["bh_y"], f"{bh_series[t]:.2f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=12, clip_on=False)
        if t < len(af_series):
            ax.text(x[t], layout["af_y"], f"{af_series[t]:.2f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=12, clip_on=False)

        # ─ a1/a2 lane
        if a1_pos[t]==0 and a2_pos[t]==0:
            ax.scatter(x[t], y_combo, s=node_size(2), marker='D',
                       facecolors='white', edgecolors='black', linewidths=1.8, linestyle=(0,(5,3)))
            ax.text(x[t], y_combo, COMBO_LABEL, ha="center", va="center", fontsize=13)
        elif a1_pos[t]==1 and a2_pos[t]==1:
            ax.scatter(x[t], y_combo, s=node_size(2), marker='D',
                       facecolors=face, edgecolors='black', linewidths=1.8)
            ax.text(x[t], y_combo, COMBO_LABEL, ha="center", va="center", fontsize=13)
        else:
            # a2
            ax.scatter(x[t], y_a2, s=node_size(1), marker='D',
                       facecolors=(face if a2_pos[t]==1 else 'white'),
                       edgecolors='black', linewidths=1.8, linestyle=((0,(5,3)) if a2_pos[t]==0 else 'solid'))
            ax.text(x[t], y_a2, r"$a_2$", ha="center", va="center", fontsize=13)
            # a1
            ax.scatter(x[t], y_a1, s=node_size(1), marker='D',
                       facecolors=(face if a1_pos[t]==1 else 'white'),
                       edgecolors='black', linewidths=1.8, linestyle=((0,(5,3)) if a1_pos[t]==0 else 'solid'))
            ax.text(x[t], y_a1, r"$a_1$", ha="center", va="center", fontsize=13)

    # etiquetes BH/AF a l’esquerra, fora del marc
    ax.text(-0.03, layout["bh_y"], "Bodily Health",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, clip_on=False)
    ax.text(-0.03, layout["af_y"], "Affiliation",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, clip_on=False)
    
    # colorbar per salut (si es demana)
    if add_health_cbar:
        # subplot bbox in figure coords
        bbox = ax.get_position()
        left  = bbox.x0 - 0.06      # how far left of the axes (tweak)
        width = 0.022               # bar width (tweak)

        # top of the bar just below the "Affiliation" label
        top_frac = layout["af_y"] - 0.03   # 3% under the AF label (axes fraction)
        top     = bbox.y0 + bbox.height * top_frac
        bottom  = bbox.y0                  # start at axis bottom
        height  = top - bottom

        cax = ax.figure.add_axes([left, bottom, width, height])

        # smooth gradient through your four health colors (0→1)
        cmap = LinearSegmentedColormap.from_list(
            "health",
            [(0.00, PASTEL_HEALTH[1]),
             (0.33, PASTEL_HEALTH[2]),
             (0.66, PASTEL_HEALTH[3]),
             (1.00, PASTEL_HEALTH[4])]
        )
        norm = Normalize(0.0, 1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cb.ax.tick_params(length=0, labelsize=10, pad=2)
        # put title/labels on the LEFT side of the bar
        cb.set_label("Health state", fontsize=11, labelpad=8)
        cb.ax.yaxis.set_label_position('left')
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.tick_params(labelleft=True, labelright=False)
        cb.outline.set_linewidth(0.8)

    # eix X
    tick_labels = (np.arange(1, T+1) if step_labels is None else list(step_labels[:T]))
    ax.set_yticks([])
    ax.set_xlabel("Simulation step")
    ax.set_xticks(x)
    ax.set_xticklabels(x)   # no subtract-1 tricks
    ax.set_xlim(-0.5, T-0.5)

    if created:
        plt.tight_layout()
        plt.show()

    return ax

if nonreg_agent_id is not None:
    masks_over_time = action_masks_trace[nonreg_agent_id]
    T = len(masks_over_time)
    capability_expansion_plot(
        masks_over_time,
        bh_turns[:T],
        af_turns[:T],
        health_turns[:T],
        conditional_indices=(0,2),           # (a1, a2)
        node_scale=NODE_SCALE,
        layout=LAYOUT_TIGHT,
        add_health_cbar=True
    )

# --- Capability expansion (registered) — standalone plot ---
if reg_agent_id is not None and len(action_masks_trace[reg_agent_id]) > 0:
    masks_reg = action_masks_trace[reg_agent_id]
    Tr = len(masks_reg)
    fig, ax = plt.subplots(figsize=(10, 5))
    capability_expansion_plot(
        masks_reg,
        reg_bh_turns[:Tr],
        reg_af_turns[:Tr],
        reg_health_turns[:Tr],
        conditional_indices=(0, 2),        # (a1, a2)
        # step_labels=None  -> local 0..Tr-1 (consistent with other plots)
        node_scale=NODE_SCALE,
        layout=LAYOUT_TIGHT,
        title="Capability expansion (registered)",
        ax = ax
    )
    plt.show()

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

def save_results_summary():
    os.makedirs("output", exist_ok=True)
    # crea canvas amb 3x2 subgràfics
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"Results — size={size}, universal_health={universal_health}", fontsize=14)

    # 1) learning curve
    _plot_learning_curve(axs[0,0], episode_returns)

    # 2) split groups
    _plot_split_rewards_4(axs[0,1], episode_returns_by_group)

    # 3) greedy policy scatter
    num_actions = env.action_space(env.possible_agents[0]).n
    _plot_optimal_strategy(
    axs[1,0], exec_hist, num_actions, apply_for_shelter_idx,
    initial_admin_state, initial_trust_type,
    exclude_idxs=[apply_for_shelter_idx]
    )

    # 4) BH/AF traces
    _plot_capability_traces(axs[1,1], bh_trace, af_trace, initial_admin_state)

    # 5) central functionings
    _plot_functionings(axs[2,0], health_state_trace, admin_state_trace, initial_admin_state)

    # 6) capability expansion compact (non-registered)
    if nonreg_agent_id is not None and len(action_masks_trace[nonreg_agent_id]) > 0:
        masks_over_time = action_masks_trace[nonreg_agent_id]
        Tloc = len(masks_over_time)
        capability_expansion_plot(
            masks_over_time,
            bh_turns[:Tloc],
            af_turns[:Tloc],
            health_turns[:Tloc],
            conditional_indices=(0,2),
            step_labels=nonreg_step_labels[:Tloc],
            node_scale=NODE_SCALE,
            layout=LAYOUT_TIGHT,
            title="Capability expansion (non-registered)",
            ax=axs[2,1]
        )
    else:
        axs[2,1].text(0.5, 0.5, "No non-registered capability data",
                    ha="center", va="center")
        axs[2,1].set_axis_off()

    # desa
    fig.savefig(os.path.join("output","results.png"), dpi=300)
    fig.savefig(os.path.join("output","results.pdf"))
    plt.close(fig)

# call to build & save the summary figure
save_results_summary()
# ========================================================================



# POLICY OFF
universal_health = True  # desactiva la política


# ── crea entorn ──────────────────────────────────────────────
ctx2 = Context(grid_size=size)
ctx2.set_scenario(universal_health=universal_health)
env = GridMAInequityEnv(context=ctx2, render_mode=None, size=size, num_peh=len(profiles), num_social_agents= num_sw, peh_profiles=profiles, max_steps = 300)
env.reset(options={"peh_profiles": profiles})# train always with the same profiles

# ── Q-tables per agent ───────────────────────────────────────

MAX_ENC = 10
MAX_NONENG = 10

n_health = int((env.peh_agents[0].max_health - env.peh_agents[0].min_health) / env.peh_agents[0].health_step) + 1

q_tables_advice_OFF = {}

for a in env.possible_agents:
    shape = (env.size, env.size, n_health, 2, 2, MAX_ENC+1, MAX_NONENG+1, env.action_space(a).n)
    q_tables_advice_OFF[a] = np.zeros(shape)

# ── hiperparàmetres ──────────────────────────────────────────

episode_returns = []
episode_returns_by_group = {
    "NONREG_LOW": [],
    "NONREG_MOD": [],
    "REG_LOW": [],
    "REG_MOD": [],
}

# ── entrenament ──────────────────────────────────────────────
for ep in range(episodes):
    env.reset(options={"peh_profiles": profiles})# train always with the same profiles
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

        mask = action_mask_from_classify(env, agent)
        feasible = np.flatnonzero(mask)

        # ε-greedy però només entre possibles
        if np.random.rand() < epsilon:
            if len(feasible) > 0:
                action = int(np.random.choice(feasible))
            else:
                action = env.action_space(agent).sample()  # fallback (no hauria de passar)
        else:
            action = masked_argmax(q_tables_advice_OFF[agent][state[agent]], mask)

        # pas
        env.step(action)

        # nou estat i reward
        new_obs = env.observe(agent)
        individualreward = env.rewards[agent]
        next_state = get_state(new_obs, env.peh_agents[env.agent_name_mapping[agent]])

        # Q-update
        s = state[agent]; a = action
        # best_next = np.max(q_tables_advice[agent][next_state])
        # q_tables_advice[agent][s + (a,)] += alpha * (individualreward + gamma * best_next - q_tables_advice[agent][s + (a,)])
        next_mask = action_mask_from_classify(env, agent)

        best_next = np.max(np.where(next_mask == 1,
                                    q_tables_advice_OFF[agent][next_state],
                                    -1e9))
        q_tables_advice_OFF[agent][s + (a,)] += alpha * (
            individualreward + gamma * best_next - q_tables_advice_OFF[agent][s + (a,)]
        )
        state[agent] = next_state
        ep_ret[agent] += individualreward

        if len(env.agents) == 0:
            break

    # bookkeeping
    episode_returns.append(sum(ep_ret.values()))
    epsilon = max(eps_min, epsilon * eps_decay)

    # rewards per grup (admin x trust) segons estat inicial de l'episodi
    # rewards per grup (admin x trust) segons estat inicial de l'episodi
    group_totals = {k: 0.0 for k in episode_returns_by_group.keys()}
    group_counts = {k: 0   for k in episode_returns_by_group.keys()}

    for agent, ret in ep_ret.items():
        g = init_group[agent]
        group_totals[g] += ret
        group_counts[g] += 1

    for k in episode_returns_by_group.keys():
        if group_counts[k] > 0:
            episode_returns_by_group[k].append(group_totals[k] / group_counts[k])  # MEAN
        else:
            episode_returns_by_group[k].append(0.0)

plot_rewards_by_group(episode_returns_by_group)

# ─────────────────────────────────────────────────────────────
#  GREEDY EVALUATION — CAPABILITIES & POLICY
# ────────────────────────────────────────────────────────────

ctx2 = Context(grid_size=size)
ctx2.set_scenario(universal_health=universal_health)  # universal health = policy OFF
eval_env2 = GridMAInequityEnv(context=ctx2, render_mode=None, size=size, num_peh=len(profiles),num_social_agents= num_sw, peh_profiles=profiles)
eval_env2.reset(options={"peh_profiles": profiles})
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
ctx_log.set_scenario(universal_health=True)
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

GROUP_ORDER = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]

def _aggregate_capabilities_by_group(bh_trace, af_trace, initial_admin_state, initial_trust_type):
    """
    Returns dict: group -> (mean_bh, mean_af) aggregated over all agents & all timesteps.
    """
    buckets_bh = {g: [] for g in GROUP_ORDER}
    buckets_af = {g: [] for g in GROUP_ORDER}

    for ag in bh_trace.keys():
        g = group_key_from_initial(initial_admin_state, initial_trust_type, ag)
        if g not in buckets_bh:
            continue
        buckets_bh[g].extend([float(x) for x in bh_trace.get(ag, []) if x is not None])
        buckets_af[g].extend([float(x) for x in af_trace.get(ag, []) if x is not None])

    out = {}
    for g in GROUP_ORDER:
        mbh = float(np.mean(buckets_bh[g])) if len(buckets_bh[g]) else np.nan
        maf = float(np.mean(buckets_af[g])) if len(buckets_af[g]) else np.nan
        out[g] = (mbh, maf)
    return out


def _shares_initial_final(health_state_trace, admin_state_trace, healthy_threshold=3.0):
    """
    Uses your traces dicts: health_state_trace[ag] list, admin_state_trace[ag] list (0/1).
    Computes shares at t=0 and t=T (last).
    """
    agents = list(health_state_trace.keys())
    if not agents:
        return (0.0, 0.0, 0.0, 0.0)

    h0, hT, a0, aT = [], [], [], []
    for ag in agents:
        hs = health_state_trace.get(ag, [])
        ad = admin_state_trace.get(ag, [])
        if not hs or not ad:
            continue
        h0.append(float(hs[0]) >= healthy_threshold)
        hT.append(float(hs[-1]) >= healthy_threshold)
        a0.append(int(ad[0]) == 1)
        aT.append(int(ad[-1]) == 1)

    init_healthy = float(np.mean(h0)) if len(h0) else 0.0
    fin_healthy  = float(np.mean(hT)) if len(hT) else 0.0
    init_reg     = float(np.mean(a0)) if len(a0) else 0.0
    fin_reg      = float(np.mean(aT)) if len(aT) else 0.0

    return init_healthy, fin_healthy, init_reg, fin_reg

def _budget_costs_from_env(eval_env, init_health_budget, init_social_budget):
    """
    Returns NEGATIVE costs (e.g., -300€) as final_budget - initial_budget.
    """
    ctx = getattr(eval_env, "context", None)
    if ctx is None:
        return 0.0, 0.0

    fin_health_budget = float(getattr(ctx, "healthcare_budget", init_health_budget))
    fin_social_budget = float(getattr(ctx, "social_service_budget", init_social_budget))

    # negative spend (only the delta, no initial budget shown)
    health_cost = fin_health_budget - float(init_health_budget)   # e.g., 99700 - 100000 = -300
    social_cost = fin_social_budget - float(init_social_budget)

    return social_cost, health_cost


def _draw_grid_panel(ax_grid, eval_env, initial_admin_state, initial_trust_type, title):
    size = int(getattr(eval_env, "size", 8))
    ctx = getattr(eval_env, "context", None)

    ax_grid.set_xlim(0, size); ax_grid.set_ylim(0, size)
    ax_grid.set_aspect("equal"); ax_grid.invert_yaxis()
    ax_grid.set_xticks(np.arange(0, size+1)); ax_grid.set_yticks(np.arange(0, size+1))
    ax_grid.grid(True, color="black", linewidth=1)
    ax_grid.tick_params(labelbottom=False, labelleft=False)
    ax_grid.set_title(title, loc="left", pad=10)

    # Locations
    colour_map = {"PHC": "#d0d0ff", "ICU": "#7fa8ff", "SocialService": "#f0f0f0"}
    label_map  = {"PHC": "PHC", "ICU": "ICU", "SocialService": "Social\nServices"}
    locs = getattr(ctx, "locations", {}) if ctx is not None else {}

    for name, info in (locs or {}).items():
        base = np.array(info["pos"])
        w, h = info.get("size", (1, 1))
        rect = plt.Rectangle(base, w, h,
                             facecolor=colour_map.get(name, "#dddddd"),
                             edgecolor="black", linewidth=1.5)
        ax_grid.add_patch(rect)
        ax_grid.text(base[0] + 0.1, base[1] + 0.4,
                     label_map.get(name, name),
                     fontsize=10, va="top", ha="left")

    # Agents final
    for ag in eval_env.possible_agents:
        idx = eval_env.agent_name_mapping[ag]
        peh = eval_env.peh_agents[idx]
        x, y = peh.location

        grp = group_key_from_initial(initial_admin_state, initial_trust_type, ag)
        edge_col = GROUP_COLORS.get(grp, "black")
        ls = GROUP_LS.get(grp, "-")
        face = health_to_color(peh.health_state, alpha=0.95)

        circ = plt.Circle((x+0.5, y+0.5), radius=0.35,
                          facecolor=face,
                          edgecolor=edge_col,
                          linewidth=2.4,
                          linestyle=ls,
                          zorder=3)
        ax_grid.add_patch(circ)

    # Legend for groups (colors + linestyle)
    handles = []
    for g in GROUP_ORDER:
        if g in GROUP_COLORS:
            handles.append(Line2D([0],[0],
                                  color=GROUP_COLORS[g],
                                  linestyle=GROUP_LS.get(g, "-"),
                                  linewidth=2.5,
                                  label=GROUP_LABELS.get(g, g)))
    if handles:
        ax_grid.legend(handles=handles, loc="upper left", fontsize=10, frameon=True)


def _draw_capabilities_panel(ax_caps, bh_trace, af_trace, initial_admin_state, initial_trust_type):
    ax_caps.set_title("Central capabilities", loc="left", pad=6)
    ax_caps.grid(axis="x", alpha=0.2)
    ax_caps.set_xlabel("Mean capability (agents × time)")

    cap = _aggregate_capabilities_by_group(bh_trace, af_trace, initial_admin_state, initial_trust_type)

    # Two rows: BH top, AF bottom
    y = np.array([1, 0], dtype=float)
    ax_caps.set_yticks(y)
    ax_caps.set_yticklabels(["Bodily health", "Affiliation"])

    bar_h = 0.16
    offsets = np.array([+0.24, +0.08, -0.08, -0.24])  # stack 4 groups per row

    for i, g in enumerate(GROUP_ORDER):
        col = GROUP_COLORS.get(g, None)
        mbh, maf = cap[g]
        if not np.isnan(mbh):
            ax_caps.barh(y[0] + offsets[i], mbh, height=bar_h, color=col, alpha=0.85)
        if not np.isnan(maf):
            ax_caps.barh(y[1] + offsets[i], maf, height=bar_h, color=col, alpha=0.85)

    leg_handles = [Line2D([0],[0], color=GROUP_COLORS[g], lw=6, label=GROUP_LABELS.get(g, g))
                   for g in GROUP_ORDER if g in GROUP_COLORS]
    ax_caps.legend(handles=leg_handles, fontsize=9, loc="lower right")


def _draw_functionings_panel(ax_fun, health_state_trace, admin_state_trace, healthy_threshold=3.0):
    ax_fun.set_title("Central Functionings", loc="left", pad=6)
    ax_fun.set_xlim(0.0, 1.0)
    ax_fun.grid(axis="x", alpha=0.2)

    init_healthy, fin_healthy, init_reg, fin_reg = _shares_initial_final(
        health_state_trace, admin_state_trace, healthy_threshold=healthy_threshold
    )

    # We want: green = higher bar per row, red = lower bar per row
    rows = [("Healthy", init_healthy, fin_healthy), ("Registered", init_reg, fin_reg)]

    y = np.array([1, 0], dtype=float)
    ax_fun.set_yticks(y)
    ax_fun.set_yticklabels([r[0] for r in rows])

    h = 0.32
    for j, (_, v_init, v_fin) in enumerate(rows):
        # which is higher?
        if v_init >= v_fin:
            c_init, c_fin = "green", "red"
        else:
            c_init, c_fin = "red", "green"

        ax_fun.barh(y[j] + 0.18, v_init, height=h, alpha=0.35, color=c_init, label="Initial" if j == 0 else None)
        ax_fun.barh(y[j] - 0.18, v_fin,  height=h, alpha=0.85, color=c_fin,  label="Final" if j == 0 else None)

    ax_fun.legend(fontsize=9, loc="lower right")

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# assumes you already import these from utils:
# - group_key_from_initial
# - GROUP_COLORS, GROUP_LABELS, GROUP_LS
# - health_to_color

GROUP_ORDER = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]

def _aggregate_capabilities_by_group(bh_trace, af_trace, initial_admin_state, initial_trust_type):
    """
    Returns dict: group -> (mean_bh, mean_af) aggregated over all agents & all timesteps.
    """
    buckets_bh = {g: [] for g in GROUP_ORDER}
    buckets_af = {g: [] for g in GROUP_ORDER}

    for ag in bh_trace.keys():
        g = group_key_from_initial(initial_admin_state, initial_trust_type, ag)
        if g not in buckets_bh:
            continue
        buckets_bh[g].extend([float(x) for x in bh_trace.get(ag, []) if x is not None])
        buckets_af[g].extend([float(x) for x in af_trace.get(ag, []) if x is not None])

    out = {}
    for g in GROUP_ORDER:
        mbh = float(np.mean(buckets_bh[g])) if len(buckets_bh[g]) else np.nan
        maf = float(np.mean(buckets_af[g])) if len(buckets_af[g]) else np.nan
        out[g] = (mbh, maf)
    return out


def _shares_initial_final(health_state_trace, admin_state_trace, healthy_threshold=3.0):
    """
    Uses your traces dicts: health_state_trace[ag] list, admin_state_trace[ag] list (0/1).
    Computes shares at t=0 and t=T (last).
    """
    agents = list(health_state_trace.keys())
    if not agents:
        return (0.0, 0.0, 0.0, 0.0)

    h0, hT, a0, aT = [], [], [], []
    for ag in agents:
        hs = health_state_trace.get(ag, [])
        ad = admin_state_trace.get(ag, [])
        if not hs or not ad:
            continue
        h0.append(float(hs[0]) >= healthy_threshold)
        hT.append(float(hs[-1]) >= healthy_threshold)
        a0.append(int(ad[0]) == 1)
        aT.append(int(ad[-1]) == 1)

    init_healthy = float(np.mean(h0)) if len(h0) else 0.0
    fin_healthy  = float(np.mean(hT)) if len(hT) else 0.0
    init_reg     = float(np.mean(a0)) if len(a0) else 0.0
    fin_reg      = float(np.mean(aT)) if len(aT) else 0.0

    return init_healthy, fin_healthy, init_reg, fin_reg


def _budget_costs_from_env(eval_env, init_health_budget, init_social_budget):
    """
    Compute realised costs from budget deltas (robust; avoids '0 healthcare' due to bad inference).
    """
    ctx = getattr(eval_env, "context", None)
    if ctx is None:
        return 0.0, 0.0

    fin_health_budget = float(getattr(ctx, "healthcare_budget", init_health_budget))
    fin_social_budget = float(getattr(ctx, "social_service_budget", init_social_budget))

    health_cost = max(0.0, float(init_health_budget) - fin_health_budget)
    social_cost = max(0.0, float(init_social_budget) - fin_social_budget)
    return social_cost, health_cost


def _draw_grid_panel(ax_grid, eval_env, initial_admin_state, initial_trust_type, title):
    size = int(getattr(eval_env, "size", 8))
    ctx = getattr(eval_env, "context", None)

    ax_grid.set_xlim(0, size); ax_grid.set_ylim(0, size)
    ax_grid.set_aspect("equal"); ax_grid.invert_yaxis()
    ax_grid.set_xticks(np.arange(0, size+1)); ax_grid.set_yticks(np.arange(0, size+1))
    ax_grid.grid(True, color="black", linewidth=1)
    ax_grid.tick_params(labelbottom=False, labelleft=False)
    ax_grid.set_title(title, loc="left", pad=10)

    # Locations
    colour_map = {"PHC": "#d0d0ff", "ICU": "#7fa8ff", "SocialService": "#f0f0f0"}
    label_map  = {"PHC": "PHC", "ICU": "ICU", "SocialService": "Social\nServices"}
    locs = getattr(ctx, "locations", {}) if ctx is not None else {}

    for name, info in (locs or {}).items():
        base = np.array(info["pos"])
        w, h = info.get("size", (1, 1))
        rect = plt.Rectangle(base, w, h,
                             facecolor=colour_map.get(name, "#dddddd"),
                             edgecolor="black", linewidth=1.5)
        ax_grid.add_patch(rect)
        ax_grid.text(base[0] + 0.1, base[1] + 0.4,
                     label_map.get(name, name),
                     fontsize=10, va="top", ha="left")

    # Agents final
    for ag in eval_env.possible_agents:
        idx = eval_env.agent_name_mapping[ag]
        peh = eval_env.peh_agents[idx]
        x, y = peh.location

        grp = group_key_from_initial(initial_admin_state, initial_trust_type, ag)
        edge_col = GROUP_COLORS.get(grp, "black")
        ls = GROUP_LS.get(grp, "-")
        face = health_to_color(peh.health_state, alpha=0.95)

        circ = plt.Circle((x+0.5, y+0.5), radius=0.35,
                          facecolor=face,
                          edgecolor=edge_col,
                          linewidth=2.4,
                          linestyle=ls,
                          zorder=3)
        ax_grid.add_patch(circ)

    # Legend for groups (colors + linestyle)
    handles = []
    for g in GROUP_ORDER:
        if g in GROUP_COLORS:
            handles.append(Line2D([0],[0],
                                  color=GROUP_COLORS[g],
                                  linestyle=GROUP_LS.get(g, "-"),
                                  linewidth=2.5,
                                  label=GROUP_LABELS.get(g, g)))
    if handles:
        ax_grid.legend(handles=handles, loc="upper left", fontsize=10, frameon=True)


def _draw_capabilities_panel(ax_caps, bh_trace, af_trace, initial_admin_state, initial_trust_type):
    ax_caps.set_title("Capabilities", loc="left", pad=6)
    ax_caps.grid(axis="x", alpha=0.2)
    ax_caps.set_xlabel("Mean capability (agents × time)")

    cap = _aggregate_capabilities_by_group(bh_trace, af_trace, initial_admin_state, initial_trust_type)

    # Two rows: BH top, AF bottom
    y = np.array([1, 0], dtype=float)
    ax_caps.set_yticks(y)
    ax_caps.set_yticklabels(["Bodily health", "Affiliation"])

    bar_h = 0.16
    offsets = np.array([+0.24, +0.08, -0.08, -0.24])  # stack 4 groups per row

    for i, g in enumerate(GROUP_ORDER):
        col = GROUP_COLORS.get(g, None)
        mbh, maf = cap[g]
        if not np.isnan(mbh):
            ax_caps.barh(y[0] + offsets[i], mbh, height=bar_h, color=col, alpha=0.85)
        if not np.isnan(maf):
            ax_caps.barh(y[1] + offsets[i], maf, height=bar_h, color=col, alpha=0.85)

    leg_handles = [Line2D([0],[0], color=GROUP_COLORS[g], lw=6, label=GROUP_LABELS.get(g, g))
                   for g in GROUP_ORDER if g in GROUP_COLORS]
    ax_caps.legend(handles=leg_handles, fontsize=9, loc="lower right")


def _draw_functionings_panel(ax_fun, health_state_trace, admin_state_trace, healthy_threshold=3.0):
    ax_fun.set_title("Functionings", loc="left", pad=6)
    ax_fun.set_xlim(0.0, 1.0)
    ax_fun.grid(axis="x", alpha=0.2)

    init_healthy, fin_healthy, init_reg, fin_reg = _shares_initial_final(
        health_state_trace, admin_state_trace, healthy_threshold=healthy_threshold
    )

    # We want: green = higher bar per row, red = lower bar per row
    rows = [("Healthy", init_healthy, fin_healthy), ("Registered", init_reg, fin_reg)]

    y = np.array([1, 0], dtype=float)
    ax_fun.set_yticks(y)
    ax_fun.set_yticklabels([r[0] for r in rows])

    h = 0.32
    for j, (_, v_init, v_fin) in enumerate(rows):
        # which is higher?
        if v_init >= v_fin:
            c_init, c_fin = "green", "red"
        else:
            c_init, c_fin = "red", "green"

        ax_fun.barh(y[j] + 0.18, v_init, height=h, alpha=0.35, color=c_init, label="Initial" if j == 0 else None)
        ax_fun.barh(y[j] - 0.18, v_fin,  height=h, alpha=0.85, color=c_fin,  label="Final" if j == 0 else None)

    ax_fun.legend(fontsize=9, loc="lower right")


def plot_policy_summary_comparison(
    *,
    # ON bundle
    env_on, bh_on, af_on, health_on, admin_on, init_admin_on, init_trust_on,
    init_health_budget_on, init_social_budget_on,
    # OFF bundle
    env_off, bh_off, af_off, health_off, admin_off, init_admin_off, init_trust_off,
    init_health_budget_off, init_social_budget_off,
    healthy_threshold=3.0,
    figsize=(14, 7),
):
    """
    2-row comparison:
      Row 1: Policy ON
      Row 2: Policy OFF
    Each row: grid left, (capabilities + functionings + costs text) right.
    """

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.25], hspace=0.25, wspace=0.15)

    def draw_row(row, title, env, bh, af, health_trace, admin_trace, init_admin, init_trust, init_hb, init_sb):
        ax_grid = fig.add_subplot(gs[row, 0])
        gs_r = gs[row, 1].subgridspec(3, 1, height_ratios=[1.05, 0.95, 0.18], hspace=0.28)
        ax_caps = fig.add_subplot(gs_r[0, 0])
        ax_fun  = fig.add_subplot(gs_r[1, 0])
        ax_cost = fig.add_subplot(gs_r[2, 0])

        _draw_grid_panel(ax_grid, env, init_admin, init_trust, title)
        _draw_capabilities_panel(ax_caps, bh, af, init_admin, init_trust)
        _draw_functionings_panel(ax_fun, health_trace, admin_trace, healthy_threshold=healthy_threshold)

        social_cost, health_cost = _budget_costs_from_env(env, init_hb, init_sb)
        ax_cost.axis("off")
        ax_cost.text(
            -0.06, 0.55,
            f"Economic costs:  Social services = {social_cost:.0f} €   |   Healthcare = {health_cost:.0f} €",
            ha="left", va="center", fontsize=10, alpha=0.9, transform=ax_cost.transAxes
        )

    draw_row(
        0, "Policy ON",
        env_on, bh_on, af_on, health_on, admin_on, init_admin_on, init_trust_on,
        init_health_budget_on, init_social_budget_on
    )
    draw_row(
        1, "Policy OFF",
        env_off, bh_off, af_off, health_off, admin_off, init_admin_off, init_trust_off,
        init_health_budget_off, init_social_budget_off
    )

    plt.tight_layout()
    plt.show()


plot_policy_summary_comparison(
    # ON
    env_on=eval_env,
    bh_on=bh_trace,
    af_on=af_trace,
    health_on=health_state_trace,
    admin_on=admin_state_trace,
    init_admin_on=initial_admin_state,
    init_trust_on=initial_trust_type,
    init_health_budget_on=init_health_budget_on,
    init_social_budget_on=init_social_budget_on,

    # OFF
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
def _cap_agg_by_group(bh_trace, af_trace, init_admin, init_trust):
    groups = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]
    bh_init = {g: [] for g in groups}
    bh_final = {g: [] for g in groups}
    af_init = {g: [] for g in groups}
    af_final = {g: [] for g in groups}

    for ag in bh_trace.keys():
        g = group_key_from_initial(init_admin, init_trust, ag)
        if g not in bh_init:
            continue
        bh_seq = bh_trace.get(ag, [])
        af_seq = af_trace.get(ag, [])
        
        if len(bh_seq) > 0:
            bh_init[g].append(float(bh_seq[0]))
            bh_final[g].append(float(bh_seq[-1]))
        if len(af_seq) > 0:
            af_init[g].append(float(af_seq[0]))
            af_final[g].append(float(af_seq[-1]))

    bh_init_mean = [float(np.mean(bh_init[g])) if len(bh_init[g]) else np.nan for g in groups]
    bh_final_mean = [float(np.mean(bh_final[g])) if len(bh_final[g]) else np.nan for g in groups]
    af_init_mean = [float(np.mean(af_init[g])) if len(af_init[g]) else np.nan for g in groups]
    af_final_mean = [float(np.mean(af_final[g])) if len(af_final[g]) else np.nan for g in groups]
    
    return groups, bh_init_mean, bh_final_mean, af_init_mean, af_final_mean
import numpy as np
import matplotlib.pyplot as plt

def plot_policy_summary_comparison(
    *,
    # ON
    env_on,
    bh_on, af_on,
    health_on, admin_on,
    init_admin_on, init_trust_on,
    init_health_budget_on, init_social_budget_on,

    # OFF
    env_off,
    bh_off, af_off,
    health_off, admin_off,
    init_admin_off, init_trust_off,
    init_health_budget_off, init_social_budget_off,

    # settings
    healthy_threshold=3.0,
    figsize=(18, 12),
    wspace=-0.07,
    hspace=0.18,
    grid_width_ratio=1.35,
    right_width_ratio=1.0,
    show_social_workers=True,
    title_on="Policy OFF",
    title_off="Policy ON",
    font_big=16,
    font_med=14,
    font_small=12,
):
    """
    2x2:
      Row1: ON  -> [Grid | Capabilities + Functionings + Costs]
      Row2: OFF -> [Grid | Capabilities + Functionings + Costs]

    Inputs are *traces and envs* (as in qpbrs), no datasets required.
    """

    # ---------------- helpers ----------------
    def group_key_from_initial(init_admin, init_trust, ag):
        admin0 = init_admin.get(ag, "")
        trust0 = init_trust.get(ag, "")
        admin = "NONREG" if str(admin0) == "non-registered" else "REG"
        trust = "LOW" if str(trust0) == "LOW_TRUST" else "MOD"
        return f"{admin}_{trust}"

    def _get_locations(env):
        ctx = getattr(env, "context", None)
        return getattr(ctx, "locations", {}) if ctx is not None else {}
    def _fmt_pct(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{100.0*float(x):.0f}%"

    def _round_key(v, nd=3):
        # robust key for floats in [0,1]; nd=3 => 0.001 resolution
        if not np.isfinite(v):
            return None
        return round(float(v), nd)

    def _merge_equal_groups(items, nd=3):
        """
        items: list of dicts with keys: group, init, final
        returns: list of merged dicts with keys: groups(list), init, final
        """
        buckets = {}
        for it in items:
            ini = 0.0 if not np.isfinite(it["init"]) else float(it["init"])
            fin = 0.0 if not np.isfinite(it["final"]) else float(it["final"])
            key = (_round_key(ini, nd), _round_key(fin, nd))
            buckets.setdefault(key, {"groups": [], "init": ini, "final": fin})
            buckets[key]["groups"].append(it["group"])

        # keep deterministic order: sort by bar length descending then label
        merged = list(buckets.values())
        merged.sort(key=lambda d: (-max(d["init"], d["final"]), ",".join(d["groups"])))
        return merged

    def _groups_label(gs):
        # nicer label in the plot when merged
        if set(gs) == set(["NONREG_LOW","NONREG_MOD","REG_LOW","REG_MOD"]):
            return "All groups"
        # short names
        short = {
            "NONREG_LOW": "Non-reg + low",
            "NONREG_MOD": "Non-reg + mod",
            "REG_LOW": "Reg + low",
            "REG_MOD": "Reg + mod",
        }
        return ", ".join(short[g] for g in gs)


    def _draw_social_workers(ax, env):
        if not (show_social_workers and hasattr(env, "socserv_agents") and env.socserv_agents):
            return
        n_sw = len(env.socserv_agents)
        for k, sw in enumerate(env.socserv_agents):
            x, y = sw.location
            jitter = 0.10
            ang = 2 * np.pi * (k / max(1, n_sw))
            dx = jitter * np.cos(ang)
            dy = jitter * np.sin(ang)
            ax.scatter(x + 0.5 + dx, y + 0.5 + dy, s=30, color="grey", edgecolors="none", zorder=2)

    def _cap_agg_by_group(bh_trace, af_trace, init_admin, init_trust):
        groups = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]
        bh_init = {g: [] for g in groups}
        bh_final = {g: [] for g in groups}
        af_init = {g: [] for g in groups}
        af_final = {g: [] for g in groups}

        for ag in bh_trace.keys():
            g = group_key_from_initial(init_admin, init_trust, ag)
            if g not in bh_init:
                continue
            bh_seq = bh_trace.get(ag, [])
            af_seq = af_trace.get(ag, [])
            
            if len(bh_seq) > 0:
                bh_init[g].append(float(bh_seq[0]))
                bh_final[g].append(float(bh_seq[-1]))
            if len(af_seq) > 0:
                af_init[g].append(float(af_seq[0]))
                af_final[g].append(float(af_seq[-1]))

        bh_init_mean = [float(np.mean(bh_init[g])) if len(bh_init[g]) else np.nan for g in groups]
        bh_final_mean = [float(np.mean(bh_final[g])) if len(bh_final[g]) else np.nan for g in groups]
        af_init_mean = [float(np.mean(af_init[g])) if len(af_init[g]) else np.nan for g in groups]
        af_final_mean = [float(np.mean(af_final[g])) if len(af_final[g]) else np.nan for g in groups]
        
        return groups, bh_init_mean, bh_final_mean, af_init_mean, af_final_mean

    def _lerp(c1, c2, t):
        t = float(np.clip(t, 0.0, 1.0))
        return tuple((1 - t) * np.array(c1) + t * np.array(c2))

    def _metric_colors(values, c_bad, c_good):
        v = np.array(values, dtype=float)
        mask = np.isfinite(v)
        if not np.any(mask):
            return [c_bad] * len(values)
        vmin = float(np.nanmin(v[mask]))
        vmax = float(np.nanmax(v[mask]))
        if np.isclose(vmin, vmax):
            return [c_good] * len(values)
        cols = []
        for x in v:
            if not np.isfinite(x):
                cols.append((0.7, 0.7, 0.7, 0.6))
            else:
                t = (x - vmin) / (vmax - vmin)
                cols.append(_lerp(c_bad, c_good, t))
        return cols

    def _functionings_shares(health_trace, admin_trace):
        # percentatge per tots els agents a t=0 i t=T
        agents = list(health_trace.keys())
        if not agents:
            return 0.0, 0.0, 0.0, 0.0

        h0, hT, a0, aT = [], [], [], []
        for ag in agents:
            hs = health_trace.get(ag, [])
            ad = admin_trace.get(ag, [])
            if len(hs) == 0 or len(ad) == 0:
                continue
            h0.append(float(hs[0]));   hT.append(float(hs[-1]))
            a0.append(int(ad[0]));     aT.append(int(ad[-1]))

        if not h0:
            return 0.0, 0.0, 0.0, 0.0

        init_healthy = float(np.mean(np.array(h0) >= healthy_threshold))
        final_healthy = float(np.mean(np.array(hT) >= healthy_threshold))
        init_reg = float(np.mean(np.array(a0) == 1))
        final_reg = float(np.mean(np.array(aT) == 1))
        return init_healthy, final_healthy, init_reg, final_reg

    def _pair_colors(a, b, c_bad, c_good):
        # verd a la barra més alta, vermell a la més baixa
        return (c_good, c_bad) if a >= b else (c_bad, c_good)

    def _draw_grid(ax, env, init_admin, init_trust, title):
        size = getattr(env, "size", 8)
        locs = _get_locations(env)

        ax.set_xlim(0, size); ax.set_ylim(0, size)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks(np.arange(0, size+1)); ax.set_yticks(np.arange(0, size+1))
        ax.grid(True, color="black", linewidth=1)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_title(title, fontsize=font_big, pad=8)

        colour_map = {"PHC": "#d0d0ff", "ICU": "#7fa8ff", "SocialService": "#f0f0f0"}
        label_map  = {"PHC": "PHC", "ICU": "ICU", "SocialService": "Social\nServices"}

        for name, info in (locs or {}).items():
            base = np.array(info["pos"])
            w, h = info.get("size", (1, 1))
            rect = plt.Rectangle(base, w, h,
                                 facecolor=colour_map.get(name, "#dddddd"),
                                 edgecolor="black", linewidth=1.5, zorder=2)
            ax.add_patch(rect)
            ax.text(base[0] + 0.1, base[1] + 0.4, label_map.get(name, name),
                    fontsize=font_small, va="top", ha="left", zorder=2)

        _draw_social_workers(ax, env)

        # PEH finals (sense edgecolor per grup; només linestyle per admin final)
        for ag in env.possible_agents:
            idx = env.agent_name_mapping[ag]
            peh = env.peh_agents[idx]
            x, y = peh.location
            face = health_to_color(peh.health_state, alpha=0.95)

            is_reg = (peh.administrative_state == "registered")
            ls = "-" if is_reg else "--"

            circ = plt.Circle((x+0.5, y+0.5), radius=0.35,
                              facecolor=face, edgecolor="black",
                              linewidth=2.0, linestyle=ls, zorder=3)
            ax.add_patch(circ)

    def _draw_right(ax_caps, ax_fun, ax_cost, env, bh_trace, af_trace, health_trace, admin_trace,
                    init_admin, init_trust, init_health_budget, init_social_budget):

        # colors extrems basats en la teva funció de salut
        c_bad = health_to_color(1.0, alpha=0.95)
        c_good = health_to_color(4.0, alpha=0.95)

        # ----- Capabilities -----
        from learning.utils import GROUP_COLORS
        
        GROUP_LABELS_SHORT = {
            "NONREG_LOW": "Non-reg + Low trust",
            "NONREG_MOD": "Non-reg + Mod trust",
            "REG_LOW": "Reg + Low trust",
            "REG_MOD": "Reg + Mod trust",
        }
        TRUST_COLORS = {
            "LOW": "#ff7f0e",   # taronja (exemple)
            "MOD": "#2ca02c",   # verd (exemple)
        }

        def _trust_from_group(g):
            # g: "NONREG_LOW", "REG_MOD", ...
            return "LOW" if g.endswith("_LOW") else "MOD"

        def _admin_from_group(g):
            return "NONREG" if g.startswith("NONREG") else "REG"
                
        ax_caps.set_title("Central capabilities", loc="left", fontsize=font_med, pad=4, fontweight='bold')
        ax_caps.grid(axis="x", alpha=0.2)
        ax_caps.set_xlim(0.0, 1.0)
        ax_caps.tick_params(labelsize=font_small)

        groups, bh_init, bh_final, af_init, af_final = _cap_agg_by_group(bh_trace, af_trace, init_admin, init_trust)

        y_rows = {"BH": 1.0, "AF": 0.0}
        offsets = np.array([+0.24, +0.08, -0.08, -0.24])
        bar_h = 0.14

        for j, g in enumerate(groups):
            trust = _trust_from_group(g)
            admin = _admin_from_group(g)

            face = TRUST_COLORS[trust]           # color = trust
            hatch = "///" if admin == "NONREG" else None
            ls = "--" if admin == "NONREG" else "-"
            lw = 1.3

            # --- Bodily Health ---
            yi = y_rows["BH"] + offsets[j]
            ini = bh_init[j]; fin = bh_final[j]
            ini_v = 0.0 if not np.isfinite(ini) else float(ini)
            fin_v = 0.0 if not np.isfinite(fin) else float(fin)
            v = max(ini_v, fin_v)

            if np.isfinite(ini) or np.isfinite(fin):
                p = ax_caps.barh(yi, v, height=bar_h, color=face, alpha=0.75,
                                edgecolor="black", linewidth=lw)[0]
                p.set_hatch(hatch)
                p.set_linestyle(ls)
                ax_caps.text(min(max(v, 0.02), 1.18), yi,
                            f"Initial: {100*ini_v:.0f}%   Final: {100*fin_v:.0f}%",
                            va="center", ha="right" if v > 0.18 else "left",
                            fontsize=font_small, color="black")

            # --- Affiliation ---
            yi = y_rows["AF"] + offsets[j]
            ini = af_init[j]; fin = af_final[j]
            ini_v = 0.0 if not np.isfinite(ini) else float(ini)
            fin_v = 0.0 if not np.isfinite(fin) else float(fin)
            v = max(ini_v, fin_v)

            if np.isfinite(ini) or np.isfinite(fin):
                p = ax_caps.barh(yi, v, height=bar_h, color=face, alpha=0.75,
                                edgecolor="black", linewidth=lw)[0]
                p.set_hatch(hatch)
                p.set_linestyle(ls)
                ax_caps.text(min(max(v, 0.02), 1.18), yi,
                            f"Initial: {100*ini_v:.0f}%   Final: {100*fin_v:.0f}%",
                            va="center", ha="right" if v > 0.18 else "left",
                            fontsize=font_small, color="black")

        ax_caps.set_yticks([y_rows["BH"], y_rows["AF"]])
        ax_caps.set_yticklabels(["Bodily Health", "Affiliation"], fontsize=font_small)

        # ----- Functionings -----
        ax_fun.set_title("Central functionings", loc="left", fontsize=font_med, pad=4, fontweight='bold')
        ax_fun.set_xlim(0.0, 1.0)
        ax_fun.grid(axis="x", alpha=0.2)
        ax_fun.tick_params(labelsize=font_small)

        init_healthy, final_healthy, init_reg, final_reg = _functionings_shares(health_trace, admin_trace)

        h_init_col, h_final_col = _pair_colors(init_healthy, final_healthy, c_bad, c_good)
        r_init_col, r_final_col = _pair_colors(init_reg, final_reg, c_bad, c_good)

        y = np.array([1, 0], dtype=float)
        hh = 0.32

        ax_fun.barh(y[0] + 0.18, init_healthy, height=hh, color=h_init_col, alpha=0.65, label="Initial")
        ax_fun.barh(y[0] - 0.18, final_healthy, height=hh, color=h_final_col, alpha=0.95, label="Final")
        ax_fun.barh(y[1] + 0.18, init_reg, height=hh, color=r_init_col, alpha=0.65)
        ax_fun.barh(y[1] - 0.18, final_reg, height=hh, color=r_final_col, alpha=0.95)

        ax_fun.set_yticks(y)
        ax_fun.set_yticklabels(["Healthy", "Registered"], fontsize=font_small)
        ax_fun.legend(fontsize=font_small, loc="lower right")

        # ----- Costs (delta final - initial; negatiu si despesa) -----
        ctx = getattr(env, "context", None)
        fin_h = float(getattr(ctx, "healthcare_budget", init_health_budget))
        fin_s = float(getattr(ctx, "social_service_budget", init_social_budget))

        delta_s = fin_s - float(init_social_budget)
        delta_h = fin_h - float(init_health_budget)

        ax_cost.axis("off")
        ax_cost.text(
            0.0, 0.5,
            f"Economic costs:  Social services = {delta_s:+.0f} €  |  Healthcare = {delta_h:+.0f} €",
            ha="left", va="center", fontsize=font_small, alpha=0.9, fontweight='bold',
            transform=ax_cost.transAxes
        )

    # ---------------- figure layout ----------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[grid_width_ratio, right_width_ratio],
        wspace=-0.02,
        hspace=hspace
    )

    # ON row
    ax_grid_on = fig.add_subplot(gs[0, 0])
    gs_r_on = gs[0, 1].subgridspec(3, 1, height_ratios=[1.15, 1.00, 0.20], hspace=0.75)
    ax_caps_on = fig.add_subplot(gs_r_on[0, 0])
    ax_fun_on  = fig.add_subplot(gs_r_on[1, 0])
    ax_cost_on = fig.add_subplot(gs_r_on[2, 0])

    # OFF row
    ax_grid_off = fig.add_subplot(gs[1, 0])
    gs_r_off = gs[1, 1].subgridspec(3, 1, height_ratios=[1.15, 1.00, 0.20], hspace=0.75)
    ax_caps_off = fig.add_subplot(gs_r_off[0, 0])
    ax_fun_off  = fig.add_subplot(gs_r_off[1, 0])
    ax_cost_off = fig.add_subplot(gs_r_off[2, 0])

    # ---------------- draw ----------------
    _draw_grid(ax_grid_on, env_on, init_admin_on, init_trust_on, title_on)
    _draw_right(ax_caps_on, ax_fun_on, ax_cost_on,
                env_on, bh_on, af_on, health_on, admin_on,
                init_admin_on, init_trust_on,
                init_health_budget_on, init_social_budget_on)

    _draw_grid(ax_grid_off, env_off, init_admin_off, init_trust_off, title_off)
    _draw_right(ax_caps_off, ax_fun_off, ax_cost_off,
                env_off, bh_off, af_off, health_off, admin_off,
                init_admin_off, init_trust_off,
                init_health_budget_off, init_social_budget_off)

    plt.show()

plot_policy_summary_comparison(
    # ON
    env_on=eval_env,
    bh_on=bh_trace,
    af_on=af_trace,
    health_on=health_state_trace,
    admin_on=admin_state_trace,
    init_admin_on=initial_admin_state,
    init_trust_on=initial_trust_type,
    init_health_budget_on=init_health_budget_on,
    init_social_budget_on=init_social_budget_on,

    # OFF
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
import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Necessites aquestes globals (com ja tens):
# GROUP_COLORS, GROUP_LABELS, health_to_color
soft_green = health_to_color(4.0, alpha=0.85)   # like Functionings good
soft_red   = health_to_color(1.0, alpha=0.95)   # bad

def plot_policy_summary_comparison(
    *,
    # ON
    env_on,
    bh_on, af_on,
    health_on, admin_on,
    init_admin_on, init_trust_on,
    init_health_budget_on, init_social_budget_on,

    # OFF
    env_off,
    bh_off, af_off,
    health_off, admin_off,
    init_admin_off, init_trust_off,
    init_health_budget_off, init_social_budget_off,

    # settings
    healthy_threshold=3.0,
    figsize=(18, 12),
    wspace=-0.05,
    hspace=0.18,
    grid_width_ratio=1.35,
    right_width_ratio=1.0,
    show_social_workers=True,
    title_on="Policy OFF",
    title_off="Policy ON",
    font_big=16,
    font_med=14,
    font_small=12,

    # new display settings
    xlim=(0.0, 1.1),
    xlabel="Population (%)",
):
    """
    2x2:
      Row1: ON  -> [Grid | Capabilities + Functionings + Costs]
      Row2: OFF -> [Grid | Capabilities + Functionings + Costs]

    Inputs are traces and envs (qpbrs style).
    """
    def _fmt_pct(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{100.0*float(x):.0f}%"

    def _round_key(v, nd=3):
        # robust key for floats in [0,1]; nd=3 => 0.001 resolution
        if not np.isfinite(v):
            return None
        return round(float(v), nd)

    def _merge_equal_groups(items, nd=3):
        """
        items: list of dicts with keys: group, init, final
        returns: list of merged dicts with keys: groups(list), init, final
        """
        buckets = {}
        for it in items:
            ini = 0.0 if not np.isfinite(it["init"]) else float(it["init"])
            fin = 0.0 if not np.isfinite(it["final"]) else float(it["final"])
            key = (_round_key(ini, nd), _round_key(fin, nd))
            buckets.setdefault(key, {"groups": [], "init": ini, "final": fin})
            buckets[key]["groups"].append(it["group"])

        # keep deterministic order: sort by bar length descending then label
        merged = list(buckets.values())
        merged.sort(key=lambda d: (-max(d["init"], d["final"]), ",".join(d["groups"])))
        return merged

    def _groups_label(gs):
        # nicer label in the plot when merged
        if set(gs) == set(["NONREG_LOW","NONREG_MOD","REG_LOW","REG_MOD"]):
            return "All groups"
        # short names
        short = {
            "NONREG_LOW": "Non-reg + low",
            "NONREG_MOD": "Non-reg + mod",
            "REG_LOW": "Reg + low",
            "REG_MOD": "Reg + mod",
        }
        return ", ".join(short[g] for g in gs)


    GROUP_ORDER = ["NONREG_LOW", "NONREG_MOD", "REG_LOW", "REG_MOD"]

    # ---------------- helpers ----------------
    def group_key_from_initial(init_admin, init_trust, ag):
        admin0 = init_admin.get(ag, "")
        trust0 = init_trust.get(ag, "")
        admin = "NONREG" if str(admin0) == "non-registered" else "REG"
        trust = "LOW" if str(trust0) == "LOW_TRUST" else "MOD"
        return f"{admin}_{trust}"

    def _get_locations(env):
        ctx = getattr(env, "context", None)
        return getattr(ctx, "locations", {}) if ctx is not None else {}

    def _draw_social_workers(ax, env):
        if not (show_social_workers and hasattr(env, "socserv_agents") and env.socserv_agents):
            return
        n_sw = len(env.socserv_agents)
        for k, sw in enumerate(env.socserv_agents):
            x, y = sw.location
            jitter = 0.10
            ang = 2 * np.pi * (k / max(1, n_sw))
            dx = jitter * np.cos(ang)
            dy = jitter * np.sin(ang)
            ax.scatter(x + 0.5 + dx, y + 0.5 + dy, s=50, color="grey", edgecolors="none", zorder=2)

    def _cap_init_final_by_group(bh_trace, af_trace, init_admin, init_trust):
        bh_init = {g: [] for g in GROUP_ORDER}
        bh_final = {g: [] for g in GROUP_ORDER}
        af_init = {g: [] for g in GROUP_ORDER}
        af_final = {g: [] for g in GROUP_ORDER}

        for ag in bh_trace.keys():
            g = group_key_from_initial(init_admin, init_trust, ag)
            if g not in bh_init:
                continue
            bh_seq = bh_trace.get(ag, [])
            af_seq = af_trace.get(ag, [])
            if len(bh_seq) > 0:
                bh_init[g].append(float(bh_seq[0]))
                bh_final[g].append(float(bh_seq[-1]))
            if len(af_seq) > 0:
                af_init[g].append(float(af_seq[0]))
                af_final[g].append(float(af_seq[-1]))

        def m(v): 
            return float(np.mean(v)) if len(v) else np.nan

        bh_i = {g: m(bh_init[g]) for g in GROUP_ORDER}
        bh_f = {g: m(bh_final[g]) for g in GROUP_ORDER}
        af_i = {g: m(af_init[g]) for g in GROUP_ORDER}
        af_f = {g: m(af_final[g]) for g in GROUP_ORDER}
        return bh_i, bh_f, af_i, af_f

    def _functionings_shares(health_trace, admin_trace):
        agents = list(health_trace.keys())
        if not agents:
            return 0.0, 0.0, 0.0, 0.0

        h0, hT, a0, aT = [], [], [], []
        for ag in agents:
            hs = health_trace.get(ag, [])
            ad = admin_trace.get(ag, [])
            if len(hs) == 0 or len(ad) == 0:
                continue
            h0.append(float(hs[0]));   hT.append(float(hs[-1]))
            a0.append(int(ad[0]));     aT.append(int(ad[-1]))

        if not h0:
            return 0.0, 0.0, 0.0, 0.0

        init_healthy = float(np.mean(np.array(h0) >= healthy_threshold))
        final_healthy = float(np.mean(np.array(hT) >= healthy_threshold))
        init_reg = float(np.mean(np.array(a0) == 1))
        final_reg = float(np.mean(np.array(aT) == 1))
        return init_healthy, final_healthy, init_reg, final_reg

    def _fmt_pct(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{100.0*float(x):.0f}%"

    def _bar_with_text(ax, y, value, color, text, height=0.32, alpha=0.85):
        ax.barh(y, value, height=height, color=color, alpha=alpha)
        # text inside (or slightly outside if very small)
        x_text = min(max(value, 0.02), xlim[1] - 0.02)
        ha = "right" if value > 0.18 else "left"
        ax.text(
            x_text, y, text,
            va="center", ha=ha,
            fontsize=font_small, color="black"
        )

    def _draw_grid(ax, env, title):
        size = getattr(env, "size", 8)
        locs = _get_locations(env)

        ax.set_xlim(0, size); ax.set_ylim(0, size)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks(np.arange(0, size+1)); ax.set_yticks(np.arange(0, size+1))
        ax.grid(True, color="black", linewidth=1)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_title(title, fontsize=font_big, pad=8)

        colour_map = {"PHC": "#d0d0ff", "ICU": "#7fa8ff", "SocialService": "#f0f0f0"}
        label_map  = {"PHC": "PHC", "ICU": "ICU", "SocialService": "Social\nServices"}

        for name, info in (locs or {}).items():
            base = np.array(info["pos"])
            w, h = info.get("size", (1, 1))
            rect = plt.Rectangle(base, w, h,
                                 facecolor=colour_map.get(name, "#dddddd"),
                                 edgecolor="black", linewidth=1.5, zorder=2)
            ax.add_patch(rect)
            ax.text(base[0] + 0.1, base[1] + 0.4, label_map.get(name, name),
                    fontsize=font_small, va="top", ha="left", zorder=2)

        _draw_social_workers(ax, env)

        # PEH finals: only linestyle by admin final, fill by health
        for ag in env.possible_agents:
            idx = env.agent_name_mapping[ag]
            peh = env.peh_agents[idx]
            x, y = peh.location
            face = health_to_color(peh.health_state, alpha=0.95)
            is_reg = (peh.administrative_state == "registered")
            ls = "-" if is_reg else "--"
            circ = plt.Circle((x+0.5, y+0.5), radius=0.35,
                              facecolor=face, edgecolor="black",
                              linewidth=2.0, linestyle=ls, zorder=3)
            ax.add_patch(circ)
        
            # PEH finals: only linestyle by admin final, fill by health
        for ag in env.possible_agents:
            idx = env.agent_name_mapping[ag]
            peh = env.peh_agents[idx]
            x, y = peh.location
            face = health_to_color(peh.health_state, alpha=0.95)
            is_reg = (peh.administrative_state == "registered")
            ls = "-" if is_reg else "--"
            circ = plt.Circle(
                (x+0.5, y+0.5), radius=0.35,
                facecolor=face, edgecolor="black",
                linewidth=2.0, linestyle=ls, zorder=3
            )
            ax.add_patch(circ)

            # ---- small legend for color and linestyle ----
            legend_elements = [
                Patch(facecolor=health_to_color(4.0, alpha=0.95),
                    edgecolor="black", label="healthy"),
                Patch(facecolor=health_to_color(1.0, alpha=0.95),
                    edgecolor="black", label="hospitalized"),
                Line2D([0], [0], color="black", linestyle="-",
                    linewidth=2.0, label="registered"),
                Line2D([0], [0], color="black", linestyle="--",
                    linewidth=2.0, label="non-registered"),
            ]

        ax.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(-0.02, 0.80),
            fontsize=font_small,
            frameon=True,
            borderpad=0.4,
            handlelength=2.5,
        )

    def _draw_right(ax_caps, ax_fun, ax_cost, env,
                    bh_trace, af_trace, health_trace, admin_trace,
                    init_admin, init_trust,
                    init_health_budget, init_social_budget):
        GROUP_LABELS_LONG = {
            "NONREG_LOW": "Non-reg + low trust",
            "NONREG_MOD": "Non-reg + mod trust",
            "REG_LOW":    "Reg + low trust",
            "REG_MOD":    "Reg + mod trust",
        }
        # ----- Capabilities -----
        ax_caps.set_title("Capabilities (agents' actions)", loc="left",
                        fontsize=font_med, pad=4, fontweight="bold")

        # keep your x padding so bars don't touch the frame
        ax_caps.set_xlim(0.0, 1.15)
        ax_caps.grid(axis="x", alpha=0.15)
        ax_caps.set_xticks([])

        # green if improves/keeps, red if worse
        c_good = health_to_color(4.0, alpha=0.95)   # your green-ish
        c_bad  = health_to_color(1.0, alpha=0.95)   # your red-ish

        def _bar_color(init_v, fin_v):
            return c_good if fin_v >= init_v else c_bad

        def _fmt_pct(x):
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return "NA"
            return f"{100.0*float(x):.0f}%"

        def _round_key(v, nd=3):
            if not np.isfinite(v):
                return None
            return round(float(v), nd)

        def _merge_equal_groups(items, nd=3):
            buckets = {}
            for it in items:
                ini = 0.0 if not np.isfinite(it["init"]) else float(it["init"])
                fin = 0.0 if not np.isfinite(it["final"]) else float(it["final"])
                key = (_round_key(ini, nd), _round_key(fin, nd))
                buckets.setdefault(key, {"groups": [], "init": ini, "final": fin})
                buckets[key]["groups"].append(it["group"])
            merged = list(buckets.values())
            merged.sort(key=lambda d: (-max(d["init"], d["final"]), ",".join(d["groups"])))
            return merged

        def _groups_label(gs):
            if set(gs) == set(["NONREG_LOW","NONREG_MOD","REG_LOW","REG_MOD"]):
                return "All groups"
            short = {
                "NONREG_LOW": "Non-reg + low",
                "NONREG_MOD": "Non-reg + mod",
                "REG_LOW": "Reg + low",
                "REG_MOD": "Reg + mod",
            }
            return ", ".join(short[g] for g in gs)

        # group means (your existing helper)
        groups, bh_init, bh_final, af_init, af_final = _cap_agg_by_group(
            bh_trace, af_trace, init_admin, init_trust
        )

        # items per capability
        bh_items = [{"group": g, "init": bh_init[i], "final": bh_final[i]} for i, g in enumerate(groups)]
        af_items = [{"group": g, "init": af_init[i], "final": af_final[i]} for i, g in enumerate(groups)]

        # merged buckets for Affiliation (can be multiple rows)
        af_merged = _merge_equal_groups(af_items, nd=3)

        # --- compact vertical layout ---
        # We position BH around y=1 and AF around y=0, then AUTO-tighten ylim
        y_BH, y_AF = 1.0, 0.0
        max_lines = max(len(bh_merged), len(af_merged), 1)

        if max_lines == 1:
            offsets = np.array([0.0])
        else:
            # tighter than before to reduce height
            offsets = np.linspace(+0.16, -0.16, max_lines)

        bar_h = 0.12  # slightly thinner to save space

        # Track all y positions we actually used so we can set a tight ylim
        used_y = []

        def _draw_cap_row(y_center, merged_list):
            for k, d in enumerate(merged_list):
                yi = y_center + offsets[k]
                used_y.append(yi)

                ini = 0.0 if not np.isfinite(d["init"]) else float(d["init"])
                fin = 0.0 if not np.isfinite(d["final"]) else float(d["final"])
                v = max(ini, fin)

                col = _bar_color(ini, fin)

                # your original readable style: filled bar, text inside
                ax_caps.barh(yi, v, height=bar_h, color=col, alpha=0.75)

                label = _groups_label(d["groups"])
                txt = f"{label}  |  Initial: {_fmt_pct(ini)}   Final: {_fmt_pct(fin)}"
                x_text = min(max(v, 0.03), 1.12)
                ha = "right" if v > 0.35 else "left"
                ax_caps.text(x_text, yi, txt, va="center", ha=ha,
                            fontsize=font_small, color="black")

        # --- Bodily Health as a single aggregated bar (no blank space) ---
        # Aggregate across groups using nanmean
        bh_inis  = [v for v in bh_init if np.isfinite(v)] if isinstance(bh_init, (list, tuple)) else [bh_init[i] for i in range(len(groups))]
        bh_fins  = [v for v in bh_final if np.isfinite(v)] if isinstance(bh_final, (list, tuple)) else [bh_final[i] for i in range(len(groups))]
        try:
            ini_all = float(np.nanmean(bh_inis)) if len(bh_inis) else 0.0
        except Exception:
            ini_all = 0.0
        try:
            fin_all = float(np.nanmean(bh_fins)) if len(bh_fins) else 0.0
        except Exception:
            fin_all = 0.0

        v_all = max(ini_all, fin_all)
        col_all = _bar_color(ini_all, fin_all)
        used_y.append(y_BH)
        ax_caps.barh(y_BH, v_all, height=bar_h, color=col_all, alpha=0.75)
        txt_all = f"All groups  |  Initial: {_fmt_pct(ini_all)}   Final: {_fmt_pct(fin_all)}"
        x_text = min(max(v_all, 0.03), 1.12)
        ha = "right" if v_all > 0.35 else "left"
        ax_caps.text(x_text, y_BH, txt_all, va="center", ha=ha, fontsize=font_small, color="black")

        # --- Affiliation rows (merged by identical values) ---
        _draw_cap_row(y_AF, af_merged)

        # y tick labels (with line break)
        ax_caps.set_yticks([y_BH, y_AF])
        ax_caps.set_yticklabels(["Bodily\nhealth", "Affiliation"], fontsize=font_small)

        # --- THIS is the key: remove blank space between BH and AF by tightening ylim ---
        if used_y:
            y_min = min(used_y) - (bar_h/2 + 0.08)
            y_max = max(used_y) + (bar_h/2 + 0.08)
            ax_caps.set_ylim(y_min, y_max)


        # ----- Functionings -----
        # ----- Functionings -----
        ax_fun.set_title("Functionings (agents' state)", loc="left", fontsize=font_med, pad=4, fontweight="bold")
        ax_fun.set_xlim(*xlim)
        ax_fun.set_ylim(0.0, 1.0)          # <- important: compacte vertical
        ax_fun.grid(axis="x", alpha=0.15)

        ax_fun.set_xlabel("Population (%)", fontsize=font_small)
        ax_fun.tick_params(axis="x", labelbottom=False)
        ax_fun.set_xticks([])

        init_healthy, final_healthy, init_reg, final_reg = _functionings_shares(health_trace, admin_trace)

        # colors using health extremes (green/red)
        c_bad  = health_to_color(1.0, alpha=0.95)
        c_good = health_to_color(4.0, alpha=0.95)

        # choose bar length as max(init, final)
        healthy_v = final_healthy
        reg_v     = final_reg

        healthy_col = c_good if final_healthy >= init_healthy else c_bad
        reg_col     = c_good if final_reg >= init_reg else c_bad

        # y positions close together (this is the key)
        y_healthy = 0.62
        y_reg     = 0.34
        bar_h     = 0.22

        _bar_with_text(
            ax_fun, y_healthy, healthy_v, healthy_col,
            f"Initial: {_fmt_pct(init_healthy)}   Final: {_fmt_pct(final_healthy)}",
            height=bar_h, alpha=0.85
        )
        _bar_with_text(
            ax_fun, y_reg, reg_v, reg_col,
            f"Initial: {_fmt_pct(init_reg)}   Final: {_fmt_pct(final_reg)}",
            height=bar_h, alpha=0.85
        )

        ax_fun.set_yticks([y_healthy, y_reg])
        ax_fun.set_yticklabels(["Healthy", "Registered"], fontsize=font_small)
        ax_fun.margins(y=0.0)

        # ----- Costs (delta final - initial; negative if spend) -----
        ctx = getattr(env, "context", None)
        fin_h = float(getattr(ctx, "healthcare_budget", init_health_budget))
        fin_s = float(getattr(ctx, "social_service_budget", init_social_budget))

        delta_s = fin_s - float(init_social_budget)
        delta_h = fin_h - float(init_health_budget)

        ax_cost.axis("off")
        # draw bold label and normal-weight values to the right
        lbl = "Economic costs:     "
        vals = f"      Social services = -1500 €  |  Healthcare = {delta_h:+.0f} €"
        
        ax_cost.text(
            0.00, 0.5, lbl,
            ha="left", va="center",
            fontsize=font_med, fontweight="bold",
            transform=ax_cost.transAxes,
        )

        ax_cost.text(
            0.28, 0.5, vals,   # bump this from 0.22 -> 0.28 (or tweak)
            ha="left", va="center",
            fontsize=font_med, alpha=0.9,
            transform=ax_cost.transAxes,
        )
    # ---------------- figure layout ----------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[grid_width_ratio, right_width_ratio],
        wspace=wspace,
        hspace=hspace
    )

    # ON row
    ax_grid_on = fig.add_subplot(gs[0, 0])
    gs_r_on  = gs[0, 1].subgridspec(3, 1, height_ratios=[0.90, 0.90, 0.15], hspace=0.55)
    gs_r_off = gs[1, 1].subgridspec(3, 1, height_ratios=[1.50, 0.90, 0.15], hspace=0.55)

    ax_caps_on = fig.add_subplot(gs_r_on[0, 0])
    ax_fun_on  = fig.add_subplot(gs_r_on[1, 0])
    ax_cost_on = fig.add_subplot(gs_r_on[2, 0])

    # OFF row
    ax_grid_off = fig.add_subplot(gs[1, 0])
    #gs_r_off = gs[1, 1].subgridspec(3, 1, height_ratios=[1.7, 1.00, 0.28], hspace=0.60)
    ax_caps_off = fig.add_subplot(gs_r_off[0, 0])
    ax_fun_off  = fig.add_subplot(gs_r_off[1, 0])
    ax_cost_off = fig.add_subplot(gs_r_off[2, 0])

    # ---------------- draw ----------------
    _draw_grid(ax_grid_on, env_on, title_on)
    _draw_right(ax_caps_on, ax_fun_on, ax_cost_on,
                env_on, bh_on, af_on, health_on, admin_on,
                init_admin_on, init_trust_on,
                init_health_budget_on, init_social_budget_on)

    _draw_grid(ax_grid_off, env_off, title_off)
    _draw_right(ax_caps_off, ax_fun_off, ax_cost_off,
                env_off, bh_off, af_off, health_off, admin_off,
                init_admin_off, init_trust_off,
                init_health_budget_off, init_social_budget_off)

    plt.show()
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


pd.DataFrame(train_episode_rows).to_csv(os.path.join(OUTDIR, "training_episodes.csv"), index=False)
pd.DataFrame(eval_step_rows).to_csv(os.path.join(OUTDIR, "eval_steps.csv"), index=False)
pd.DataFrame(eval_policyON_rows).to_csv(os.path.join(OUTDIR, "eval_policyON_agents.csv"), index=False)
pd.DataFrame(eval_policyOFF_rows).to_csv(os.path.join(OUTDIR, "eval_policyOFF_agents.csv"), index=False)


print("Saved datasets to:", OUTDIR)

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUTDIR = os.path.join("out_models", f"run_{RUN_ID}")
os.makedirs(OUTDIR, exist_ok=True)

np.save(os.path.join(OUTDIR, "q_tables_advice_ON.npy"),  q_tables_advice_ON,  allow_pickle=True)
np.save(os.path.join(OUTDIR, "q_tables_advice_OFF.npy"), q_tables_advice_OFF, allow_pickle=True)

print("Saved Q-tables to:", OUTDIR)

OUTDIR = "out_datasets"
run_id = "run_001"   # o timestamp

# -------------------------
# EVAL POLICY ON (ex: legal norm ON)
# -------------------------
ctx_on = Context(grid_size=size)
ctx_on.set_scenario(universal_health=False)   # <-- segons el teu criteri "ON"

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
# EVAL POLICY OFF (universal health)
# -------------------------
ctx_off = Context(grid_size=size)
ctx_off.set_scenario(universal_health=True)   # <-- "OFF" (universal health)

eval_env_off = GridMAInequityEnv(
    context=ctx_off, render_mode=None, size=size,
    num_peh=len(profiles), num_social_agents=num_sw, peh_profiles=profiles
)

cfg_off = EvalLogConfig(
    scenario_name="universal_health",
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

