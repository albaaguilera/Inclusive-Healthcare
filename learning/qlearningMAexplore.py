# ─────────────────────────────────────────────────────────────
# SCRIPT: 2 AGENTS (REGISTERED / NON-REGISTERED)
# Q-learning + avaluació greedy + plots
# + Capability Expansion agrupant a2,a4,a5 i separant a1/a3 (dashed si impossibles)
# ─────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from CASimulation.environment.model import GridMAInequityEnv
from environment.context import Context, ACTION_NAMES, Actions

# ── paràmetres bàsics ────────────────────────────────────────
size = 6
universal_health = False  # True: legal norm OFF, False: legal norm ON

# --- mida dels rombos i layout vertical (un sol lloc) ---
NODE_SCALE = 7.0  # 6–9 dóna bons resultats

LAYOUT_TIGHT = dict(
    y_group=0.76, y_a2=0.46, y_combo=0.34, y_a1=0.22,
    bh_y=0.93,    # ← dins el marc
    af_y=0.88,    # ← dins el marc
    ylim=(0.06, 1.02)
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

# ── helpers ──────────────────────────────────────────────────
def get_state(obs, peh):
    x, y = obs["agent"]
    h    = obs["health_state"]
    h_idx   = int(round((h - peh.min_health) / peh.health_step))
    admin   = int(peh.administrative_state == "non-registered")
    counter = peh.engagement_counter
    return (x, y, h_idx, admin, counter)

def moving_average(arr, w=10):
    return np.convolve(arr, np.ones(w)/w, mode="valid")

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

# grouped label (a2,a4,a5) -> (bar{a1}, bar{a2}, a5)
GROUP_LABEL = r"$\bar{a}_1,\ \bar{a}_2,\ a_5$"

# ── crea entorn ──────────────────────────────────────────────
ctx = Context(grid_size=size)
ctx.set_scenario(universal_health=universal_health)
env = GridMAInequityEnv(context=ctx, render_mode=None, size=size)  # sense render durant training

# ── Q-tables per agent ───────────────────────────────────────
n_health = int((env.peh_agents[0].max_health - env.peh_agents[0].min_health) / env.peh_agents[0].health_step) + 1
q_tables_ut = {}
q_tables_eg = {}
for a in env.possible_agents:
    shape = (env.size, env.size, n_health, 2, 4, env.action_space(a).n)
    q_tables_ut[a] = np.zeros(shape)
    q_tables_eg[a] = np.zeros(shape)

# ── hiperparàmetres ──────────────────────────────────────────
episodes = 400
alpha, gamma = 0.2, 0.99
epsilon, eps_min, eps_decay = 0.1, 0.01, 0.995
max_steps = 150

episode_returns_ut = []
episode_returns_reg = []
episode_returns_nonreg = []

# ── entrenament ──────────────────────────────────────────────
for ep in range(episodes):
    env.reset()
    obs = {a: env.observe(a) for a in env.agents}
    state = {a: get_state(obs[a], env.peh_agents[env.agent_name_mapping[a]]) for a in env.agents}
    ep_ret_ut = {a: 0.0 for a in env.agents}

    for _ in range(max_steps):
        agent = env.agent_selection
        if env.dones[agent]:
            env.step(None)
            if len(env.agents) == 0:
                break
            continue

        # ε-greedy
        if np.random.rand() < epsilon:
            action = env.action_space(agent).sample()
        else:
            action = int(np.argmax(q_tables_ut[agent][state[agent]]))

        # pas
        env.step(action)

        # nou estat i reward
        new_obs = env.observe(agent)
        individualreward = env.rewards[agent]
        next_state = get_state(new_obs, env.peh_agents[env.agent_name_mapping[agent]])

        # Q-update
        s = state[agent]; a = action
        best_next = np.max(q_tables_ut[agent][next_state])
        q_tables_ut[agent][s + (a,)] += alpha * (individualreward + gamma * best_next - q_tables_ut[agent][s + (a,)])

        state[agent] = next_state
        ep_ret_ut[agent] += individualreward

        if len(env.agents) == 0:
            break

    # bookkeeping
    episode_returns_ut.append(sum(ep_ret_ut.values()))
    epsilon = max(eps_min, epsilon * eps_decay)

    # rewards per registered / non-registered
    reg_reward_total = 0.0
    nonreg_reward_total = 0.0
    for agent, reward in ep_ret_ut.items():
        idx = env.agent_name_mapping[agent]
        is_registered = (env.peh_agents[idx].administrative_state == "registered")
        if is_registered:
            reg_reward_total += reward
        else:
            nonreg_reward_total += reward
    episode_returns_reg.append(reg_reward_total)
    episode_returns_nonreg.append(nonreg_reward_total)

# ── plots d'aprenentatge ─────────────────────────────────────
plt.figure(figsize=(10, 4))
ut_ma = moving_average(episode_returns_ut, 10)
plt.plot(episode_returns_ut, alpha=0.25)
plt.plot(np.arange(len(ut_ma)) + 9, ut_ma, lw=2)
plt.xlabel("Episode step", fontsize=14)
plt.ylabel("Aggregated Rewards", fontsize=14)
plt.legend(); plt.show()

# Reg vs Non-reg
ma_nonreg = moving_average(episode_returns_nonreg, 10)
ma_reg = moving_average(episode_returns_reg, 10)

plt.figure(figsize=(10, 4))
plt.plot(episode_returns_nonreg, alpha=0.25, linestyle="-", label="Non-registered PEH", color="#FFD580")  # lighter orange
plt.plot(ma_nonreg, linestyle="-", color= "tab:orange")
plt.plot(episode_returns_reg, alpha=0.25, linestyle="-", label="Registered PEH", color="#ADD8E6")  # lighter blue
plt.plot(ma_reg, linestyle="-", color= "tab:blue")
plt.xlabel("Episode step", fontsize=14)
plt.ylabel("Rewards", fontsize=14)
plt.legend(fontsize=13)
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
#  GREEDY EVALUATION — CAPABILITIES & POLICY
# ─────────────────────────────────────────────────────────────
ctx = Context(grid_size=size)
ctx.set_scenario(universal_health=universal_health)
eval_env = GridMAInequityEnv(context=ctx, render_mode=None, size=size)
eval_env.reset()

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
for ag in eval_env.possible_agents:
    idx = eval_env.agent_name_mapping[ag]
    peh_agent = eval_env.peh_agents[idx]
    initial_admin_state[ag] = peh_agent.administrative_state
    health_state_trace[ag].append(peh_agent.health_state)
    admin_state_trace[ag].append(1 if peh_agent.administrative_state == "registered" else 0)

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

while eval_env.agents:
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
    action = int(np.argmax(q_tables_ut[agent][state[agent]]))
    eval_env.step(action)

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

# ─────────────────────────────────────────────────────────────
#  Estratègia òptima per step (sense "apply_for_shelter")
# ─────────────────────────────────────────────────────────────
ACTION_LUT = list(ACTION_NAMES.values())
colors  = {ag: col for ag, col in zip(exec_hist.keys(),
                                      plt.rcParams['axes.prop_cycle'].by_key()['color'])}
markers = {"registered": "o", "non-registered": "x"}

apply_for_shelter_idx = list(ACTION_NAMES.keys()).index(Actions.APPLY_AND_GET_SHELTER)

plt.figure(figsize=(10, 4))
for ag, seq in exec_hist.items():
    for t, a_idx, status in seq:
        if a_idx == apply_for_shelter_idx:
            continue
        marker = markers[status]
        if marker == "o":
            plt.scatter(t, a_idx,
                        color=colors[ag],
                        marker=marker,
                        s=90, edgecolors="k",
                        label=f"{status} PEH" if t == 0 else None)
        else:
            plt.scatter(t, a_idx,
                        color=colors[ag],
                        marker=marker,
                        s=90,
                        label=f"{status} PEH" if t == 0 else None)

# etiquetes compactes a₁..a₅ si n'hi ha 5
num_actions = env.action_space(env.possible_agents[0]).n
plt.yticks(range(num_actions), [a_label(i) for i in range(num_actions)], fontsize=14)
plt.xlabel("Simulation step", fontsize=14)
plt.grid(True, axis="y", alpha=.3); plt.legend(ncol=1, fontsize=13)
plt.tight_layout(); plt.show()

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

plot_central_functionings(health_state_trace, admin_state_trace, initial_admin_state)

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

def _plot_optimal_strategy(ax, exec_hist, num_actions, apply_for_shelter_idx, exclude_idxs=None):
    if exclude_idxs is None:
        exclude_idxs = []
    exclude = set(exclude_idxs + [apply_for_shelter_idx])  # <- treu també a₅

    markers = {"registered": "o", "non-registered": "x"}
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_by_agent = {ag: c for ag, c in zip(exec_hist.keys(), palette)}

    # pinta punts excepte els exclosos
    for ag, seq in exec_hist.items():
        for t_global, a_idx, status in seq:
            if a_idx in exclude:
                continue
            m = markers.get(status, "o")
            if m == "o":
                ax.scatter(t_global, a_idx, color=colors_by_agent[ag], marker=m, s=35, edgecolors="k")
            else:
                ax.scatter(t_global, a_idx, color=colors_by_agent[ag], marker=m, s=35)

    # y-ticks només pels que queden
    visible = [i for i in range(num_actions) if i not in exclude]
    ax.set_yticks(visible)
    ax.set_yticklabels([a_label(i) for i in visible])

    ax.set_xlabel("Simulation step")
    ax.set_title("(C) Optimal strategy")
    ax.grid(True, axis="y", alpha=.3)


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
    _plot_learning_curve(axs[0,0], episode_returns_ut)

    # 2) split reg/non-reg
    _plot_split_rewards(axs[0,1], episode_returns_reg, episode_returns_nonreg)

    # 3) greedy policy scatter
    num_actions = env.action_space(env.possible_agents[0]).n
    _plot_optimal_strategy(axs[1,0], exec_hist, num_actions, apply_for_shelter_idx, exclude_idxs=[apply_for_shelter_idx])

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
