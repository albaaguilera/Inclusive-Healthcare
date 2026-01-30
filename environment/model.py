from enum import Enum 
import numpy as np
import pygame
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
from .context import Context, Actions, update_all_capability_scores, ACTION_NAMES
from .agent import PEHAgent, SocServAgent
from .render import render_frame
num_peh = 4  # default number of PEH agents
num_social_agents = 10  # default number of social service agents
class GridMAInequityEnv(AECEnv):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 0.15}
    """A simple grid world environment for inequity simulation.
    - render_mode: "human" for visual rendering, "ansi" for text-based rendering.
    - size: Size of the grid (default is 5).
    - agent_profile: Dictionary containing the agent's health, registration and personal state.
    """

    def __init__(self, render_mode=None, size=5, context=None, num_peh=num_peh, num_social_agents=num_social_agents, peh_profiles=None, max_steps=50):
        # CONTEXT AND AGENT
        super().__init__()
        self.context = context if context is not None else Context(grid_size=size)
        self.size = size
        self.num_peh = num_peh
        self.num_social_agents = num_social_agents
        self.step_count = 0
        self.socserv_speed = 5
        self.num_peh = int(num_peh)
        self.peh_profiles = peh_profiles

        self.max_cycles = int(max_steps)

        if self.num_social_agents <= self.num_peh:
            self.social_assignments = np.random.choice(self.num_peh, size=self.num_social_agents, replace=False)
        else:
            self.social_assignments = np.random.choice(self.num_peh, size=self.num_social_agents, replace=True)

        self.possible_agents = [f"peh_{i}" for i in range(self.num_peh)]
        self.peh_agents = [PEHAgent(start_loc=np.array([-1, -1], dtype=int)) for _ in range(self.num_peh)]
        self.agents = [f"peh_{i}" for i in range(self.num_peh)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.agents)}
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()
        self.possible_actions_history = {agent: [] for agent in self.agents}
        self.impossible_actions_history = {agent: [] for agent in self.agents}
        self.capabilities = {agent: {} for agent in self.agents}
        self.functionings = {agent: {} for agent in self.agents}

        self.current_possible_actions   = {ag: [] for ag in self.agents}
        self.current_impossible_actions = {ag: [] for ag in self.agents}


        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.render_mode = render_mode

        # Observations and actions spaces
        self.observation_spaces = {
            agent: spaces.Dict({
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "health_state": spaces.Box(1.0, 5.0, shape=(), dtype=float)
            }) for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(5) for agent in self.agents
        }

        # TRANSITION PROBABILITIES
        # self.health_P = [self.context.build_transition_table(agent)
        #          for agent in self.peh_agents]

    def _is_adjacent_to_social_agent(self, agent):
        ax, ay = agent.location
        for soc_agent in self.socserv_agents:
            sx, sy = soc_agent.location
            d = max(abs(ax - sx), abs(ay - sy))   # Chebyshev
            if d == 1:                            # IMPORTANT: només costat/diagonal
                return True
        return False

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _get_obs(self):
        return [{
        "agent": np.array(agent.location, dtype=int),
        "health_state": np.array(agent.health_state, dtype=float),
    } for agent in self.peh_agents]
    
    def _get_info(self):
        info_list = []
        for name in self.agents:                    # <- ja no fem servir i
            idx   = self.agent_name_mapping[name]
            agent = self.peh_agents[idx]

            info_list.append({
                "agent": name,
                "administrative_state": agent.administrative_state,
                "engagement_counter": agent.engagement_counter,
                "health_state": agent.health_state,
                "housing_state": agent.housing_state,
                "last_action": (
                    self.possible_actions_history[name][-1].name
                    if self.possible_actions_history[name] else None
                ),
                "possible_actions": [
                    a.name for a in self.possible_actions_history[name]
                ],
                "impossible_actions": [
                    a.name for a in self.impossible_actions_history[name]
                ],
                "capabilities": self.capabilities[name],
                "functionings": self.functionings[name],
            })
        return info_list

    def observe(self, agent):
        a = self.peh_agents[self.agent_name_mapping[agent]]
        return {
            "agent": np.array(a.location, dtype=int),
            "health_state": np.array(a.health_state, dtype=float),
            "administrative_state": a.administrative_state,
            "adjacent_to_social_agent": self._is_adjacent_to_social_agent(a),
        }

    def reset(self, seed=None, options=None):
        # -----------------------------
        # RNG + bookkeeping base
        # -----------------------------
        self.np_random = np.random.default_rng(seed)
        rng = self.np_random

        self.possible_agents = [f"peh_{i}" for i in range(self.num_peh)]
        self.agents = self.possible_agents[:]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        self.agent_name_mapping = {name: i for i, name in enumerate(self.agents)}

        self.dones = {a: False for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}

        self.step_count = 0
        self.possible_actions_history = {agent: [] for agent in self.agents}
        self.impossible_actions_history = {agent: [] for agent in self.agents}
        self.capabilities = {agent: {} for agent in self.agents}
        self.functionings = {agent: {} for agent in self.agents}
        self.current_possible_actions   = {ag: [] for ag in self.agents}
        self.current_impossible_actions = {ag: [] for ag in self.agents}

        # -----------------------------
        # Forbidden cells = service rectangles
        # -----------------------------
        forbidden = set()
        for info in self.context.locations.values():
            pos = np.array(info["pos"], dtype=int)
            size = info.get("size", (1, 1))
            for dx in range(int(size[0])):
                for dy in range(int(size[1])):
                    forbidden.add(tuple((pos + np.array([dx, dy], dtype=int)).tolist()))

        # Free cells (not forbidden)
        all_cells = [(i, j) for i in range(self.size) for j in range(self.size)
                    if (i, j) not in forbidden]

        if len(all_cells) < self.num_peh:
            raise ValueError(
                f"Not enough free cells to place PEH: free={len(all_cells)} < num_peh={self.num_peh}. "
                f"Increase grid size or reduce forbidden/service area."
            )

        chosen_indices = rng.choice(len(all_cells), size=self.num_peh, replace=False)

        # -----------------------------
        # Profiles (from options or self)
        # -----------------------------
        profiles = None
        if isinstance(options, dict):
            profiles = options.get("peh_profiles", None)
        if profiles is None:
            profiles = getattr(self, "peh_profiles", None)

        # -----------------------------
        # Create PEH agents
        # -----------------------------
        self.peh_agents = []
        for i, idx in enumerate(chosen_indices):
            loc = np.array(all_cells[int(idx)], dtype=int)

            if profiles is not None and i < len(profiles):
                prof = dict(profiles[i])

                admin = str(prof.get("admin_state", "non-registered")).upper().replace("-", "_")
                admin = "registered" if admin == "REGISTERED" else "non-registered"

                trust  = str(prof.get("trust_type", "MODERATE_TRUST"))
                health = float(prof.get("health_state", 3.0))
                income = float(prof.get("income", 400.0))

                attrs = {
                    "age": int(prof.get("age", 40)),
                    "gender": str(prof.get("gender", "female")),
                    "homelessness_duration": int(prof.get("homelessness_duration", 2)),
                    "history_of_abuse": bool(prof.get("history_of_abuse", False)),
                }

                a = PEHAgent(
                    start_loc=loc,
                    health=health,
                    admin_state=admin,
                    trust_type=trust,
                    personal_attributes=attrs,
                    income=income,
                )
            else:
                a = PEHAgent(start_loc=loc)

            self.peh_agents.append(a)

        # Transition tables for each PEH (health)
        self.health_P = [self.context.build_transition_table(agent) for agent in self.peh_agents]
        # -----------------------------
        # Social workers: spawn near PEH (optional)
        # -----------------------------
        spawn_near = True
        spawn_radius = 1  
        if isinstance(options, dict):
            spawn_near = bool(options.get("spawn_sw_near_peh", True))
            spawn_radius = int(options.get("sw_spawn_radius", 1))

        if self.num_social_agents <= self.num_peh:
            self.social_assignments = rng.choice(self.num_peh, size=self.num_social_agents, replace=False)
        else:
            self.social_assignments = rng.choice(self.num_peh, size=self.num_social_agents, replace=True)

        def _adjacent_cells(center_xy, R=1):
            cx, cy = int(center_xy[0]), int(center_xy[1])
            out = []
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    if dx == 0 and dy == 0:
                        continue
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.size and 0 <= y < self.size:
                        d = max(abs(dx), abs(dy))
                        if d <= R:
                            out.append((x, y))
            rng.shuffle(out)
            return out

        peh_cells = set(tuple(peh.location.tolist()) for peh in self.peh_agents)

        self.socserv_agents = []
        occupied_sw = set() 

        for i in range(self.num_social_agents):
            target_idx = int(self.social_assignments[i])
            peh_loc = self.peh_agents[target_idx].location

            placed = False
            if spawn_near:
                for xy in _adjacent_cells(peh_loc, spawn_radius):
                    if xy in forbidden:
                        continue
                    if xy in peh_cells:
                        continue

                    if xy not in occupied_sw:
                        self.socserv_agents.append(SocServAgent(np.array(xy, dtype=int)))
                        occupied_sw.add(xy)
                        placed = True
                        break

                if not placed:
                    for xy in _adjacent_cells(peh_loc, spawn_radius):
                        if xy in forbidden or xy in peh_cells:
                            continue
                        self.socserv_agents.append(SocServAgent(np.array(xy, dtype=int)))
                        placed = True
                        break

            if not placed:
                # fallback: oficina
                loc = self._service_cell("SocialService")
                self.socserv_agents.append(SocServAgent(loc))

        if self.render_mode == "human":
            render_frame(self)

        return {a: self.observe(a) for a in self.agents}
        
    def _get_health_color(self, agent):
        # Normalize health_state into [0,1]
        h = agent.health_state
        span = agent.max_health - agent.min_health
        ratio = (h - agent.min_health) / span
        ratio = max(0.0, min(1.0, ratio))
        r = int(255 * (1 - ratio))
        g = int(255 * ratio)
        r = max(0, min(255, r))
        g = max(0, min(255, g))

        return (r, g, 0)
    
    def _classify_actions(self, peh: PEHAgent):
        """
        Return two lists with the actions that are possible / impossible
        for the *given* homeless agent in its current state.
        """
        possible, impossible = [], []
        for act in Actions:

            if act == Actions.RECEIVE_MEDICAL_ATTENTION:
                feasible = (peh.administrative_state == "registered"
                or getattr(self.context, "policy_inclusive_healthcare", True))

            elif act == Actions.ENGAGE_SOCIAL_SERVICES:
                feasible = self._is_adjacent_to_social_agent(peh)

            elif act == Actions.APPLY_AND_GET_SHELTER:
                feasible = (
                    self._is_adjacent_to_social_agent(peh)
                    and peh.engagement_counter > 1
                    and self.context.shelters_available > 0
                )

            else:                      # KEEP_FORWARD, REMAIN_DISENGAGED
                feasible = True

            (possible if feasible else impossible).append(act)

        return possible, impossible
    
    def step(self, action):
         # ------------------------------------------------------------
        agent = self.agent_selection          # e.g. "peh_3"
        idx        = self.agent_name_mapping[agent]
        peh        = self.peh_agents[idx]
       
        
        # 0)  LIVE OPPORTUNITY SET *before* any mutations
        poss_now, imposs_now = self._classify_actions(peh)
        self.current_possible_actions  [agent] = poss_now
        self.current_impossible_actions[agent] = imposs_now

        if self.render_mode in ("ansi", "human"):
            print(f"\nStep {self.step_count}")
            # print(f"Agent               : {agent}")
            # print(f"Chosen action       : {Actions(action).name if action is not None else 'None'}")
            # print("Currently possible  :", [a.name for a in poss_now])
            # print("Currently impossible:", [a.name for a in imposs_now])

        # for i, s_agent in enumerate(self.socserv_agents):
        #     target_idx = self.social_assignments[i]
        #     target_loc = self.peh_agents[target_idx].location
        #     forbidden = [a.location for j, a in enumerate(self.socserv_agents) if j != i]
        #     forbidden += [peh.location for peh in self.peh_agents]
            # we move the agent according to the socserv_speed
            # for _ in range(self.socserv_speed):
            #     s_agent.move_towards_peh(target_loc, self.size, forbidden,
            #                             np.random.default_rng())
            #     # we check if the social service agent is adjacent to a PEH agent
            #     if any(np.max(np.abs(s_agent.location - peh.location)) <= 1
            #         for peh in self.peh_agents):
            #         # and go back to the soc serv locaiton once they have interacted
            #         s_agent.location = self._service_cell("SocialService")
            #         break   
            #     # actualitza forbidden perquè el SS ja s’ha mogut
            #     forbidden[i] = s_agent.location
        
        # MOVE SOCIAL SERVICE AGENT 
        # All social service agents start at the SocialService office and move towards their assigned PEH agent.
        for i, s_agent in enumerate(self.socserv_agents):
            target_idx = self.social_assignments[i]
            target_loc = self.peh_agents[target_idx].location

            forbidden = [peh.location for peh in self.peh_agents]  # <-- no trepitjar PEH
            s_agent.move_towards_peh(target_loc, self.size, forbidden, self.np_random)

        adj = self._is_adjacent_to_social_agent(peh)
        if adj:
            peh.encounter_counter += 1
        #charge cost social service
        self.context.charge_cost("SocialServiceWorker")

        if action is None:
            self.dones[agent] = True
            self.agent_selection = self.agent_selector.next()
            return
        idx = self.agent_name_mapping[agent]
        a = self.peh_agents[idx]

        # PEH AGENT ACTION 
        old_h = a.health_state
        admin = a.administrative_state
        adj = 1 if self._is_adjacent_to_social_agent(a) else 0
        key = (old_h, admin, adj, action)  
        transitions = self.health_P[idx][key]
        probs = [p for p, _, _ in transitions]
        chosen = np.random.choice(len(transitions), p=probs)
        _, new_h, reward = transitions[chosen]


        if action == Actions.RECEIVE_MEDICAL_ATTENTION.value:
            if transitions[0][0] == 0.0:
                self.impossible_actions_history[agent].append(Actions.RECEIVE_MEDICAL_ATTENTION)
            else:
                self.possible_actions_history[agent].append(Actions.RECEIVE_MEDICAL_ATTENTION)
                self.context.charge_cost("VisitCAP")

        elif action == Actions.APPLY_AND_GET_SHELTER.value:
            if a.engagement_counter > 1 and self.context.shelters_available > 0:
                self.possible_actions_history[agent].append(Actions.APPLY_AND_GET_SHELTER)
                self.housing_state = "ETHOS1"
                self.context.shelters_available -= 1
            else:
                self.impossible_actions_history[agent].append(Actions.APPLY_AND_GET_SHELTER)

        elif action == Actions.ENGAGE_SOCIAL_SERVICES.value:
            if self._is_adjacent_to_social_agent(a):
                self.possible_actions_history[agent].append(Actions.ENGAGE_SOCIAL_SERVICES)
                if Actions.ENGAGE_SOCIAL_SERVICES in self.impossible_actions_history[agent]:
                    self.impossible_actions_history[agent].remove(Actions.ENGAGE_SOCIAL_SERVICES)
                a.engagement_counter = min(a.engagement_counter + 1, 3)
                if a.engagement_counter >= 2 and a.administrative_state == "non-registered":
                    a.administrative_state = "registered"
            else:
                self.impossible_actions_history[agent].append(Actions.ENGAGE_SOCIAL_SERVICES)
    
        elif action == Actions.REMAIN_DISENGAGED.value:
            if self._is_adjacent_to_social_agent(a):
                a.non_engagement_counter += 1
                self.possible_actions_history[agent].append(Actions.REMAIN_DISENGAGED)
                if a.engagement_counter == 0 and a.administrative_state == "registered":
                    a.administrative_state = "non-registered"
            else:
                self.impossible_actions_history[agent].append(Actions.REMAIN_DISENGAGED)
        elif action == Actions.KEEP_FORWARD.value:
            self.possible_actions_history[agent].append(Actions.KEEP_FORWARD) # this one is always possible

        self.capabilities[agent] = {
            "Being able to have good health": Actions.RECEIVE_MEDICAL_ATTENTION not in self.current_possible_actions[agent],
            "Being able to have adequate shelter": Actions.APPLY_AND_GET_SHELTER not in self.current_possible_actions[agent],
            "Being able to move freely from place to place": False
        }
        self.functionings[agent] = {
            "Having good health": a.health_state > 2,
            "Having adequate shelter": a.housing_state == "ETHOS1"
        }
        
        a.health_state = new_h
        self.rewards[agent] = reward

        eps = 1e-9
        terminated = (new_h <= a.min_health + eps) or (new_h >= a.max_health - eps)
        # charge hospitalization only if terminated at min health
        if terminated and (new_h <= a.min_health + eps):
            self.context.charge_cost("Hospitalization")
        self.terminations[agent] = terminated
        self.dones[agent] = terminated  

        # i el time limit com a truncation, separat
        self.step_count += 1
        if self.step_count >= self.max_cycles:
            for ag in list(self.agents):
                self.truncations[ag] = True
                self.dones[ag] = True
            self.agents = []
            return

        update_all_capability_scores(self)

        # print(f"[Step {self.step_count}] Agent capabilities:")
        # for agent_name in self.agents:
        #     print(f" - {agent_name}: {self.capabilities[agent_name]}")

        #print(f"Agent {agent} — Action: {action}, Health: {a.health_state:.2f}, Reward: {reward:.2f}, Possible: {self.possible_actions_history[agent]}, Impossible: {self.impossible_actions_history[agent]}")
        self.agent_selection = self.agent_selector.next()
        if self.render_mode == "human":
            render_frame(self)

        # ── neteja agents “done” ─────────────────────────────────────
        for ag in list(self.agents):     
            if self.dones[ag]:
                self.agents.remove(ag)


        if not self.agents:
            self.agent_selection = None

    def _service_cell(self, name: str) -> np.ndarray:
        info = self.context.locations[name]
        base = np.array(info["pos"], dtype=int)
        size = info.get("size", (1, 1))
        dx, dy = np.random.randint(0, size[0]), np.random.randint(0, size[1])

        return base + np.array([dx, dy])
    
    def render(self):
        if self.render_mode == "human":
            render_frame(self)
        # elif self.render_mode == "ansi":
        #     print(self._get_info())
        

    def close(self):
        pygame.display.quit()
        pygame.quit()

# Example of rendering the environment
if __name__ == "__main__":
    env = GridMAInequityEnv(render_mode="human", size=7, max_steps=20)
    obs = env.reset()
    done = {a: False for a in env.agents}

    while env.agents:
        for agent in env.agents:
            action = env.action_space(agent).sample()
            env.step(action)
    env.close()