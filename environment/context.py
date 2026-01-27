
# CONTEXT FOR THE ENVIRONMENT
# This file defines the context for the physical and regulatory environment, including resources, synthetic agent profiles (pers conv factors and choice factors), locations (env conv factors), legal norms (social conversion factors).
# Therefore, it includes information about the update of resources and conv factors (technical and social context), 
# cration of transition probabilities defining the possibilities of actions for agents.

import numpy as np
from enum import Enum

grid_size = 7 # Default grid size for the environment

Governmental_costs = {
    "Hospitalization": 1000, #Healthcare EUROS/ every time a person is hospitalized
    "VisitCAP": 30, # Healthcare EUROS/ every time a person receives medical attention
    "SocialServiceWorker": 10  #SocServ EUROS/ for every social service worker added in the simulation
    }

Resources = {
    "Number_shelters": 5, 
    "Number_socialserviceworkers": 5,
    "Healthcare budget": 5000,
    "Social service budget": 5000,
    "Number_hospitalbeds": 10,
    "Number_healthcareworkers": 5,
    "Number_ambulances": 5
    }

Locations = {
    "PHC":           {"pos": (0, 0),     "size": (2, 2)},
    "ICU":      {"pos": (7, 7),     "size": (2, 2)},
    "SocialService": {"pos": (7, 0),     "size": (2, 2)},
}

class Actions(Enum):
    RECEIVE_MEDICAL_ATTENTION = 0, 1, 0.2
    KEEP_FORWARD              = 1, 0.0, 0.0
    ENGAGE_SOCIAL_SERVICES    = 2, 0.0, 0.8
    REMAIN_DISENGAGED         = 3, 0.0, 0.0
    APPLY_AND_GET_SHELTER     = 4, 0, 0.0

    def __new__(cls, value, alpha=0.0, beta=0.0):
        obj = object.__new__(cls)
        obj._value_ = value          # Ensure Enum recognizes the int value
        obj.alpha = alpha            # Custom attribute for Bodily Health
        obj.beta = beta              # Custom attribute for Affiliation
        return obj


ACTION_NAMES = {
    Actions.RECEIVE_MEDICAL_ATTENTION: "Request and receive medical attention",
    Actions.KEEP_FORWARD: "Do not request and do not receive medical attention",
    Actions.ENGAGE_SOCIAL_SERVICES: "Engage with social services",
    Actions.REMAIN_DISENGAGED: "Remain disengaged from social services",
    Actions.APPLY_AND_GET_SHELTER: "Apply for shelter and get it",
}

class Context:
    """
    Includes resources, physical locations, costs, and conversion factors.
    """
    def __init__(self, resources=None, grid_size=10):
        resources = resources or Resources
        self.shelters_available            = resources.get("Number_shelters", 5)
        self.healthcare_budget            = resources.get("Healthcare budget", 5000)
        self.social_service_budget        = resources.get("Social service budget", 5000)
        self.locations                     = Locations.copy()
        self.grid_size                    = grid_size
        self.np_random = np.random
        self._init_locations()

    def _init_locations(self):
        if self.grid_size <= 5:
            # Mides reduïdes per graelles petites
            self.locations = {
                "PHC": {"pos": (0, 0), "size": (1, 1)},
                "ICU": {"pos": (self.grid_size - 1, 0), "size": (1, 1)},
                "SocialService": {"pos": (0, self.grid_size - 1), "size": (1, 1)}
            }
        else:
            # Mides normals per graelles més grans
            self.locations = {
                "PHC": {"pos": (0, 0), "size": (2, 2)},
                "ICU": {"pos": (self.grid_size - 2, 0), "size": (2, 2)},
                "SocialService": {"pos": (0, self.grid_size - 2), "size": (2, 2)}
            }

    def charge_cost(self, cost_key: str):
        """
        Deduct from the appropriate government budget.
        """
        cost = Governmental_costs.get(cost_key, 0)
        if cost_key == "Hospitalization" or cost_key == "VisitCAP":
            self.healthcare_budget = max(0, self.healthcare_budget - cost)
        elif cost_key == "SocialServiceWorker":
            self.social_service_budget = max(0, self.social_service_budget - cost)
    
    def service_location(self, name: str) -> np.ndarray:
        """
        Return a random and unoccupied cell within the named service area.
        Otherwise, 'pos' is the top-left and 'size' is (width, height).
        """
        loc = self.locations[name]
        # single cell
        if isinstance(loc, tuple):
            return np.array(loc)
        # rectangular block
        base = np.array(loc["pos"])
        w, h = loc.get("size", (1, 1))
        dx = self.np_random.randint(0, w)
        dy = self.np_random.randint(0, h)
        return base + np.array([dx, dy])

    def to_dict(self):
        return {
            "healthcare_budget": self.healthcare_budget,
            "social_service_budget": self.social_service_budget,
            "shelters_available": self.shelters_available
        }
    
    # ------- NEW: scenario flags (optional) -----------------
    def set_scenario(self, *, policy_inclusive_healthcare=False):
        """
        Set legal-norm toggles or resource knobs that influence the
        transition probabilities.
        Example: policy_inclusive_healthcare=True ⇒ every PEH gets p_treat = 1.0
        """
        self.policy_inclusive_healthcare = policy_inclusive_healthcare

    # ------- NEW: table builder -----------------------------
    def build_transition_table(self, peh_agent) -> dict:
        """
        Return a dict  health_P[(h, admin, action)] = [(p, next_h, r), …]
        The logic can branch on self.policy_inclusive_healthcare, budgets, etc.
        """
        health_P = {}
        actions = [a.value for a in Actions]
        admin_states = ["registered", "non-registered"]
        adj_states = [0, 1]

        health_states = np.arange(
            peh_agent.min_health,
            peh_agent.max_health + 1e-9,
            peh_agent.health_step
        ).tolist()

        for h in health_states:
            for admin in admin_states:
                for adj in adj_states:    
                    for a in actions:
                        key, trans = (h, admin, adj, a), []

                        if a == Actions.RECEIVE_MEDICAL_ATTENTION.value:
                            # ----- legal norm demo --------------------
                            p_treat = 1.0 if (admin == "registered"
                                            or getattr(self, "policy_inclusive_healthcare", False)) else 0.0
                            nh_good = min(h + peh_agent.health_update, peh_agent.max_health)
                            nh_bad  = max(h - peh_agent.health_step, peh_agent.min_health)
                            r_fail = -1.0 if nh_bad == peh_agent.min_health else -0.05
                            trans = [(p_treat, nh_good, +0.10),
                                    (1.0 - p_treat, nh_bad, r_fail)]

                        else:
                            nh = max(h - peh_agent.health_step, peh_agent.min_health)
                            if a == Actions.ENGAGE_SOCIAL_SERVICES.value:
                                p_eng = 1.0 if adj == 1 else 0.0
                                nh = max(h - peh_agent.health_step, peh_agent.min_health)
                                rew_succ = +0.10 if nh != peh_agent.min_health else -1.0
                                rew_fail = -0.05 if nh != peh_agent.min_health else -1.0
                                trans = [(p_eng, nh, rew_succ), (1.0-p_eng, nh, rew_fail)]
                            else: 
                                rew = -1 if nh == peh_agent.min_health else 0
                                trans = [(1.0, nh, rew)]

                        health_P[key] = trans
            
        return health_P
    
    def update_capability_scores(self, agent):
        ## SINGLE AGENT VERSION
        #print("\n[DEBUG] ---- Updating Capability Scores ----")

        # Step 1: Print current possible actions
        #print("[DEBUG] Possible actions history:")
        # for a in agent.current_possible_actions:
        #     print(f" - {a} (value={a.value})")

        # Step 2: Check whether is_possible is working
        def is_possible(action):
            result = int(action in agent.current_possible_actions)
            #print(f"[DEBUG] Is action '{action.name}' possible? {'Yes' if result else 'No'}")
            return result

        # Step 3: Compute alpha and beta sums and check values
        alphas = [(a, a.alpha) for a in Actions if a.alpha > 0]
        betas = [(a, a.beta) for a in Actions if a.beta > 0]

        #print("\n[DEBUG] Actions contributing to Bodily Health (alpha):")
        # for a, alpha in alphas:
        #     print(f" - {a.name}: alpha = {alpha}")

        # print("\n[DEBUG] Actions contributing to Affiliation (beta):")
        # for a, beta in betas:
        #     print(f" - {a.name}: beta = {beta}")

        sum_alpha = sum(alpha for _, alpha in alphas)
        sum_beta = sum(beta for _, beta in betas)

        # print(f"\n[DEBUG] Sum alpha: {sum_alpha}")
        # print(f"[DEBUG] Sum beta: {sum_beta}")

        # Step 4: Compute capability scores, printing each contribution
        bh_numerator = 0
        for a, alpha in alphas:
            contribution = alpha * is_possible(a)
            bh_numerator += contribution
            #print(f"[DEBUG] BH contribution from {a.name}: {alpha} * possible = {contribution}")

        af_numerator = 0
        for a, beta in betas:
            contribution = beta * is_possible(a)
            af_numerator += contribution
            #print(f"[DEBUG] Affiliation contribution from {a.name}: {beta} * possible = {contribution}")

        bodily_health = bh_numerator / sum_alpha if sum_alpha > 0 else 0
        affiliation = af_numerator / sum_beta if sum_beta > 0 else 0

        # print(f"\n[DEBUG] Final Bodily Health score: {bodily_health}")
        # print(f"[DEBUG] Final Affiliation score: {affiliation}")

        agent.central_capabilities = {
            "Bodily Health": bodily_health,
            "Affiliation": affiliation
        }

def update_all_capability_scores(env):
    """
    Re-compute capability scores for every PEH agent from the
    *current* feasible-action set stored in
        env.current_possible_actions[agent_name]
    """
    sum_alpha = sum(a.alpha for a in Actions if a.alpha > 0)
    sum_beta  = sum(a.beta  for a in Actions if a.beta  > 0)

    for ag in env.possible_agents:
        poss_set = set(env.current_possible_actions.get(ag, []))
        # print(f"\n[DEBUG] Agent: {ag}")
        # print(f"  Current possible actions: {[a.name for a in poss_set]}")  
        # for a in Actions:
        #     print(f"    Action: {a.name}, alpha: {a.alpha}, beta: {a.beta}, in poss_set: {a in poss_set}")

        bh_num = sum(a.alpha for a in Actions if a.alpha > 0 and a in poss_set)
        af_num = sum(a.beta  for a in Actions if a.beta  > 0 and a in poss_set)

        env.capabilities[ag]["Bodily Health"] = bh_num / sum_alpha if sum_alpha else 0
        env.capabilities[ag]["Affiliation"]   = af_num / sum_beta  if sum_beta  else 0

        idx = env.agent_name_mapping[ag]
        agent_obj = env.peh_agents[idx]
        agent_obj.central_capabilities = {
            "Bodily Health": env.capabilities[ag]["Bodily Health"],
            "Affiliation": env.capabilities[ag]["Affiliation"] 
        }



## Agent Population 

num_peh_agents = 10  