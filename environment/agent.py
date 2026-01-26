# agents.py
import numpy as np
from enum import Enum
from .context import Context, Actions

ACTION_NAMES = {
    Actions.RECEIVE_MEDICAL_ATTENTION: "Request and receive medical attention",
    Actions.KEEP_FORWARD: "Do not request and do not receive medical attention",
    Actions.ENGAGE_SOCIAL_SERVICES: "Engage with social services",
    Actions.REMAIN_DISENGAGED: "Remain disengaged from social services",
    Actions.APPLY_AND_GET_SHELTER: "Apply for shelter and get it",
}

class PEHAgent:
    def __init__(
        self,
        start_loc,
        health=3.0,
        admin_state="non-registered",
        trust_type="MODERATE_TRUST",
        personal_attributes=None,
        income=None,
        housing_state=None,
    ):
        self.location = start_loc.copy()
        self.health_state = float(health)

        # IMPORTANT: no randomitzar admin_state si el passes
        self.administrative_state = str(admin_state)
        self.trust_type = str(trust_type)

        self.housing_state = housing_state if housing_state is not None else np.random.choice(["ETHOS0","ETHOS1","ETHOS2"])
        self.engagement_counter = 0

        base_attrs = {
            "nationality": np.random.choice(["spanish", "non-spanish"]),
            "age": int(np.random.randint(15, 75)),
            "gender": np.random.choice(["male", "female", "non-binary"]),
            "homelessness_duration": int(np.random.randint(1, 10)),
            "history_of_abuse": bool(np.random.choice([True, False])),
        }
        if personal_attributes:
            base_attrs.update(personal_attributes)
        self.personal_attributes = base_attrs

        if income is None:
            income = float(np.random.uniform(80.0, 1200.0))
        self.income = float(income)

        self.min_health = 1.0
        self.max_health = 4.0
        self.health_step = 0.5
        self.health_update = 0.5

        self.encounter_counter = 0        # prev_encounters
        self.non_engagement_counter = 0   # trust_building (tal com ho has definit al sintètic)
        # self.engagement_counter ja la tens

        # Track actions chosen by the agent.
        self.possible_actions_history = [] #central capabilities
        self.impossible_actions_history = [] #central deprived capabilities

        # Track SPECIFIC capabilities and functionings (to be central)
        self.capabilities = {
            "Being able to have good health": True,
            "Being able to have adequate shelter": False,
            "Being able to move freely from place to place": True
        }
        self.functionings = {
            "Having good health": True,
            "Having adequate shelter": False,
            "Moving freely from place to place": True
        }
        self.central_capabilities = {
            "Bodily Health": 0.0,  # Will be updated based on specific capabilities
            "Affiliation": 0.0      # Will be updated based on specific capabilities
        }

    # def update_after_step(self, new_health, chosen_action, reward):
    #     self.health = new_health
    #     # track possible/impossible
    #     if chosen_action in self.impossible_actions:
    #         self.impossible_actions.append(chosen_action)
    #     else:
    #         self.possible_actions.append(chosen_action)
    #     # handle engagement counter bumping
    #     if chosen_action == Actions.ENGAGE_SOCIAL_SERVICES:
    #         self.engagement_counter = min(self.engagement_counter+1, 3)
    #         if self.engagement_counter >= 2 and self.admin == "non-registered":
    #             self.admin = "registered"
    #             self.engagement_counter = 0

    # def can_engage(self, socserv_locations):
    #     # return True if any social worker is adjacent
    #     ax, ay = self.location
    #     for loc in socserv_locations:
    #         sx, sy = loc
    #         if abs(ax-sx) <= 1 and abs(ay-sy) <= 1:
    #             return True
    #     return False

class SocServAgent:
    def __init__(self,  location: tuple = None):
        self.location = (
            np.array(location, dtype=int)
            if location is not None
            else np.random.randint(0, Context().grid_size, size=2)
        )
        self.num_socserv = 3

    def wander(self, grid_size, forbidden_locs, rng):
        # try up to 10 random steps without colliding
        for _ in range(10):
            step = rng.choice([-1,0,1], size=2)
            cand = np.clip(self.location + step, 0, grid_size-1)
            if not any((cand == f).all() for f in forbidden_locs):
                self.location = cand
                break
    def move_towards(self, target, grid_size, forbidden_locs, rng):
        """
        Mou el servei social un pas cap a l'objectiu (el PEHAgent),
        evitant col·lisions amb altres agents socials.
        """
        direction = np.array(target) - self.location
        step = np.sign(direction)  # -1, 0, o 1 per cada direcció
        candidates = [self.location + np.array([dx, dy])
                      for dx in [0, step[0]]
                      for dy in [0, step[1]]
                      if dx != 0 or dy != 0]

        rng.shuffle(candidates)  # barreja per evitar moviment rígid
        for cand in candidates:
            cand = np.clip(cand, 0, grid_size - 1)
            if not any(np.array_equal(cand, f) for f in forbidden_locs):
                self.location = cand
                break
    def move_towards_peh(self, peh_target_location, grid_size, forbidden_locs, rng):
        """
        Move one step toward the given PEHAgent location.
        Avoid forbidden locations (other agents).
        """
        direction = np.array(peh_target_location) - self.location
        step = np.sign(direction)

        # Possible candidate moves: diagonal, vertical, horizontal
        candidates = []
        if step[0] != 0:
            candidates.append(self.location + [step[0], 0])
        if step[1] != 0:
            candidates.append(self.location + [0, step[1]])
        if step[0] != 0 and step[1] != 0:
            candidates.append(self.location + step)

        rng.shuffle(candidates)

        for cand in candidates:
            cand = np.clip(cand, 0, grid_size - 1)
            if not any(np.array_equal(cand, f) for f in forbidden_locs):
                self.location = cand
                break
