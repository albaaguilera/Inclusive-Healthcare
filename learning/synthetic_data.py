# complete_pipeline.py
# ------------------------------------------------------------
# 1) Synthetic "Raval-like" engagement demonstrations
# 2) Bayesian IRL calibration (Bayesian logistic regression + Laplace)
# 3) Q-learning with Q-init potential, beta selection by target-matching loss
# 4) Plots: positions + distance vs engagement/health + final dashboard by trust type
#
# Includes: improved Social Worker outreach behavior
#   - SW spawn at random free cells (not in service rectangles)
#   - SW target PEH (nearest not-yet-engaged by default)
#   - SW move 1 Chebyshev step toward target after each env.step()
#   - If target engages OR SW waits near target without engagement, SW retargets
# ------------------------------------------------------------

import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

from gymnasium import spaces

# --- Your environment imports (as in your current project) ---
from environment.model import GridMAInequityEnv
from environment.context import Context, Actions, update_all_capability_scores
from environment.agent import PEHAgent, SocServAgent


# ============================================================
# Utils
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def chebyshev_dist(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(a - b)))

def nearest_social_distance(peh_loc: np.ndarray, soc_locs: List[np.ndarray]) -> int:
    if not soc_locs:
        return 999
    return min(chebyshev_dist(peh_loc, s) for s in soc_locs)

def clip_int(x, lo, hi) -> int:
    return int(max(lo, min(hi, x)))

def snap_to_half(x: float, lo: float = 1.0, hi: float = 4.0) -> float:
    """Snap x to {1.0, 1.5, ..., 4.0} and clip."""
    x = float(x)
    x = round(x / 0.5) * 0.5
    return float(np.clip(x, lo, hi))

def random_step(rng: np.random.Generator) -> np.ndarray:
    dx = rng.integers(-1, 2)
    dy = rng.integers(-1, 2)
    return np.array([dx, dy], dtype=int)


# ============================================================
# Part 1: Synthetic "Raval-like" data generator (with trust types)
# ============================================================

TRUST_TYPES = ["LOW_TRUST", "MODERATE_TRUST"]

TRUST_CLUSTER_SPEC = {
    "LOW_TRUST": {
        "gender": "male with p=0.85 else female",
        "age": "Normal(58, 8) clipped [18,75]",
        "history_of_abuse": "Bernoulli(p=0.65)",
        "income": "Normal(520,160) clipped [80,1200]",
        "homelessness_duration": "UniformInt[7,15]",
        "health_init": "Uniform[3.5,4.0] snapped to 0.5"
    },
    "MODERATE_TRUST": {
        "gender": "Categorical([male,female,non-binary]=[0.60,0.35,0.05])",
        "age": "Normal(38, 9) clipped [18,75]",
        "history_of_abuse": "Bernoulli(p=0.10)",
        "income": "Normal(360,140) clipped [80,1200]",
        "homelessness_duration": "UniformInt[1,6]",
        "health_init": "Uniform[3.0,3.5] snapped to 0.5"
    },
}

def print_cluster_spec_once():
    print("\n" + "="*70)
    print("TRUST CLUSTER SPECIFICATION (used for fixed profiles)")
    print("="*70)
    for k, v in TRUST_CLUSTER_SPEC.items():
        print(f"\n{k}:")
        for kk, vv in v.items():
            print(f"  - {kk}: {vv}")
    print("="*70 + "\n")

@dataclass
class SyntheticParams:
    grid_size: int = 8
    n_people: int = 300
    n_social_workers: int = 11
    horizon_days: int = 50
    seed: int = 42

    peh_move_prob: float = 0.50
    soc_move_prob: float = 0.10
    soc_home_radius: int = 1

    encounter_dist_threshold: int = 1
    p_encounter_if_adj: float = 0.35

    health_min: float = 1.0
    health_max: float = 4.0
    health_step: float = 0.5

    low_trust_health_range: Tuple[float, float] = (2.5, 3.5)
    mod_trust_health_range: Tuple[float, float] = (1.5, 2.5)

    daily_health_drift: float = -0.02
    daily_health_noise: float = 0.05
    engage_health_boost: float = 0.05

    p_low_trust: float = 0.50
    p_registered: float = 0.50
    p_registered_low: float = 0.50
    p_registered_mod: float = 0.50
    exact_proportions: bool = True
    registered_depends_on_trust: bool = True 
    #if true, we use p_registered_low/mod; if false, we use p_regegistered 

    low_male: float = 0.85
    low_age_mu: float = 58.0
    low_age_sigma: float = 8.0
    low_p_abuse: float = 0.65
    low_income_mu: float = 520.0
    low_income_sigma: float = 160.0
    low_dur_min: int = 7
    low_dur_max: int = 15

    mod_male: float = 0.60
    mod_age_mu: float = 38.0
    mod_age_sigma: float = 9.0
    mod_p_abuse: float = 0.10
    mod_income_mu: float = 360.0
    mod_income_sigma: float = 140.0
    mod_dur_min: int = 1
    mod_dur_max: int = 6

    income_min: float = 80.0
    income_max: float = 1200.0

def sample_fixed_profile(rng: np.random.Generator, p_low_trust: float = 0.45, forced_trust_type: Optional[str] = None, forced_registered: Optional[int] = None) -> Dict[str, Any]:
    trust_type = "LOW_TRUST" if (rng.random() < p_low_trust) else "MODERATE_TRUST"

    if trust_type == "LOW_TRUST":
        gender = "male" if (rng.random() < 0.85) else "female"
        age = int(np.clip(rng.normal(58.0, 8.0), 18, 75))
        abuse = bool(rng.random() < 0.65)
        income = float(np.clip(rng.normal(520.0, 160.0), 80.0, 1200.0))
        dur = int(rng.integers(7, 16))
        h0 = float(rng.uniform(3.75, 4.0))  # menys urgència

        p_reg = sp.p_registered_low if sp.registered_depends_on_trust else sp.p_registered
    else:
        gender = rng.choice(["male", "female", "non-binary"], p=[0.60, 0.35, 0.05])
        age = int(np.clip(rng.normal(38.0, 9.0), 18, 75))
        abuse = bool(rng.random() < 0.10)
        income = float(np.clip(rng.normal(360.0, 140.0), 80.0, 1200.0))
        dur = int(rng.integers(1, 7))
        h0 = float(rng.uniform(3.5, 4.0))  # més urgència

        p_reg = sp.p_registered_mod if sp.registered_depends_on_trust else sp.p_registered

    registered = forced_registered if forced_registered is not None else int(rng.random() < p_reg)
    admin_state = "REGISTERED" if registered == 1 else "NON_REGISTERED"        

    return dict(
        trust_type=trust_type,
        admin_state=admin_state,
        registered=float(registered),
        gender=str(gender),
        age=int(age),
        history_of_abuse=bool(abuse),
        income=float(income),
        homelessness_duration=int(dur),
        health_state=float(snap_to_half(h0, 1.0, 4.0))
    )

@dataclass
class ThetaTrue:
    theta0: float = 0.01
    theta_prev: float = 0.30
    theta_health_urg: float = 0.10
    theta_dur: float = -0.30
    theta_abuse: float = -0.50
    theta_trust: float = 0.10
    theta_age: float = -0.10
    #theta_male: float = -0.20
    theta_income_urg: float = 0.05
    #theta_dist: float = -0.25


def sample_profile(rng: np.random.Generator, sp: SyntheticParams, forced_trust_type: Optional[str] = None, forced_registered: Optional[int] = None) -> Dict[str, Any]:

    trust_type = forced_trust_type or ("LOW_TRUST" if (rng.random() < sp.p_low_trust) else "MODERATE_TRUST")
    
    if trust_type == "LOW_TRUST":
        gender = "male" if (rng.random() < sp.low_male) else "female"
        age = int(np.clip(rng.normal(sp.low_age_mu, sp.low_age_sigma), 18, 75))
        abuse = int(rng.random() < sp.low_p_abuse)
        income = float(np.clip(rng.normal(sp.low_income_mu, sp.low_income_sigma), sp.income_min, sp.income_max))
        dur = int(rng.integers(sp.low_dur_min, sp.low_dur_max + 1))
        h0 = rng.uniform(*sp.low_trust_health_range)
        p_reg = sp.p_registered_low if sp.registered_depends_on_trust else sp.p_registered
    else:
        gender = rng.choice(["male", "female", "non-binary"], p=[sp.mod_male, 1.0 - sp.mod_male - 0.05, 0.05])
        age = int(np.clip(rng.normal(sp.mod_age_mu, sp.mod_age_sigma), 18, 75))
        abuse = int(rng.random() < sp.mod_p_abuse)
        income = float(np.clip(rng.normal(sp.mod_income_mu, sp.mod_income_sigma), sp.income_min, sp.income_max))
        dur = int(rng.integers(sp.mod_dur_min, sp.mod_dur_max + 1))
        h0 = rng.uniform(*sp.mod_trust_health_range)
        p_reg = sp.p_registered_mod if sp.registered_depends_on_trust else sp.p_registered

    health0 = snap_to_half(h0, sp.health_min, sp.health_max)
    registered = forced_registered if forced_registered is not None else int(rng.random() < p_reg)
    admin_state = "registered" if registered == 1 else "non-registered"

    return dict(
        trust_type=trust_type,
        age=float(age),
        gender=str(gender),
        history_of_abuse=float(abuse),
        income=float(income),
        homelessness_duration=float(dur),
        health_state=float(health0),
        admin_state=admin_state,
        registered=float(registered),
    )


def generate_synthetic_raval_demonstrations(sp: SyntheticParams, th: ThetaTrue) -> pd.DataFrame:
    rng = np.random.default_rng(sp.seed)

    soc_home = np.array([0, sp.grid_size - 2], dtype=int)

    soc_locs = []
    for _ in range(sp.n_social_workers):
        jitter = rng.integers(-1, 2, size=2)
        loc = np.clip(soc_home + jitter, 0, sp.grid_size - 1)
        soc_locs.append(loc.astype(int))

    def _exact_flags(n: int, p: float) -> np.ndarray:
        k = int(round(n * p))
        arr = np.array([1] * k + [0] * (n - k), dtype=int)
        rng.shuffle(arr)
        return arr

    # 1) Trust type: exactament segons p_low_trust
    if sp.exact_proportions:
        low_flags = _exact_flags(sp.n_people, sp.p_low_trust)  # 1=LOW, 0=MOD
    else:
        low_flags = (rng.random(sp.n_people) < sp.p_low_trust).astype(int)

    forced_trust = np.array(["LOW_TRUST" if f == 1 else "MODERATE_TRUST" for f in low_flags], dtype=object)

    # 2) Registered: pot dependre o no del trust type
    forced_registered = np.zeros(sp.n_people, dtype=int)

    if sp.exact_proportions:
        if sp.registered_depends_on_trust:
            idx_low = np.where(forced_trust == "LOW_TRUST")[0]
            idx_mod = np.where(forced_trust == "MODERATE_TRUST")[0]

            forced_registered[idx_low] = _exact_flags(len(idx_low), sp.p_registered_low)
            forced_registered[idx_mod] = _exact_flags(len(idx_mod), sp.p_registered_mod)
        else:
            forced_registered = _exact_flags(sp.n_people, sp.p_registered)
    else:
        if sp.registered_depends_on_trust:
            for i in range(sp.n_people):
                p_reg = sp.p_registered_low if forced_trust[i] == "LOW_TRUST" else sp.p_registered_mod
                forced_registered[i] = int(rng.random() < p_reg)
        else:
            forced_registered = (rng.random(sp.n_people) < sp.p_registered).astype(int)

    # 3) Crea perfils amb aquests valors forçats
    profiles = [
        sample_profile(rng, sp, forced_trust_type=forced_trust[i], forced_registered=int(forced_registered[i]))
        for i in range(sp.n_people)
    ]


    def sample_loc_near(home: np.ndarray, size: int, rng: np.random.Generator, radius: int) -> np.ndarray:
        x0, y0 = int(home[0]), int(home[1])
        xs = np.arange(max(0, x0 - radius), min(size, x0 + radius + 1))
        ys = np.arange(max(0, y0 - radius), min(size, y0 + radius + 1))
        candidates = np.array([(x, y) for x in xs for y in ys], dtype=int)
        return candidates[rng.integers(0, len(candidates))]

    def sample_loc_far(home: np.ndarray, size: int, rng: np.random.Generator, min_dist: int) -> np.ndarray:
        for _ in range(2000):
            loc = rng.integers(0, size, size=2).astype(int)
            if chebyshev_dist(loc, home) >= min_dist:
                return loc
        return rng.integers(0, size, size=2).astype(int)

    peh_locs = []
    for i in range(sp.n_people):
        tt = profiles[i]["trust_type"]
        if tt == "MODERATE_TRUST":
            if rng.random() < 0.75:
                loc = sample_loc_near(soc_home, sp.grid_size, rng, radius=2)
            else:
                loc = rng.integers(0, sp.grid_size, size=2).astype(int)
        else:
            if rng.random() < 0.75:
                loc = sample_loc_far(soc_home, sp.grid_size, rng, min_dist=4)
            else:
                loc = rng.integers(0, sp.grid_size, size=2).astype(int)
        peh_locs.append(loc.astype(int))

    prev_encounters = np.zeros(sp.n_people, dtype=int)
    prev_non_eng = np.zeros(sp.n_people, dtype=int)

    rows = []

    for day in range(sp.horizon_days):
        # SW movement
        for j in range(sp.n_social_workers):
            if rng.random() < sp.soc_move_prob:
                cand = np.clip(soc_locs[j] + random_step(rng), 0, sp.grid_size - 1)
                if chebyshev_dist(cand, soc_home) <= sp.soc_home_radius:
                    soc_locs[j] = cand.astype(int)

        # PEH movement + health drift
        for i in range(sp.n_people):
            if rng.random() < sp.peh_move_prob:
                peh_locs[i] = np.clip(peh_locs[i] + random_step(rng), 0, sp.grid_size - 1).astype(int)

            h = float(profiles[i]["health_state"])
            h = h + sp.daily_health_drift + rng.normal(0.0, sp.daily_health_noise)
            profiles[i]["health_state"] = snap_to_half(h, sp.health_min, sp.health_max)

        # encounters + engage sampling
        for i in range(sp.n_people):
            dist = nearest_social_distance(peh_locs[i], soc_locs)

            if dist > sp.encounter_dist_threshold:
                continue
            if dist <= 1 and (rng.random() > sp.p_encounter_if_adj):
                continue

            prof = profiles[i]
            prev_enc = float(prev_encounters[i])
            trust = float(prev_non_eng[i])

            age = float(prof["age"])
            gender = str(prof["gender"])
            abuse = float(prof["history_of_abuse"])
            dur = float(prof["homelessness_duration"])
            income = float(prof["income"])
            health = float(prof["health_state"])

            prev_s = math.log1p(prev_enc)
            health_urg = (2.5 - health) / 0.5
            dur_z = (dur - 5.0) / 3.0
            age_z = (age - 45.0) / 10.0
            income_urg = (450.0 - income) / 200.0
            male = 1.0 if gender == "male" else 0.0

            u = (
                th.theta0
                + th.theta_prev * prev_s
                + th.theta_health_urg * health_urg
                + th.theta_dur * dur_z
                + th.theta_abuse * abuse
                + th.theta_trust * trust
                + th.theta_age * age_z
                #+ th.theta_male * male
                + th.theta_income_urg * income_urg
                #+ th.theta_dist * float(dist)
            )

            p = float(sigmoid(np.array(u)))
            engage = int(rng.random() < p)

            rows.append({
                "person_id": i,
                "trust_type": prof["trust_type"],
                "day": day,
                "encounter_number": int(prev_enc),

                "prev_encounters": prev_enc,
                "health_state": health,
                "admin_state": prof["admin_state"],
                "registered": float(prof["registered"]),
                "homelessness_duration": dur,
                "history_of_abuse": abuse,
                "trust_building": trust,
                "age": age,
                "gender": gender,
                "income": income,
                #"distance_to_sw": float(dist),

                "engagement_probability": p,
                "engage": engage,
            })

            prev_encounters[i] += 1
            if engage == 0:
                prev_non_eng[i] += 1
            else:
                profiles[i]["health_state"] = snap_to_half(
                    profiles[i]["health_state"] + sp.engage_health_boost,
                    sp.health_min, sp.health_max
                )

    df = pd.DataFrame(rows)

    print("\n" + "="*70)
    print("SYNTHETIC RAVAL-LIKE DATA SUMMARY (with trust types)")
    print("="*70)
    print(f"People: {sp.n_people} | Social workers: {sp.n_social_workers} | Grid: {sp.grid_size}x{sp.grid_size} | Days: {sp.horizon_days}")
    print(f"Total encounters (rows): {len(df)}")

    enc_counts = df.groupby("person_id").size().reindex(range(sp.n_people)).fillna(0).astype(int)
    print(f"People with 0 encounters: {(enc_counts==0).sum()} / {sp.n_people}")
    print(f"Avg encounters/person: {enc_counts.mean():.2f}")

    if len(df) > 0:
        print(f"Overall engagement rate (per encounter): {df['engage'].mean():.2%}")
        print("\nEngagement rate by trust_type (per encounter):")
        print(df.groupby("trust_type")["engage"].mean())
        print("\nEngagement rate by abuse (per encounter):")
        print(df.groupby("history_of_abuse")["engage"].mean())

    print("="*70 + "\n")
    return df


# ============================================================
# Part 2: Bayesian IRL calibration (Bayesian logistic regression + Laplace)
# ============================================================

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


def fit_bayesian_logit_laplace(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str = "engage",
    prior_mu: Optional[np.ndarray] = None,
    prior_sigma: float = 2.0,
    add_intercept: bool = True,
) -> Dict[str, Any]:
    if len(df) == 0:
        raise ValueError("Cannot calibrate: df has 0 rows.")

    Xdf, x_columns = _build_design_df(df, feature_cols, fitted_columns=None)
    X = Xdf.to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    Xs = (X - X_mean) / X_std

    if add_intercept:
        Xs = np.column_stack([np.ones(len(Xs)), Xs])
        names = ["intercept"] + x_columns
    else:
        names = x_columns[:]

    d = Xs.shape[1]
    if prior_mu is None:
        prior_mu = np.zeros(d)
    else:
        prior_mu = np.asarray(prior_mu).reshape(-1)

    prior_var = (prior_sigma ** 2)
    prior_prec = np.eye(d) / prior_var

    w = prior_mu.copy()

    def nll_and_grad_hess(wv):
        z = Xs @ wv
        p = sigmoid(z)

        eps = 1e-9
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        diff = (wv - prior_mu)
        nlp = 0.5 * diff.T @ prior_prec @ diff
        obj = nll + nlp

        grad = Xs.T @ (p - y) + prior_prec @ diff
        W = p * (1 - p)
        H = Xs.T @ (Xs * W[:, None]) + prior_prec
        return obj, grad, H

    for _ in range(30):
        _, grad, H = nll_and_grad_hess(w)
        step = np.linalg.solve(H, grad)
        w_new = w - step
        if np.linalg.norm(step) < 1e-6:
            w = w_new
            break
        w = w_new

    _, _, H_map = nll_and_grad_hess(w)
    cov = np.linalg.inv(H_map)

    print("\n" + "="*70)
    print("BAYESIAN LOGIT CALIBRATION (MAP + LAPLACE)")
    print("="*70)
    for i, nm in enumerate(names):
        sd = math.sqrt(max(cov[i, i], 0.0))
        print(f"{nm:>22s}: w_map = {w[i]:>8.4f}   (approx sd = {sd:>7.4f})")
    print("="*70 + "\n")

    return dict(
        w_map=w,
        cov=cov,
        feature_names=names,
        x_columns=x_columns,
        X_mean=X_mean,
        X_std=X_std,
        add_intercept=add_intercept,
    )


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

def plot_prior_vs_posterior(calib, prior_mu=None, prior_sigma=2.0, title="Prior vs posterior (Laplace)"):
    names = calib["feature_names"]
    w_map = calib["w_map"]
    post_sd = np.sqrt(np.clip(np.diag(calib["cov"]), 0.0, None))

    d = len(w_map)
    if prior_mu is None:
        prior_mu = np.zeros(d)
    prior_mu = np.asarray(prior_mu).reshape(-1)

    prior_sd = np.ones(d) * prior_sigma

    x = np.arange(d)

    plt.figure(figsize=(10, max(4, 0.25 * d)))
    plt.errorbar(x, prior_mu, yerr=prior_sd, fmt="o", capsize=3, label="Prior (μ ± σ)")
    plt.errorbar(x, w_map, yerr=post_sd, fmt="o", capsize=3, label="Posterior (MAP ± sd)")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.show()


# print synthetic dataset
sp = SyntheticParams(n_people=300, p_low_trust=0.5, exact_proportions=True, registered_depends_on_trust=True, p_registered_low=0.5, p_registered_mod=0.5)
th = ThetaTrue()

df = generate_synthetic_raval_demonstrations(sp, th)
print(df.head())

# print priors and posteriors
feature_cols = [
    "prev_encounters",
    "health_state",
    "homelessness_duration",
    "history_of_abuse",
    #"trust_building",
    "age",
    "income",
]

calib = fit_bayesian_logit_laplace(
    df,
    feature_cols=feature_cols,
    y_col="engage",
    prior_mu=None,       # zero-mean prior
    prior_sigma=2.0,
    add_intercept=True,
)

# B) Bayesian IRL calibration
feature_cols = [
    "prev_encounters",
    "health_state",
    "homelessness_duration",
    "history_of_abuse",
    "trust_building",
    "age",
    # "gender",
    "income",
    #"distance_to_sw",
]

calib = fit_bayesian_logit_laplace(
    df=df,
    feature_cols=feature_cols,
    y_col="engage",
    prior_sigma=2.0,
    add_intercept=True,
)
theta_map = calib["w_map"]
cov = calib["cov"]

out = {
    "feature_names": calib["feature_names"],
    "x_columns": calib["x_columns"],
    "w_map": theta_map.tolist(),
    "cov_diag": np.diag(cov).tolist(),
    "X_mean": calib["X_mean"].tolist(),
    "X_std": calib["X_std"].tolist(),
}
with open("output/irl_calibration_results_raval.json", "w") as f:
    json.dump(out, f, indent=2)
print("✓ Saved: output/irl_calibration_results_raval.json")
plot_prior_vs_posterior(calib, prior_mu=None, prior_sigma=2.0)

# One row per agent (person)
agents = df.drop_duplicates(subset=["person_id"])[
    ["person_id", "trust_type", "admin_state"]
].copy()

print("\n=== Agent counts ===")
print("Total agents:", agents["person_id"].nunique())

print("\nBy trust_type:")
print(agents["trust_type"].value_counts())

print("\nBy admin_state:")
print(agents["admin_state"].value_counts())

print("\nBy trust_type x admin_state:")
print(pd.crosstab(agents["trust_type"], agents["admin_state"]))

print("\nBy trust_type (%):")
print((agents["trust_type"].value_counts(normalize=True) * 100).round(1))

print("\nBy admin_state (%):")
print((agents["admin_state"].value_counts(normalize=True) * 100).round(1))

print("\nBy trust_type x admin_state (row %):")
print((pd.crosstab(agents["trust_type"], agents["admin_state"], normalize="index") * 100).round(1))


# TAKE A SMALL SAMPLE FOR SIMULATION WITH non-reg/reg and low/mod trust
# --- Step 1: sample 4 agents (one per trust x admin category) ---

# one row per person (population snapshot)
pop = df.sort_values(["person_id", "day", "encounter_number"]).drop_duplicates("person_id").copy()

# normalize admin labels just in case you mixed cases ("registered" vs "REGISTERED")
pop["admin_state"] = pop["admin_state"].str.upper().str.replace("-", "_")

# categories we want
targets = [
    ("LOW_TRUST",      "NON_REGISTERED"),
    ("MODERATE_TRUST", "NON_REGISTERED"),
    ("LOW_TRUST",      "REGISTERED"),
    ("MODERATE_TRUST", "REGISTERED"),
]

picked = []
for trust, admin in targets:
    cand = pop[(pop["trust_type"] == trust) & (pop["admin_state"] == admin)]
    if len(cand) == 0:
        raise ValueError(f"No agents found for category ({trust}, {admin}). "
                         f"Try adjusting p_low_trust / p_registered_* or exact_proportions.")
    row = cand.sample(n=1, random_state=sp.seed).iloc[0]
    picked.append(row)

sample4 = pd.DataFrame(picked)[[
    "person_id", "trust_type", "admin_state",
    "age", "gender", "income", "homelessness_duration", "history_of_abuse", "health_state"
]].reset_index(drop=True)
# --- FORCE SAME INITIAL HEALTH FOR THE 4-PROFILE CASE ONLY ---
FIXED_H4 = 3.0  
sample4["health_state"] = FIXED_H4

print("\n=== Sample of 4 agents (trust x admin) ===")
print(sample4)

profiles4 = sample4.to_dict(orient="records")
print("\nProfiles4 dicts:")
for p in profiles4:
    print(p)
import json
with open("output/peh_sample4.json", "w") as f:
    json.dump(profiles4, f, indent=2)


# TAKE A SMALL SAMPLE FOR SIMULATION WITH non-reg/reg and low/mod trust
# --- Now: sample 16 agents (4 per trust x admin category) ---

pop = df.sort_values(["person_id", "day", "encounter_number"]).drop_duplicates("person_id").copy()
pop["admin_state"] = pop["admin_state"].str.upper().str.replace("-", "_")
cells = [
    ("LOW_TRUST",      "NON_REGISTERED"),
    ("MODERATE_TRUST", "NON_REGISTERED"),
    ("LOW_TRUST",      "REGISTERED"),
    ("MODERATE_TRUST", "REGISTERED"),
]

picked_rows = []
for trust, admin in cells:
    cand = pop[(pop["trust_type"] == trust) & (pop["admin_state"] == admin)]
    if len(cand) < 4:
        raise ValueError(
            f"Not enough agents in category ({trust}, {admin}) to sample 4; "
            f"found {len(cand)}. Try adjusting SyntheticParams proportions."
        )
    # sample 4 distinct agents in this cell
    block = cand.sample(n=4, random_state=sp.seed)
    picked_rows.append(block)

sample16 = (
    pd.concat(picked_rows, ignore_index=True)[[
        "person_id", "trust_type", "admin_state",
        "age", "gender", "income",
        "homelessness_duration", "history_of_abuse", "health_state"
    ]]
    .reset_index(drop=True)
)
FIXED_H4 = 3.0 
sample16["health_state"] = FIXED_H4

print("\n=== Sample of 16 agents (4 per trust x admin cell) ===")
print(sample16)
profiles16 = sample16.to_dict(orient="records")
print("\nProfiles16 dicts:")
for p in profiles16:
    print(p)

with open("output/peh_sample16.json", "w") as f:
    json.dump(profiles16, f, indent=2)


# TAKE A SMALL SAMPLE FOR SIMULATION WITH non-reg/reg and low/mod trust
# --- Now: sample 8 agents (2 per trust x admin category) ---

pop = df.sort_values(["person_id", "day", "encounter_number"]).drop_duplicates("person_id").copy()
pop["admin_state"] = pop["admin_state"].str.upper().str.replace("-", "_")
cells = [
    ("LOW_TRUST",      "NON_REGISTERED"),
    ("MODERATE_TRUST", "NON_REGISTERED"),
    ("LOW_TRUST",      "REGISTERED"),
    ("MODERATE_TRUST", "REGISTERED"),
]

picked_rows = []
for trust, admin in cells:
    cand = pop[(pop["trust_type"] == trust) & (pop["admin_state"] == admin)]
    if len(cand) < 2:
        raise ValueError(
            f"Not enough agents in category ({trust}, {admin}) to sample 2; "
            f"found {len(cand)}. Try adjusting SyntheticParams proportions."
        )
    # sample 2 distinct agents in this cell
    block = cand.sample(n=2, random_state=sp.seed)
    picked_rows.append(block)

sample8 = (
    pd.concat(picked_rows, ignore_index=True)[[
        "person_id", "trust_type", "admin_state",
        "age", "gender", "income",
        "homelessness_duration", "history_of_abuse", "health_state"
    ]]
    .reset_index(drop=True)
)
FIXED_H4 = 3.0 
sample8["health_state"] = FIXED_H4
print("\n=== Sample of 8 agents (2 per trust x admin cell) ===")
print(sample8)
profiles8 = sample8.to_dict(orient="records")
print("\nProfiles8 dicts:")
for p in profiles8:
    print(p)

with open("output/peh_sample8.json", "w") as f:
    json.dump(profiles8, f, indent=2)

