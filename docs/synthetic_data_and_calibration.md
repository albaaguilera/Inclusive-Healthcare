# Synthetic Data and Calibration

The project includes a synthetic data generator in `learning/synthetic_data.py`. Its role is to create Raval-like engagement demonstrations and fit a Bayesian logistic model used as an IRL-style potential during shaped Q-learning.

## Why Synthetic Data Exists

The simulation needs a plausible behavioral signal for social-service engagement. Instead of hard-coding one behavior for all PEH agents, the project generates heterogeneous synthetic demonstrations with:

- Trust groups.
- Administrative status.
- Health.
- Age.
- Income.
- Homelessness duration.
- History of abuse.
- Previous encounters and non-engagement.

This produces data that can calibrate an engagement probability model.

## Trust Profiles

The generator uses two trust types:

- `LOW_TRUST`
- `MODERATE_TRUST`

The profiles differ in demographic and vulnerability distributions. For example, low-trust profiles are generated with longer homelessness duration and higher probability of abuse history, while moderate-trust profiles are generated with different age, income, and duration distributions.

The generated cohorts used by the experiments are saved as JSON files such as:

- `output/peh_sample4.json`
- `output/peh_sample8.json`
- `output/peh_sample16.json`
- `output/peh_sample25.json`

For unsupported population sizes, `learning.qpbrs_seeds.ensure_generated_profiles(...)` can derive a deterministic stratified cohort from the N=16 template.

## Demonstration Simulation

Synthetic demonstrations are generated over a configurable number of people, social workers, grid size, and days.

At each simulated day:

1. Social workers move near their social-service home area.
2. PEH agents move stochastically.
3. Health drifts with noise.
4. Encounters are detected by distance.
5. Engagement is sampled from a latent logistic model.
6. Previous encounters and non-engagement counters are updated.

The resulting dataset contains one row per encounter.

## Calibration Model

The calibration uses Bayesian logistic regression with a Gaussian prior and Laplace approximation around the MAP estimate.

The target variable is:

```text
engage
```

The feature set used in the pipeline includes:

- `prev_encounters`
- `health_state`
- `homelessness_duration`
- `history_of_abuse`
- `trust_building`
- `age`
- `income`

The fitted artifact stores:

- MAP weights.
- Approximate covariance.
- Feature names.
- Standardization means and standard deviations.
- Design-matrix column ordering.

## Use as an IRL Potential

The calibrated model is loaded during PBRS/Q-learning and evaluated against an agent's current environment state. The resulting scalar acts as `Phi(s)` in potential-based reward shaping:

```text
beta * (gamma * Phi(s') - Phi(s))
```

This gives the learner a shaped signal based on engagement-related factors without replacing the environment's base reward or policy comparison metrics.

## Reproducing Calibration

Run:

```bash
python -m learning.synthetic_data
```

This script prints summaries, fits the calibration, and writes outputs under `output/`, including the calibration JSON used by later experiments.

