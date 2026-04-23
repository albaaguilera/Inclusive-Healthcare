# Learning Algorithm

The main reproducible learning pipeline is in `learning/qpbrs_seeds.py`. It trains and evaluates separate policies for inclusive healthcare `ON` and `OFF`, repeats the experiment across seeds, and aggregates rewards, outcomes, and strategies.

## Algorithm Summary

The project uses independent tabular Q-learning for each PEH agent. During training, each PEH agent has its own Q-table indexed by the environment state and action.

The update is the standard Q-learning rule:

```text
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

where:

- `alpha` is the learning rate.
- `gamma` is the discount factor.
- `r` is the reward, optionally shaped during training.
- `s` and `s'` are the current and next states.
- `a` is the selected action.

## Default Training Parameters

`ExperimentConfig` defines the main defaults:

| Parameter | Default |
| --- | ---: |
| Grid size | `7` |
| PEH agents | `8` |
| Social workers | `15` |
| Episodes | `400` |
| Training max steps | `100` |
| Evaluation max steps | `500` |
| Learning rate `alpha` | `0.2` |
| Discount `gamma` | `0.99` |
| Initial epsilon | `0.1` |
| Minimum epsilon | `0.01` |
| Epsilon decay | `0.995` |
| PBRS beta | `0.02` |
| Seeds | `0 1 2 3 4` |

These can be overridden through CLI arguments in the multi-seed and scalability scripts.

## State Representation

The learned state is:

```text
(x, y, health_index, admin, adjacent_to_social_worker, encounter_count, non_engagement_count)
```

Encounter and non-engagement counts are capped by `max_enc` and `max_noneng` to keep Q-tables finite.

## Action Masking

The policy uses masks derived from the environment's feasible action classifier. During exploration, random actions are sampled from feasible actions. During exploitation, `masked_argmax(...)` ignores infeasible actions.

This is important because the legal and social setting changes which actions are meaningful:

- Non-registered agents cannot receive medical attention when policy is OFF.
- Social-service actions depend on adjacency.
- Shelter depends on engagement and available shelter capacity.

## Exploration

Training uses epsilon-greedy exploration:

1. With probability `epsilon`, choose a random feasible action.
2. Otherwise, choose the feasible action with the highest Q-value.
3. After each episode, decay epsilon down to `eps_min`.

## Potential-Based Reward Shaping

The `policy ON` training path adds a potential-based reward shaping term:

```text
shaped_reward = base_reward + beta * (gamma * Phi(s') - Phi(s))
```

`Phi` is an IRL-derived potential estimated from synthetic engagement data. The shaping term encourages behavior aligned with calibrated engagement potential while keeping the reward transformation structured around state potentials.

In this codebase, shaping is applied during `policy ON` training. Evaluation summaries use the environment rewards collected while following the learned policy.

## IRL Potential

The potential comes from a Bayesian logistic calibration over synthetic Raval-like engagement demonstrations. Features include:

- Previous encounters.
- Health state.
- Homelessness duration.
- History of abuse.
- Trust-building or non-engagement.
- Age.
- Income.

The helper functions `load_irl(...)`, `irl_potential_from_env(...)`, and `cached_irl_potential(...)` connect the calibrated model to the environment.

## Training Procedure

For each seed and scenario:

1. Load or generate PEH profiles.
2. Build an environment with the policy scenario set.
3. Initialize one Q-table per PEH agent.
4. For every episode, reset the environment with a deterministic seed.
5. Run the PEH turn cycle.
6. Select actions using epsilon-greedy masked Q-values.
7. Apply environment transitions.
8. Add PBRS shaping if applicable.
9. Update the active PEH agent's Q-table.
10. Record total and group-level rewards.

## Evaluation Procedure

Evaluation is greedy:

1. Reset a fresh environment for the scenario and seed.
2. Record initial health, registration, trust type, budgets, and capabilities.
3. At each PEH turn, choose the masked argmax action.
4. Record step-level action, reward, state, capability, and budget data.
5. At the end, record per-agent final health, registration, capabilities, counters, return, and local step count.

## Multi-Seed Aggregation

After running all seeds, the script builds:

- Training summaries by seed and across seeds.
- Evaluation summaries by seed and across seeds.
- Paired `ON - OFF` policy differences.
- Dominant strategies by group and local step.
- Strategy consensus across seeds.
- Representative artifacts for final figures.

The purpose is to check whether the policy effect is robust rather than tied to a single random initialization.

