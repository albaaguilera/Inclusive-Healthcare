# Simulation Overview

The simulation is a grid-world model of inclusive healthcare policy for people experiencing homelessness. It combines an agent-based environment with reinforcement learning so that PEH agents can learn strategies under different legal and social conditions.

## Environment

The main environment is `GridMAInequityEnv` in `environment/model.py`. It subclasses PettingZoo's `AECEnv`, so agents act sequentially. The active PEH agent is selected by `env.agent_selection`, takes an action, receives a reward, and the selector advances.

The environment contains:

- A square grid of configurable size.
- Service areas for primary healthcare (`PHC`), intensive care (`ICU`), and social services.
- PEH agents, named `peh_0`, `peh_1`, and so on.
- Social service agents that move toward assigned PEH agents.
- Governmental resources and budgets.
- Policy flags stored in the `Context`.

## Policy Scenarios

The central policy toggle is `policy_inclusive_healthcare`, set through `Context.set_scenario(...)`.

- `policy OFF`: a PEH agent can receive medical attention only if they are registered.
- `policy ON`: all PEH agents can receive medical attention, including non-registered agents.

This scenario flag changes transition probabilities for the `RECEIVE_MEDICAL_ATTENTION` action.

## PEH State

The learning state is built in `learning/utils.py` by `get_state(...)`. It includes:

- `x, y`: grid position.
- `h_idx`: discretized health state.
- `admin`: administrative status encoded from registration.
- `adj`: whether a social service agent is adjacent.
- `enc`: capped encounter counter.
- `noneng`: capped non-engagement counter.

The Q-table shape is therefore:

```text
grid_x x grid_y x health x admin x adjacency x encounters x non_engagement x actions
```

## Actions

Actions are defined in `environment/context.py` as an enum:

| Action | Meaning |
| --- | --- |
| `RECEIVE_MEDICAL_ATTENTION` | Request and receive medical attention when policy and registration allow it. |
| `KEEP_FORWARD` | Continue without requesting medical attention. |
| `ENGAGE_SOCIAL_SERVICES` | Engage with social services when a social worker is adjacent. |
| `REMAIN_DISENGAGED` | Decline or avoid engagement. |
| `APPLY_AND_GET_SHELTER` | Apply for shelter after sufficient social-service engagement and if shelter remains available. |

Action feasibility is recomputed at every PEH step by `GridMAInequityEnv._classify_actions(...)`.

## Transition Logic

Health transitions are built in `Context.build_transition_table(...)`.

Medical attention:

- If treatment is available, health increases by `health_update`, up to the maximum health.
- If treatment is not available, health decreases.
- Failed treatment access can produce a negative reward, especially when health reaches the minimum.

Other actions:

- Health generally drifts downward by one health step.
- Social-service engagement succeeds only when a social worker is adjacent.
- Shelter access depends on adjacency, prior engagement, and available shelters.

An agent terminates when health reaches its minimum or maximum, or the run truncates when the maximum step count is reached.

## Rewards

The base reward is produced by the environment transition table:

- Positive reward for successful access to care or successful engagement.
- Small negative reward for failed engagement or unavailable care.
- Larger negative reward when health falls to the minimum.

During training under `policy ON`, the learning code can add potential-based reward shaping from the calibrated IRL potential. This changes the training signal but preserves the structure of the environment reward used for evaluation summaries.

## Budgets and Costs

The `Context` tracks:

- `healthcare_budget`
- `social_service_budget`
- `shelters_available`

Costs are charged for:

- Medical visits.
- Hospitalization when an agent terminates at minimum health.
- Social service worker activity.

Evaluation summaries report healthcare and social-service spending by comparing initial and final budgets.

## Capabilities and Functionings

The project uses Capability Approach language to distinguish opportunities from achieved outcomes.

Central capability scores are recomputed from the current feasible action set:

- `Bodily Health`: associated with the possibility of receiving medical attention.
- `Affiliation`: associated with the possibility of engaging social services.

Functionings represent achieved states, such as having good health or adequate shelter. Evaluation artifacts track initial and final health, administrative state, and capability scores for each PEH agent.

## Episode Flow

At a high level:

1. Reset the environment with a seed and a PEH profile cohort.
2. Place PEH agents on free grid cells.
3. Spawn social service agents, usually near assigned PEH agents.
4. For each PEH turn, classify feasible actions.
5. Move social service agents toward their assigned PEH targets.
6. Apply the PEH action and sample the transition.
7. Update health, registration, engagement counters, histories, budgets, rewards, and capabilities.
8. Continue until all agents terminate or the step limit is reached.

