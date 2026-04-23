# Agent Behavior

The simulation has two kinds of agents: PEH agents and social service agents. PEH agents are the learning agents. Social service agents are environmental actors that influence PEH opportunities.

## PEH Agents

PEH agents are implemented by `PEHAgent` in `environment/agent.py`.

Each PEH agent stores:

- Location on the grid.
- Health state from `1.0` to `4.0` in `0.5` increments.
- Administrative state: `registered` or `non-registered`.
- Trust type: usually `LOW_TRUST` or `MODERATE_TRUST`.
- Housing state, such as `ETHOS0`, `ETHOS1`, or `ETHOS2`.
- Engagement and non-engagement counters.
- Encounter counter with social service agents.
- Personal attributes such as age, gender, homelessness duration, abuse history, income, and nationality.
- Histories of possible and impossible actions.
- Capability and functioning dictionaries.

## PEH Groups

Experiments aggregate PEH agents into four groups:

| Group | Meaning |
| --- | --- |
| `NONREG_LOW` | Initially non-registered and low trust. |
| `NONREG_MOD` | Initially non-registered and moderate trust. |
| `REG_LOW` | Initially registered and low trust. |
| `REG_MOD` | Initially registered and moderate trust. |

These groups are used in reward curves, strategy summaries, and policy comparisons.

## PEH Action Feasibility

At each turn, the environment calls `_classify_actions(peh)` to decide which actions are currently possible.

The main rules are:

- Medical attention is possible if the PEH agent is registered or inclusive healthcare policy is ON.
- Social-service engagement is possible only when a social service agent is adjacent.
- Shelter application is possible only when a social service agent is adjacent, the PEH agent has engaged more than once, and shelters are still available.
- `KEEP_FORWARD` is always possible.
- `REMAIN_DISENGAGED` is treated as a general available behavior, though its recorded outcome depends on whether a social worker is adjacent.

The learning code uses an action mask built from this classification so agents do not intentionally select infeasible actions during Q-learning evaluation.

## Administrative State Changes

Social-service engagement can change administrative state:

- Successful engagement increments the PEH agent's engagement counter.
- Once engagement reaches the threshold, a non-registered PEH agent can become registered.
- Remaining disengaged near a social service agent can increase non-engagement and may reverse registration in the current environment logic.

This creates an important mechanism: social services can restore access to systems, but the inclusive healthcare policy can bypass registration as a barrier to medical care.

## Health Dynamics

Health changes through the transition table:

- Medical attention can improve health.
- Lack of access or non-care actions usually reduce health.
- Health is clipped to the PEH agent's minimum and maximum.
- Hitting minimum health can trigger hospitalization cost and termination.
- Hitting maximum health can also terminate the agent as a completed trajectory.

## Social Service Agents

Social service agents are implemented by `SocServAgent` in `environment/agent.py`.

They store a grid location and can:

- Wander randomly.
- Move one step toward a target.
- Move toward an assigned PEH agent while avoiding PEH cells.

In the current environment step logic, each social service agent has an assigned PEH target and moves toward that PEH at every PEH step. Social service agents are not learners; they shape the opportunity landscape by making engagement and shelter actions feasible when adjacent.

## Spatial Interaction

Adjacency uses Chebyshev distance:

```text
max(abs(peh_x - sw_x), abs(peh_y - sw_y)) == 1
```

This means horizontal, vertical, and diagonal neighbors count as adjacent. Same-cell overlap is avoided in movement and does not count as adjacency.

## What Agents Learn

PEH agents learn which action to select from the feasible action set in each state. The learned strategy can vary by:

- Policy scenario.
- Health state.
- Registration.
- Trust group.
- Social-service adjacency.
- Prior encounters and non-engagement.
- Location on the grid.

The strategy outputs summarize dominant actions by group and local step, allowing comparison of how policy changes behavior.

