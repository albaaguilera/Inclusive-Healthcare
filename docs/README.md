# Inclusive Healthcare Simulation Documentation

This documentation describes the Inclusive Healthcare Simulation project: an agent-based, multi-agent reinforcement learning environment for studying how inclusive healthcare policy changes the opportunities and outcomes of people experiencing homelessness (PEH).

The code models PEH agents moving through a grid-based social and healthcare environment. Each PEH agent has health, administrative registration status, trust-related attributes, engagement history, and access to a changing set of possible actions. Social service agents move toward PEH agents and can make social-service engagement feasible. The main policy intervention is an inclusive healthcare rule that allows non-registered PEH agents to receive medical attention.

## Documentation Map

- [Simulation Overview](simulation.md): the environment, policy scenarios, state variables, actions, rewards, budgets, capabilities, and episode flow.
- [Agent Behavior](agents.md): PEH and social service agent attributes, movement, interactions, action feasibility, and group definitions.
- [Learning Algorithm](learning_algorithm.md): Q-learning, action masking, potential-based reward shaping, training, evaluation, and multi-seed aggregation.
- [Synthetic Data and Calibration](synthetic_data_and_calibration.md): synthetic Raval-like demonstrations and Bayesian logistic calibration used as the IRL potential.
- [Experiments and Outputs](experiments_and_outputs.md): how to run the main scripts and how output folders, CSV files, figures, and summaries are organized.
- [Project Structure](project_structure.md): where the main source files live and what each module is responsible for.

## Quick Start

Install the Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

Generate synthetic data and calibration artifacts:

```bash
python -m learning.synthetic_data
```

Run the deterministic multi-seed PBRS/Q-learning experiment:

```bash
python -m learning.qpbrs_seeds --seeds 0 1 2 3 4
```

Run the scalability sweep:

```bash
python -m learning.qpbrs_scalability
```

Generate paper-ready figures for selected population sizes:

```bash
python -m learning.paper_figures
```

Generate a fast paper-figure preview before the full run:

```bash
python -m learning.paper_figures --preview --num-peh-values 16 25
```

Generate the Figure 3 style policy-evolution GIF:

```bash
python generate_gif.py
```

## Core Idea

The simulation compares two policy scenarios:

- `policy OFF`: medical attention is available only to registered PEH agents.
- `policy ON`: inclusive healthcare is active, so medical attention is available regardless of registration.

Agents learn policies in each scenario. The evaluation then compares final health, registration, capabilities, rewards, costs, and dominant strategies across PEH groups.

## Main Outputs

Experiment scripts write outputs under `output/`, usually in timestamped run folders:

- `training_episodes.csv`: per-episode training rewards.
- `eval_steps.csv`: step-level evaluation trace.
- `eval_agents.csv`: final per-agent evaluation outcomes.
- `eval_summary_by_seed.csv`: evaluation metrics by scenario and seed.
- `eval_summary_aggregated.csv`: mean and standard deviation across seeds.
- `strategy_by_seed.csv` and `strategy_summary_aggregated.csv`: dominant actions by group and local step.
- `summary.md` and `summary.json`: run metadata and compact experiment reports.
- `figures/`: generated plots for policy comparison, rewards, strategies, and scalability.

The repository also keeps a curated set of selected paper assets under:

- `output/paper_selected/n16/`
- `output/paper_selected/n25/`
- `output/paper_selected/scalability/`
- `output/paper_selected/policy_evolution_figure3_style_n16.gif`

## Reading the Results

The most important comparison is `ON` versus `OFF`. In general, look for:

- Higher final health and share of healthy agents under `ON`.
- Higher or more stable evaluation rewards under `ON`.
- Changes in registration and engagement patterns.
- Capability scores for Bodily Health and Affiliation.
- Healthcare and social-service spending differences.
- Whether the same dominant strategies are preserved across seeds and scales.

## Important Implementation Notes

The environment follows PettingZoo's AEC style: PEH agents act one at a time through an `agent_selection` cycle. Social service agents are environmental actors rather than learning agents; they move during PEH steps to create or remove opportunities for PEH agents.

The robust experiment path is `learning.qpbrs_seeds`. The older `learning.qpbrs` and `learning.qlearningMAexplore` modules contain earlier plotting and exploration code and are useful for historical context, but the multi-seed and scalability scripts are the clearest reproducible entry points.
