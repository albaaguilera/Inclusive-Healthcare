# Experiments and Outputs

This page describes the main experiment entry points and the files they produce.

## Main Experiment: Multi-Seed PBRS/Q-Learning

Run from the project root:

```bash
python -m learning.qpbrs_seeds --seeds 0 1 2 3 4
```

Useful options:

```bash
python -m learning.qpbrs_seeds \
  --num-peh 8 \
  --num-sw 15 \
  --size 7 \
  --episodes 400 \
  --train-max-steps 100 \
  --eval-max-steps 500 \
  --seeds 0 1 2 3 4
```

Outputs are written to:

```text
output/robustness_seeds/run_<timestamp>/
```

Key files:

| File | Meaning |
| --- | --- |
| `training_episodes.csv` | Per-episode training reward traces for each scenario and seed. |
| `training_summary_by_seed.csv` | Training statistics for each scenario and seed. |
| `training_summary_aggregated.csv` | Training means and standard deviations across seeds. |
| `eval_steps.csv` | Step-level greedy evaluation trace. |
| `eval_agents.csv` | Final per-agent evaluation states and returns. |
| `eval_summary_by_seed.csv` | Scenario-level evaluation metrics per seed. |
| `eval_summary_aggregated.csv` | Evaluation means and standard deviations across seeds. |
| `strategy_by_seed.csv` | Dominant actions by scenario, seed, group, and local step. |
| `strategy_summary_aggregated.csv` | Strategy consensus across seeds. |
| `policy_difference.csv` | Paired `ON - OFF` differences by metric. |
| `summary.md` | Human-readable run report. |
| `summary.json` | Config and output manifest. |

## Scalability Sweep

Run:

```bash
python -m learning.qpbrs_scalability
```

Outputs are written to:

```text
output/scalability/run_<timestamp>/
```

The sweep evaluates predefined scale labels such as:

- `n4_sw8_sz5`
- `n8_sw15_sz7`
- `n16_sw20_sz9`
- `n16_sw30_sz11`

Each scale has a subfolder under:

```text
output/scalability/run_<timestamp>/scales/<scale_label>/
```

The root scalability folder also includes:

- `scalability_summary.csv`
- `summary.md`
- `summary.json`
- `figures/policy_gaps_by_scale.png`
- `figures/strategy_alignment_by_scale.png`
- `figures/equilibrium_preservation_score.png`

The scalability criterion checks whether policy effects and dominant strategies remain aligned with the baseline scale.

## Paper-Ready Figures

Run:

```bash
python -m learning.paper_figures
```

For a fast layout preview before the full run:

```bash
python -m learning.paper_figures --preview --num-peh-values 16 25
```

By default this generates figures for selected population sizes across ten seeds:

```text
output/paper_ready/run_<timestamp>/
```

For each selected population size, the script saves:

- `training_episodes.csv`
- `eval_summary_by_seed.csv`
- `eval_summary_aggregated.csv`
- `strategy_by_seed.csv`
- `config.json`
- `figures/figure2_policy_off_rewards.png`
- `figures/figure2_policy_off_strategies.png`
- `figures/figure2_policy_on_rewards.png`
- `figures/figure2_policy_on_strategies.png`
- `figures/figure3_dumbbell.png`

## Figure 3 Style GIF

Run:

```bash
python generate_gif.py
```

This trains one deterministic seed and renders an animated comparison using the same design language as the current Figure 3. The default output is:

```text
output/paper_selected/policy_evolution_figure3_style_n16.gif
```

## Selected Paper Outputs

To avoid linking the README to timestamped folders, the repository keeps the chosen paper assets under:

```text
output/paper_selected/
```

This stable folder includes:

- curated `N=16` and `N=25` paper figures
- the selected scalability summary and plots
- the README GIF

## Legacy and Exploratory Scripts

The repository also contains:

- `learning.qpbrs`: earlier single-run PBRS/Q-learning and plotting workflow.
- `learning.qlearningMAexplore`: exploratory Q-learning and visualization utilities.
- `analysis.ipynb`: notebook-based analysis.
- `generate_gif.py`: Figure 3 style GIF generation for policy evolution visuals.

These are useful for development history and custom analysis, but `qpbrs_seeds`, `qpbrs_scalability`, and `paper_figures` are the clearest reproducible entry points.

## Reading CSV Outputs

Recommended first files to inspect:

1. `eval_summary_aggregated.csv` for high-level `ON` vs `OFF` metrics.
2. `policy_difference.csv` for paired differences across seeds.
3. `strategy_summary_aggregated.csv` for dominant action patterns.
4. `eval_agents.csv` for group and individual final states.
5. `training_episodes.csv` to inspect learning stability.

## Common Metrics

Evaluation summaries include:

- `eval_total_reward`
- `eval_mean_reward_per_agent`
- `final_mean_health`
- `final_share_registered`
- `final_share_healthy`
- `final_mean_cap_bh`
- `final_mean_cap_af`
- `healthcare_spend`
- `social_service_spend`
- `eval_total_steps`

Group-specific reward metrics are reported for:

- `NONREG_LOW`
- `NONREG_MOD`
- `REG_LOW`
- `REG_MOD`
