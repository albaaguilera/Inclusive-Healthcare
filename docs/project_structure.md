# Project Structure

This page summarizes the main source files and their responsibilities.

## Top-Level Files

| Path | Purpose |
| --- | --- |
| `README.md` | Existing project introduction and quick commands. |
| `requirements.txt` | Python dependencies. |
| `analysis.ipynb` | Notebook analysis. |
| `generate_gif.py` | Utility for generating Figure 3 style policy evolution GIFs. |

## Environment Package

| Path | Purpose |
| --- | --- |
| `environment/model.py` | Main PettingZoo AEC environment, reset logic, step logic, observations, action spaces, social worker movement, rewards, termination, and rendering hooks. |
| `environment/context.py` | Policy context, resources, costs, service locations, action enum, transition tables, and capability score updates. |
| `environment/agent.py` | PEH and social service agent classes. |
| `environment/render.py` | Pygame rendering helper. |

## Learning Package

| Path | Purpose |
| --- | --- |
| `learning/synthetic_data.py` | Synthetic Raval-like demonstrations and Bayesian logistic calibration. |
| `learning/qpbrs_seeds.py` | Main deterministic multi-seed Q-learning with PBRS, evaluation, aggregation, reports, and figures. |
| `learning/qpbrs_scalability.py` | Runs multi-seed experiments across predefined scales and checks equilibrium preservation. |
| `learning/paper_figures.py` | Generates paper-ready figures and core tables for selected population sizes. |
| `learning/utils.py` | Shared plotting, state encoding, masks, IRL potential helpers, logging, artifact loading, and strategy extraction. |
| `learning/qpbrs.py` | Earlier PBRS/Q-learning workflow and plotting code. |
| `learning/qlearningMAexplore.py` | Exploratory Q-learning workflow and visualization utilities. |

## Output Folders

| Path | Purpose |
| --- | --- |
| `output/peh_sample*.json` | PEH profile cohorts used by experiments. |
| `output/irl_calibration_results_raval.json` | Calibration artifact used by IRL potential helpers. |
| `output/robustness_seeds/` | Multi-seed experiment outputs. |
| `output/scalability/` | Scalability sweep outputs. |
| `output/paper_ready/` | Figure-ready outputs for manuscript use. |
| `output/paper_selected/` | Stable selected figures, GIFs, and summaries referenced by the README and paper workflow. |
| `output/figures/` | Standalone figures and strategy plots. |
| `out_datasets/` | Saved evaluation datasets from earlier or custom runs. |

## Typical Development Flow

1. Edit environment dynamics in `environment/model.py` and `environment/context.py`.
2. Edit agent attributes or movement in `environment/agent.py`.
3. Edit training or evaluation logic in `learning/qpbrs_seeds.py`.
4. Edit synthetic calibration in `learning/synthetic_data.py`.
5. Edit plots and output formatting in `learning/utils.py` or the experiment runner that owns the figure.
6. Run a small-seed experiment before launching the full sweep.

Example quick check:

```bash
python -m learning.qpbrs_seeds --seeds 0 --episodes 5 --train-max-steps 20 --eval-max-steps 50
```
