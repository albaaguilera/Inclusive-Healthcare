from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from learning.qpbrs_seeds import (
    ExperimentConfig,
    plot_figure2_rewards_only,
    plot_figure2_strategies_only,
    plot_figure3_dumbbell,
    run_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE2_FIGSIZE = (10.0, 3.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready multi-seed figures for N=8 and N=16."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--train-max-steps", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--num-sw", type=int, default=15)
    parser.add_argument("--num-peh-values", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--eps-min", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=float, default=0.995)
    parser.add_argument("--pbrs-beta", type=float, default=0.02)
    parser.add_argument("--max-enc", type=int, default=10)
    parser.add_argument("--max-noneng", type=int, default=10)
    parser.add_argument("--healthy-threshold", type=float, default=3.0)
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Generate a fast deterministic layout preview with one seed and short runs. "
            "Use this to approve figure aesthetics before the full paper export."
        ),
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace, *, num_peh: int) -> ExperimentConfig:
    return ExperimentConfig(
        size=args.size,
        num_peh=num_peh,
        num_sw=args.num_sw,
        episodes=args.episodes,
        train_max_steps=args.train_max_steps,
        eval_max_steps=args.eval_max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
        pbrs_beta=args.pbrs_beta,
        max_enc=args.max_enc,
        max_noneng=args.max_noneng,
        healthy_threshold=args.healthy_threshold,
        seeds=tuple(args.seeds),
    )


def save_core_tables(outdir: Path, outputs: Dict[str, object]) -> None:
    for name in ["training_episodes", "eval_summary_by_seed", "eval_summary_aggregated", "strategy_by_seed"]:
        value = outputs.get(name)
        if value is not None:
            value.to_csv(outdir / f"{name}.csv", index=False)


def generate_paper_figures(scale_dir: Path, outputs: Dict[str, object]) -> None:
    figures_dir = scale_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    training_df = outputs["training_episodes"]
    strategy_df = outputs["strategy_by_seed"]

    for scenario in ["OFF", "ON"]:
        plot_figure2_rewards_only(
            training_df,
            scenario=scenario,
            outpath=figures_dir / f"figure2_policy_{scenario.lower()}_rewards.png",
            figsize=FIGURE2_FIGSIZE,
            font_med=17,
            font_small=15,
            legend_font=13.5,
        )
        plot_figure2_strategies_only(
            strategy_df,
            scenario=scenario,
            outpath=figures_dir / f"figure2_policy_{scenario.lower()}_strategies.png",
            figsize=FIGURE2_FIGSIZE,
            font_med=17,
            font_small=15,
            legend_font=13.5,
        )

    plot_figure3_dumbbell(
        outputs["figure3_summary"],
        outputs["representative_eval_artifacts"],
        figures_dir,
        filename="figure3_dumbbell.png",
        figsize=(16.8, 12.2),
        font_big=18.0,
        font_med=15.5,
        font_small=12.6,
    )


def write_summary(root: Path, configs: Iterable[ExperimentConfig], scale_dirs: List[Path]) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "configs": [asdict(cfg) for cfg in configs],
        "scales": [str(path.relative_to(root)) for path in scale_dirs],
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.preview:
        args.seeds = [args.seeds[0] if args.seeds else 0]
        args.episodes = min(args.episodes, 20)
        args.train_max_steps = min(args.train_max_steps, 20)
        args.eval_max_steps = min(args.eval_max_steps, 80)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "preview" if args.preview else "run"
    root = PROJECT_ROOT / "output" / "paper_ready" / f"{prefix}_{run_id}"
    root.mkdir(parents=True, exist_ok=True)

    configs: List[ExperimentConfig] = []
    scale_dirs: List[Path] = []

    for num_peh in args.num_peh_values:
        cfg = build_cfg(args, num_peh=num_peh)
        configs.append(cfg)
        scale_dir = root / f"n{num_peh}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        outputs = run_experiment(cfg)
        save_core_tables(scale_dir, outputs)
        generate_paper_figures(scale_dir, outputs)
        (scale_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        scale_dirs.append(scale_dir)

    write_summary(root, configs, scale_dirs)
    print(f"Saved paper-ready figures to: {root}")


if __name__ == "__main__":
    main()
