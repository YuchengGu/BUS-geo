#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visual_guided_collection_gui.offline_bayes import LocalBayesOptimizer, LocalBOConfig


def parse_bounds(text: str) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            _, item = item.split("=", 1)
        low_s, high_s = item.split(",", 1)
        bounds.append((float(low_s), float(high_s)))
    if not bounds:
        raise argparse.ArgumentTypeError("bounds cannot be empty")
    return bounds


def load_prior(path: str | None) -> list[tuple[np.ndarray, float]]:
    if path is None:
        return []
    payload = json.loads(Path(path).expanduser().read_text())
    observations = payload.get("observations", payload if isinstance(payload, list) else [])
    out: list[tuple[np.ndarray, float]] = []
    for obs in observations:
        if isinstance(obs, dict):
            x = obs["x"]
            y = obs.get("F", obs.get("y"))
        else:
            x, y = obs
        out.append((np.asarray(x, dtype=float), float(y)))
    return out


def resolve_total_trials(*, n_initial: int, n_ei: int) -> int:
    if n_initial < 1:
        raise ValueError("--n-initial must be at least 1")
    if n_ei < 1:
        raise ValueError("--n-ei must be at least 1")
    return int(n_initial) + int(n_ei)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ask/tell Bayesian optimization demo.")
    parser.add_argument(
        "--bounds",
        type=parse_bounds,
        default=parse_bounds("dn=-0.005,0.005;rx=-5,5;ry=-5,5;rz=-5,5"),
        help='Bounds, e.g. "dn=-0.005,0.005;rx=-5,5;ry=-5,5;rz=-5,5".',
    )
    parser.add_argument("--prior-json", type=str, default=None, help="JSON with observations: [{'x': [...], 'F': ...}].")
    parser.add_argument("--n-initial", type=int, default=3)
    parser.add_argument("--n-ei", type=int, default=12, help="Number of EI-selected trials after the initial random trials.")
    parser.add_argument(
        "--backend",
        choices=["auto", "skopt", "sklearn_ei"],
        default="auto",
        help="auto uses scikit-optimize if installed, otherwise sklearn + local EI.",
    )
    parser.add_argument("--candidate-count", type=int, default=4096)
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--xi-boost", type=float, default=0.1)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--stagnation-window", type=int, default=3)
    parser.add_argument("--convergence-window", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    try:
        total_trials = resolve_total_trials(n_initial=args.n_initial, n_ei=args.n_ei)
    except ValueError as exc:
        parser.error(str(exc))

    optimizer = LocalBayesOptimizer(
        LocalBOConfig(
            bounds=args.bounds,
            n_initial=args.n_initial,
            max_trials=total_trials,
            backend=args.backend,
            xi=args.xi,
            xi_boost=args.xi_boost,
            candidate_count=args.candidate_count,
            stagnation_window=args.stagnation_window,
            convergence_window=args.convergence_window,
            min_improvement=args.min_improvement,
            random_state=args.random_state,
        )
    )
    for x, y in load_prior(args.prior_json):
        optimizer.tell(x, y)

    print("Interactive BO minimizes F(x). Enter an observed F for each suggested x.")
    print(f"Budget: {args.n_initial} initial random trials + {total_trials - args.n_initial} EI trials = {total_trials} total trials.")
    print("Type 'q' to stop.\n")
    while not optimizer.should_stop():
        x_next = optimizer.ask()
        trial = optimizer.n_observed + 1
        if optimizer.n_observed < args.n_initial:
            phase = f"initial {optimizer.n_observed + 1}/{args.n_initial}"
        else:
            ei_index = optimizer.n_observed - args.n_initial + 1
            label = "EI boost" if optimizer.last_ask_used_boost else "EI"
            phase = f"{label} {ei_index}/{total_trials - args.n_initial}"
        print(f"trial {trial}/{total_trials} [{phase}] x_next = {x_next.tolist()}")
        raw = input("observed F = ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break
        optimizer.tell(x_next, float(raw))
        print(f"best F = {optimizer.best_y:.6g}, best x = {optimizer.best_x.tolist()}\n")

    if optimizer.best_x is None:
        print("No observations were recorded.")
    else:
        print("Final best:")
        print(json.dumps({"x": optimizer.best_x.tolist(), "F": optimizer.best_y}, indent=2))


if __name__ == "__main__":
    main()
