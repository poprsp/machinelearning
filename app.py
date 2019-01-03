#!/usr/bin/env python3

import argparse

from machinelearning.spiral import Spiral


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spiral-values")
    p.add_argument("--spiral-metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    s = Spiral("data/spiral/spiral.csv")
    if args.spiral_values:
        s.plot_spirals(args.spiral_values)
    if args.spiral_metrics:
        s.plot_metrics(args.spiral_metrics)


if __name__ == "__main__":
    main()
