#!/usr/bin/env python3

import argparse
import timeit

from machinelearning.mnist import MNIST
from machinelearning.spiral import Spiral


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spiral-values")
    p.add_argument("--spiral-metrics")
    p.add_argument("--mnist-conv-net", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    s = Spiral("data/spiral/spiral.csv")
    if args.spiral_values:
        s.plot_spirals(args.spiral_values)
    if args.spiral_metrics:
        s.plot_metrics(args.spiral_metrics)

    m = MNIST()
    if args.mnist_conv_net:
        start = timeit.default_timer()
        conv_net = m.conv_net()
        end = timeit.default_timer()

        print("ConvNet ({:.2f} seconds):".format(end - start))
        for key, value in conv_net.items():
            print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
