"""Render existing numerical elastic arrays; no geology or forward rerun."""

from argparse import ArgumentParser

import numpy as np

from sage_avo import api


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output", help="Optional PNG destination; omit to avoid filesystem writes"
    )
    args = parser.parse_args()
    shape = (32, 12)
    truth = np.stack((np.full(shape, 3000.0), np.full(shape, 1600.0), np.full(shape, 2.3)))
    prior = truth * np.asarray([0.98, 1.02, 1.0])[:, None, None]
    prediction = truth * np.asarray([0.995, 1.005, 1.0])[:, None, None]
    figure = api.plot_elastic_comparison(truth, prior, prediction)
    if args.output:
        figure.savefig(args.output, dpi=150)
    print(f"Rendered {len(figure.axes)} axes from supplied arrays only.")
    import matplotlib.pyplot as plt

    plt.close(figure)


if __name__ == "__main__":
    main()
