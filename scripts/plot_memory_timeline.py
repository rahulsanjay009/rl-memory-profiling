"""Plot a memory_poller.py CSV trace as memory-vs-time, colored by phase."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

PHASE_COLORS = {
    "rollout": "#4C72B0",
    "update": "#DD8452",
    "idle": "#B0B0B0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default=None, help="output image path; shows interactively if omitted")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df["t"] = df["timestamp"] - df["timestamp"].min()
    df["allocated_gb"] = df["allocated_bytes"] / 1e9

    fig, ax = plt.subplots(figsize=(10, 4))
    for phase, group in df.groupby("phase"):
        ax.scatter(group["t"], group["allocated_gb"], s=4, label=phase,
                   color=PHASE_COLORS.get(phase, "#333333"))

    ax.set_xlabel("time (s)")
    ax.set_ylabel("GPU memory allocated (GB)")
    ax.set_title("GPU memory over rollout/update phases")
    ax.legend()
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
