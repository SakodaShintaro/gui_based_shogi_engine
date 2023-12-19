import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tsv_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tsv_path = args.tsv_path
    df = pd.read_csv(tsv_path, sep='\t')
    print(df.head())

    window_size = 200
    mean = df["is_ideal_action"].rolling(window_size).mean()

    plt.plot(df["iteration"], mean)
    plt.xlabel("iteration")
    plt.ylabel(f"rolling mean(window_size={window_size})")
    plt.ylim(0, 1)
    save_dir = os.path.dirname(tsv_path)
    save_path = f"{save_dir}/grid_world_log.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"saved to {save_path}")
