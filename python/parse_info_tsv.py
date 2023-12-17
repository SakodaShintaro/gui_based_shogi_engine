""" info.tsvから報酬を得た量を見る
"""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('info_tsv', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.info_tsv, sep='\t')
    print(f"合計 {df['reward'].sum()}成功 {df['reward'].mean() * 100:5.2f}%")
