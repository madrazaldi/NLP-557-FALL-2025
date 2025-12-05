import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for the emotion dataset."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("train.csv"),
        help="Path to the training CSV (expects a 'text' column plus label columns).",
    )
    parser.add_argument(
        "--dev-path",
        type=Path,
        default=Path("dev.csv"),
        help="Optional dev/validation CSV to summarize alongside train.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Name of the text column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eda"),
        help="Directory to save plots and summary tables.",
    )
    parser.add_argument(
        "--top-combos",
        type=int,
        default=10,
        help="How many most-common label combinations to report.",
    )
    return parser.parse_args()


def detect_label_cols(df: pd.DataFrame, text_col: str) -> List[str]:
    candidates = [c for c in df.columns if c != text_col]
    if not candidates:
        raise ValueError(f"No label columns found (checked everything except '{text_col}').")
    return candidates


def add_length_columns(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df["char_len"] = df[text_col].astype(str).str.len()
    df["word_len"] = df[text_col].astype(str).str.split().apply(len)
    return df


def label_distribution(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    counts = df[label_cols].sum().astype(int)
    pct = df[label_cols].mean()
    return pd.DataFrame({"label": label_cols, "count": counts.values, "pct": pct.values})


def label_combinations(df: pd.DataFrame, label_cols: List[str], top_n: int) -> pd.DataFrame:
    combos = (
        df[label_cols]
        .apply(lambda row: tuple([c for c, v in row.items() if v == 1]), axis=1)
        .value_counts()
        .reset_index()
        .rename(columns={"index": "labels", 0: "count"})
    )
    combos["labels"] = combos["labels"].apply(lambda t: ", ".join(t) if t else "none")
    return combos.head(top_n)


def plot_label_counts(dist_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(x="label", y="count", data=dist_df, palette="crest")
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_text_lengths(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    sns.histplot(df["word_len"], bins=40, kde=True, color="#4c72b0")
    plt.title(title)
    plt.xlabel("Words per sample")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, label_cols: List[str], out_path: Path, title: str) -> None:
    corr = df[label_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.75})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cooccurrence(df: pd.DataFrame, label_cols: List[str], out_path: Path, title: str) -> None:
    mat = df[label_cols].T @ df[label_cols]
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, cmap="YlGnBu", annot=True, fmt=".0f", square=True, cbar_kws={"shrink": 0.75})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_split(name: str, df: pd.DataFrame, text_col: str, output_dir: Path, top_n: int) -> Dict:
    label_cols = detect_label_cols(df, text_col)
    df_len = add_length_columns(df, text_col)

    dist_df = label_distribution(df_len, label_cols)
    combo_df = label_combinations(df_len, label_cols, top_n)

    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)

    dist_df.to_csv(split_dir / "label_distribution.csv", index=False)
    combo_df.to_csv(split_dir / "top_label_combinations.csv", index=False)

    plot_label_counts(dist_df, split_dir / "label_distribution.png", f"{name} label counts")
    plot_text_lengths(df_len, split_dir / "text_length_hist.png", f"{name} text length (words)")
    plot_corr_heatmap(df_len, label_cols, split_dir / "label_correlation.png", f"{name} label correlation")
    plot_cooccurrence(df_len, label_cols, split_dir / "label_cooccurrence.png", f"{name} label co-occurrence")

    text_stats = df_len[["char_len", "word_len"]].describe().to_dict()
    label_density = dist_df[["label", "pct"]].set_index("label")["pct"].to_dict()

    summary = {
        "rows": len(df),
        "missing_text": int(df[text_col].isna().sum()),
        "text_stats": text_stats,
        "label_frequency_pct": label_density,
        "top_combinations": combo_df.to_dict(orient="records"),
    }

    with open(split_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}

    if args.train_path.exists():
        train_df = pd.read_csv(args.train_path)
        summaries["train"] = summarize_split("train", train_df, args.text_col, output_dir, args.top_combos)
    else:
        raise FileNotFoundError(f"Train file not found: {args.train_path}")

    if args.dev_path and args.dev_path.exists():
        dev_df = pd.read_csv(args.dev_path)
        summaries["dev"] = summarize_split("dev", dev_df, args.text_col, output_dir, args.top_combos)

    with open(output_dir / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    for split, summary in summaries.items():
        print(f"{split} -> rows={summary['rows']}, missing_text={summary['missing_text']}")
        print(f"  median words: {summary['text_stats']['word_len']['50%']:.1f}")
        top_combo = summary["top_combinations"][0] if summary["top_combinations"] else {"labels": "n/a", "count": 0}
        print(f"  top combo: {top_combo['labels']} ({top_combo['count']})")


if __name__ == "__main__":
    main()
