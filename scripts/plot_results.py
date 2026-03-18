import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = "data/results/evaluation_results.csv"
DEFAULT_OUT_DIR = "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot offline evaluation results")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to evaluation CSV")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where plot images are saved",
    )
    parser.add_argument(
        "--per-language",
        action="store_true",
        help="Generate additional per-language plots if language column exists",
    )
    return parser.parse_args()


def _save_boxplot(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: str) -> None:
    grouped = [group[value_col].dropna().values for _, group in df.groupby("label")]
    labels = [name for name, _ in df.groupby("label")]

    if not grouped:
        return

    plt.figure()
    plt.boxplot(grouped, tick_labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_histogram(df: pd.DataFrame, value_col: str, title: str, xlabel: str, out_path: str) -> None:
    plt.figure()
    for label, group in df.groupby("label"):
        plt.hist(group[value_col], bins=20, alpha=0.7, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    plot_df = df[["label", x_col, y_col]].dropna()
    if plot_df.empty:
        return

    plt.figure()
    for label, group in plot_df.groupby("label"):
        plt.scatter(group[x_col], group[y_col], alpha=0.5, s=12, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    has_cross_label = "ai_affinity" in df.columns
    metric_col = "ai_affinity" if has_cross_label else "ai_probability"
    
    if has_cross_label:
        metric_title = "Label Affinity Distribution (Preference for Same-Label vs Opposite-Label)"
        metric_ylabel = "Label Affinity (%)"
        plagiarism_title = "Similarity to Opposite-Label Code Distribution (Cross-Label Mode)"
    else:
        metric_title = "AI Probability Distribution"
        metric_ylabel = "AI Probability (%)"
        plagiarism_title = "Plagiarism Percentage Distribution"
    
    metric_filename = "ai_affinity_boxplot.png" if has_cross_label else "ai_probability_boxplot.png"

    _save_boxplot(
        df,
        value_col="plagiarism_percentage",
        title=plagiarism_title,
        ylabel="Similarity (%)",
        out_path=os.path.join(args.out_dir, "plagiarism_boxplot.png"),
    )

    _save_boxplot(
        df,
        value_col=metric_col,
        title=metric_title,
        ylabel=metric_ylabel,
        out_path=os.path.join(args.out_dir, metric_filename),
    )

    _save_histogram(
        df,
        value_col=metric_col,
        title=f"{metric_ylabel} Histogram",
        xlabel=metric_ylabel,
        out_path=os.path.join(args.out_dir, f"{metric_col}_histogram.png"),
    )

    _save_histogram(
        df,
        value_col="plagiarism_percentage",
        title="Similarity to Opposite-Label Code Histogram",
        xlabel="Similarity (%)",
        out_path=os.path.join(args.out_dir, "plagiarism_histogram.png"),
    )

    if args.per_language and "language" in df.columns:
        for language, group in df.groupby("language"):
            if group["label"].nunique() < 1:
                continue

            safe_language = language.replace("/", "_").replace(" ", "_")

            _save_boxplot(
                group,
                value_col="plagiarism_percentage",
                title=f"Plagiarism % Distribution ({language})",
                ylabel="Plagiarism %",
                out_path=os.path.join(args.out_dir, f"plagiarism_boxplot_{safe_language}.png"),
            )

            _save_boxplot(
                group,
                value_col=metric_col,
                title=f"{metric_ylabel} Distribution ({language})",
                ylabel=metric_ylabel,
                out_path=os.path.join(args.out_dir, f"{metric_col}_boxplot_{safe_language}.png"),
            )

    _save_scatter(
        df,
        x_col="cross_label_sim",
        y_col="same_label_sim",
        title="Cross-label vs Same-label Similarity",
        xlabel="Cross-label Similarity",
        ylabel="Same-label Similarity",
        out_path=os.path.join(args.out_dir, "cross_vs_same_similarity_scatter.png"),
    )

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
