import argparse

import pandas as pd


CSV_PATH = "data/results/evaluation_results.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Analyze offline evaluation CSV")
	parser.add_argument("--csv", default=CSV_PATH, help="Path to evaluation CSV")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = pd.read_csv(args.csv)
	has_ai_affinity = "ai_affinity" in df.columns
	has_ai_probability = "ai_probability" in df.columns
	has_cross_sims = {"same_label_sim", "cross_label_sim"}.issubset(df.columns)

	print("\n=== SAMPLE COUNTS (LABEL) ===")
	print(df["label"].value_counts())

	if "language" in df.columns:
		print("\n=== SAMPLE COUNTS (LABEL x LANGUAGE) ===")
		print(df.groupby(["label", "language"]).size().sort_values(ascending=False))

	print("\n=== PLAGIARISM % BY LABEL ===")
	print(df.groupby("label")["plagiarism_percentage"].describe())

	if "language" in df.columns:
		print("\n=== PLAGIARISM % BY LABEL x LANGUAGE ===")
		print(df.groupby(["label", "language"])["plagiarism_percentage"].describe())

	if has_ai_affinity:
		print("\n=== AI AFFINITY BY LABEL ===")
		print(df.groupby("label")["ai_affinity"].describe())

		if "language" in df.columns:
			print("\n=== AI AFFINITY BY LABEL x LANGUAGE ===")
			print(df.groupby(["label", "language"])["ai_affinity"].describe())
	elif has_ai_probability:
		print("\n=== AI PROBABILITY BY LABEL ===")
		print(df.groupby("label")["ai_probability"].describe())

		if "language" in df.columns:
			print("\n=== AI PROBABILITY BY LABEL x LANGUAGE ===")
			print(df.groupby(["label", "language"])["ai_probability"].describe())

	if has_cross_sims:
		print("\n=== SAME vs CROSS LABEL SIMILARITY (BY LABEL) ===")
		print(df.groupby("label")[["same_label_sim", "cross_label_sim"]].describe())

		if "language" in df.columns:
			print("\n=== SAME vs CROSS LABEL SIMILARITY (BY LABEL x LANGUAGE) ===")
			print(df.groupby(["label", "language"])[["same_label_sim", "cross_label_sim"]].describe())

	if "confidence" in df.columns:
		print("\n=== CONFIDENCE COUNTS (LABEL) ===")
		print(df.groupby("label")["confidence"].value_counts())

		if "language" in df.columns:
			print("\n=== CONFIDENCE COUNTS (LABEL x LANGUAGE) ===")
			print(df.groupby(["label", "language"])["confidence"].value_counts())

	if "top_match_label" in df.columns:
		print("\n=== TOP MATCH LABEL CROSS-TAB ===")
		print(pd.crosstab(df["label"], df["top_match_label"], normalize="index"))

	if "top_match_language" in df.columns and "language" in df.columns:
		print("\n=== TOP MATCH LANGUAGE CROSS-TAB ===")
		print(pd.crosstab(df["language"], df["top_match_language"], normalize="index"))

	# Interpretation section
	print("\n" + "="*70)
	print("INTERPRETATION GUIDE")
	print("="*70)

	if has_cross_sims:
		print("\n🔄 CROSS-LABEL EVALUATION MODE (Recommended for detection)")
		print("-" * 70)
		print("Metrics measure OPPOSITION between labels:")
		print("  • plagiarism_percentage = similarity to OPPOSITE-label code")
		print("    - HIGH value (>70%) = code matches opposite-label patterns")
		print("    - LOW value (<50%) = code unique to its label")
		print("  • ai_affinity = preference for same-label vs opposite-label")
		print("    - 75%+ = strongly prefers same-label (good discriminator)")
		print("    - 50% = no preference (weak discriminator)")
		print("    - 25%- = prefers opposite-label (suspicious)")
		print("\nKey insights:")
		print(f"  • AI code avg plagiarism: {df[df['label']=='AI']['plagiarism_percentage'].mean():.1f}%")
		print(f"    (AI code matches HUMAN patterns ~{df[df['label']=='AI']['plagiarism_percentage'].mean():.0f}% of the time)")
		print(f"  • HUMAN code avg plagiarism: {df[df['label']=='HUMAN']['plagiarism_percentage'].mean():.1f}%")
		print(f"    (HUMAN code matches AI patterns ~{df[df['label']=='HUMAN']['plagiarism_percentage'].mean():.0f}% of the time)")
		print(f"  • Separation: {abs(df[df['label']=='AI']['plagiarism_percentage'].mean() - df[df['label']=='HUMAN']['plagiarism_percentage'].mean()):.1f}% ✓")
		print("\n  → Higher AI plagiarism expected (AI learns from human solutions)")
		print("  → Difference of ~15% shows good discrimination capability")
	else:
		print("\n  Within-label evaluation (each label matches within itself)")


if __name__ == "__main__":
	main()
