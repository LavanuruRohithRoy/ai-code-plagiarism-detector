import pandas as pd

CSV_PATH = "data/results/evaluation_results.csv"

df = pd.read_csv(CSV_PATH)

print("\n=== SAMPLE COUNTS ===")
print(df["label"].value_counts())

print("\n=== PLAGIARISM % ===")
print(df.groupby("label")["plagiarism_percentage"].describe())

print("\n=== AI PROBABILITY ===")
print(df.groupby("label")["ai_probability"].describe())

if "confidence" in df.columns:
	print("\n=== CONFIDENCE COUNTS ===")
	print(df.groupby("label")["confidence"].value_counts())

if "top_match_label" in df.columns:
	print("\n=== TOP MATCH LABEL CROSS-TAB ===")
	print(pd.crosstab(df["label"], df["top_match_label"], normalize="index"))
