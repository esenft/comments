from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

try:
	from sentence_transformers import SentenceTransformer
except ImportError as error:
	raise SystemExit(
		"Missing dependency 'sentence-transformers'. Install with: "
		"pip install sentence-transformers"
	) from error


THEME_KEYWORDS: dict[str, set[str]] = {
	"Ethics/Safety": {
		"ethic",
		"moral",
		"bias",
		"fair",
		"align",
		"alignment",
		"safe",
		"safety",
		"privacy",
		"censor",
		"harm",
	},
	"Questions/Support": {
		"question",
		"help",
		"how",
		"why",
		"what",
		"can",
		"could",
		"thank",
		"thanks",
		"please",
	},
	"Praise/Approval": {
		"great",
		"amazing",
		"awesome",
		"love",
		"brilliant",
		"best",
		"excellent",
		"thank",
		"thanks",
	},
	"AI Discussion": {
		"ai",
		"gpt",
		"chatgpt",
		"model",
		"intelligence",
		"human",
		"consciousness",
		"thought",
	},
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Cluster comments using sentence embeddings and visualize them in 2D.",
	)
	parser.add_argument("--database", default="comments.db", help="SQLite database path.")
	parser.add_argument("--table", default="comments", help="Table name (default: comments).")
	parser.add_argument("--clusters", type=int, default=5, help="Number of K-means clusters.")
	parser.add_argument(
		"--embedding-model",
		default="all-MiniLM-L6-v2",
		help="SentenceTransformer model name.",
	)
	parser.add_argument(
		"--max-comments",
		type=int,
		default=3000,
		help="Maximum number of comments to process (0 means all).",
	)
	parser.add_argument(
		"--reduction",
		choices=["tsne", "pca"],
		default="tsne",
		help="2D dimensionality reduction method.",
	)
	parser.add_argument(
		"--plot-out",
		default="comment_clusters.png",
		help="Output image path for cluster plot.",
	)
	parser.add_argument(
		"--analysis-out",
		default="cluster_analysis.json",
		help="Output JSON path for cluster analysis.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducible clustering/reduction.",
	)
	return parser.parse_args()


def quote_identifier(identifier: str) -> str:
	escaped = identifier.replace('"', '""')
	return f'"{escaped}"'


def is_truthy(value: object) -> bool:
	text = str(value or "").strip().lower()
	return text in {"1", "true"}


def load_comments(db_path: Path, table: str, max_comments: int) -> pd.DataFrame:
	query = (
		f"SELECT rowid, text, angry, negative, spam, response "
		f"FROM {quote_identifier(table)} "
		"WHERE text IS NOT NULL AND trim(text) <> '' "
		"ORDER BY rowid ASC"
	)
	if max_comments > 0:
		query += f" LIMIT {max_comments}"

	with sqlite3.connect(db_path) as connection:
		df = pd.read_sql_query(query, connection)

	if df.empty:
		raise ValueError("No comments found with non-empty text.")

	return df


def reduce_dimensions(
	embeddings: np.ndarray,
	method: str,
	seed: int,
) -> np.ndarray:
	if method == "pca":
		reducer = PCA(n_components=2, random_state=seed)
		return reducer.fit_transform(embeddings)

	perplexity = min(30, max(5, (len(embeddings) - 1) // 3))
	reducer = TSNE(
		n_components=2,
		random_state=seed,
		perplexity=perplexity,
		init="pca",
		learning_rate="auto",
	)
	return reducer.fit_transform(embeddings)


def top_terms_for_cluster(texts: list[str], top_n: int = 10) -> list[str]:
	if not texts:
		return []

	vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
	tfidf = vectorizer.fit_transform(texts)
	means = np.asarray(tfidf.mean(axis=0)).ravel()
	if means.size == 0:
		return []

	feature_names = np.asarray(vectorizer.get_feature_names_out())
	top_indices = means.argsort()[::-1][:top_n]
	return feature_names[top_indices].tolist()


def infer_theme(top_terms: list[str]) -> str:
	if not top_terms:
		return "General Discussion"

	scores: dict[str, int] = {name: 0 for name in THEME_KEYWORDS}
	for term in top_terms:
		for theme_name, keywords in THEME_KEYWORDS.items():
			if any(keyword in term for keyword in keywords):
				scores[theme_name] += 1

	best_theme = max(scores, key=scores.get)
	if scores[best_theme] == 0:
		return "General Discussion"
	return best_theme


def infer_alignment(
	negative_rate: float,
	angry_rate: float,
	spam_rate: float,
	needs_response_rate: float,
) -> str:
	parts: list[str] = []

	if spam_rate >= 0.01:
		parts.append("spam-like")
	if negative_rate >= 0.03:
		parts.append("negative-leaning")
	elif negative_rate <= 0.015 and angry_rate <= 0.003 and spam_rate == 0:
		parts.append("mostly positive/neutral")
	else:
		parts.append("mixed sentiment")

	if angry_rate >= 0.01:
		parts.append("contains anger")
	if needs_response_rate >= 0.02:
		parts.append("more reply-seeking")

	return ", ".join(parts)


def build_cluster_analysis(df: pd.DataFrame, n_clusters: int) -> dict:
	analysis: dict[str, object] = {
		"total_comments_analyzed": int(len(df)),
		"cluster_count": n_clusters,
		"clusters": [],
	}

	for cluster_id in range(n_clusters):
		cluster_df = df[df["cluster"] == cluster_id]
		count = int(len(cluster_df))
		if count == 0:
			analysis["clusters"].append(
				{
					"cluster_id": cluster_id,
					"size": 0,
					"theme": "General Discussion",
					"alignment": "no data",
					"legend_label": f"C{cluster_id}: General Discussion [no data]",
					"top_terms": [],
					"label_rates": {
						"angry_rate": 0.0,
						"negative_rate": 0.0,
						"spam_rate": 0.0,
						"needs_response_rate": 0.0,
					},
					"sample_comments": [],
				}
			)
			continue

		angry_rate = float(cluster_df["angry_bool"].mean())
		negative_rate = float(cluster_df["negative_bool"].mean())
		spam_rate = float(cluster_df["spam_bool"].mean())
		needs_response_rate = float(cluster_df["response_bool"].mean())

		cluster_texts = cluster_df["text"].astype(str).tolist()
		top_terms = top_terms_for_cluster(cluster_texts, top_n=10)
		samples = cluster_df["text"].astype(str).head(5).tolist()
		theme = infer_theme(top_terms)
		alignment = infer_alignment(
			negative_rate=negative_rate,
			angry_rate=angry_rate,
			spam_rate=spam_rate,
			needs_response_rate=needs_response_rate,
		)
		legend_label = f"C{cluster_id}: {theme} [{alignment}]"

		analysis["clusters"].append(
			{
				"cluster_id": cluster_id,
				"size": count,
				"share_of_dataset": round(count / len(df), 4),
				"theme": theme,
				"alignment": alignment,
				"legend_label": legend_label,
				"top_terms": top_terms,
				"label_rates": {
					"angry_rate": round(angry_rate, 4),
					"negative_rate": round(negative_rate, 4),
					"spam_rate": round(spam_rate, 4),
					"needs_response_rate": round(needs_response_rate, 4),
				},
				"sample_comments": samples,
			}
		)

	return analysis


def build_cluster_label_map(analysis: dict) -> dict[int, str]:
	cluster_map: dict[int, str] = {}
	for item in analysis.get("clusters", []):
		cluster_id = int(item["cluster_id"])
		cluster_map[cluster_id] = str(item.get("legend_label", f"C{cluster_id}"))
	return cluster_map


def plot_clusters(
	df: pd.DataFrame,
	n_clusters: int,
	plot_path: Path,
	cluster_label_map: dict[int, str],
) -> None:
	sns.set_theme(style="whitegrid")
	fig, ax = plt.subplots(figsize=(12, 8))

	palette = sns.color_palette("tab10", n_colors=n_clusters)
	ordered_cluster_ids = sorted(cluster_label_map)
	ordered_labels = [cluster_label_map[cluster_id] for cluster_id in ordered_cluster_ids]
	df_plot = df.copy()
	df_plot["cluster_label"] = df_plot["cluster"].map(cluster_label_map)
	sns.scatterplot(
		data=df_plot,
		x="x",
		y="y",
		hue="cluster_label",
		hue_order=ordered_labels,
		palette=palette,
		s=20,
		alpha=0.7,
		linewidth=0,
		ax=ax,
	)

	centroids = df.groupby("cluster")[["x", "y"]].mean().reset_index()
	ax.scatter(
		centroids["x"],
		centroids["y"],
		marker="X",
		s=300,
		c="black",
		label="Cluster centroid",
	)

	for _, row in centroids.iterrows():
		ax.text(
			row["x"],
			row["y"],
			f"C{int(row['cluster'])}",
			fontsize=9,
			fontweight="bold",
			ha="left",
			va="bottom",
		)

	ax.set_title("Comment Clusters from Sentence Embeddings", fontsize=14)
	ax.set_xlabel("Dimension 1")
	ax.set_ylabel("Dimension 2")
	legend = ax.get_legend()
	if legend is not None:
		legend.set_title("Cluster (Theme + Label Alignment)")
	plt.tight_layout()
	fig.savefig(plot_path, dpi=200)
	plt.close(fig)


def main() -> None:
	args = parse_args()
	if args.clusters < 2:
		raise ValueError("--clusters must be at least 2.")
	if args.max_comments < 0:
		raise ValueError("--max-comments must be 0 or a positive integer.")

	db_path = Path(args.database)
	if not db_path.exists():
		raise FileNotFoundError(f"Database not found: {db_path}")

	df = load_comments(db_path, args.table, args.max_comments)

	texts = df["text"].astype(str).tolist()
	model = SentenceTransformer(args.embedding_model)
	embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

	kmeans = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed)
	df["cluster"] = kmeans.fit_predict(embeddings)

	coords_2d = reduce_dimensions(embeddings, args.reduction, args.seed)
	df["x"] = coords_2d[:, 0]
	df["y"] = coords_2d[:, 1]

	df["angry_bool"] = df["angry"].apply(is_truthy)
	df["negative_bool"] = df["negative"].apply(is_truthy)
	df["spam_bool"] = df["spam"].apply(is_truthy)
	df["response_bool"] = df["response"].apply(is_truthy)

	analysis = build_cluster_analysis(df, args.clusters)
	cluster_label_map = build_cluster_label_map(analysis)

	plot_path = Path(args.plot_out)
	analysis_path = Path(args.analysis_out)

	plot_clusters(df, args.clusters, plot_path, cluster_label_map)
	analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

	print(f"Loaded comments: {len(df)}")
	print(f"Embedding model: {args.embedding_model}")
	print(f"Clusters: {args.clusters}")
	print(f"Reduction: {args.reduction}")
	for item in analysis["clusters"]:
		print(
			f"C{item['cluster_id']}: {item['theme']} | {item['alignment']} "
			f"(n={item['size']})"
		)
	print(f"Saved cluster plot: {plot_path}")
	print(f"Saved analysis JSON: {analysis_path}")


if __name__ == "__main__":
	main()
