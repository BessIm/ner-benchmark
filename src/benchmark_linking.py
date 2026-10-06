#!/usr/bin/env python3

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("results/sauerkraut_gliner")
OUTPUT_DIR = Path("results/linking")

ENTITY_LABEL = "PERSONNE"

# Seuils fuzzy a tester
FUZZY_THRESHOLDS = [75, 80, 85, 90, 95]

# Seuil de fusion pour la combinaison fuzzy + phonétique
COMBINED_THRESHOLD = 82
COMBINED_MIN_FUZZY = 65

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_NEIGHBORS = 10
EMBEDDING_THRESHOLD = 0.72

# Pour éviter les clusters absurdes
MIN_MENTION_LENGTH = 3


# ============================================================
# NORMALISATION
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).strip().casefold()

    # Supprimer les accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    # Ponctuation -> espace
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Civilites simples
    text = re.sub(
        r"^(m|m\.|monsieur|mme|mme\.|madame|dr|dr\.)\s+",
        "",
        text,
    )

    return text


def phonetic_code(text: str):
    words = normalize_text(text).split()

    primary = []
    secondary = []

    for word in words:
        p, s = doublemetaphone(word)
        if p:
            primary.append(p)
        if s:
            secondary.append(s)

    return " ".join(primary), " ".join(secondary)


def phonetic_similarity(a: str, b: str) -> float:
    a1, a2 = phonetic_code(a)
    b1, b2 = phonetic_code(b)

    scores = []
    if a1 and b1:
        scores.append(fuzz.ratio(a1, b1))
    if a2 and b2:
        scores.append(fuzz.ratio(a2, b2))

    return max(scores) if scores else 0.0


# ============================================================
# UNION-FIND POUR CONSTRUIRE LES CLUSTERS
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ============================================================
# CHARGEMENT DES MENTIONS
# ============================================================

def load_mentions():
    rows = []

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(
            f"Aucun fichier JSON dans {INPUT_DIR}"
        )

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entity in data.get("entities", []):
            if entity.get("label") != ENTITY_LABEL:
                continue

            text = str(entity.get("text", "")).strip()
            normalized = normalize_text(text)

            if len(normalized) < MIN_MENTION_LENGTH:
                continue

            rows.append({
                "file": path.name,
                "mention": text,
                "normalized": normalized,
                "score": float(entity.get("score", 0.0)),
                "start": entity.get("start"),
                "end": entity.get("end"),
            })

    if not rows:
        raise RuntimeError("Aucune PERSONNE trouvee dans les resultats.")

    return pd.DataFrame(rows)


# ============================================================
# REFERENTIEL CANDIDAT AUTOMATIQUE
# ============================================================

def build_candidates(df):
    """
    Un candidat = une forme textuelle normalisee distincte.
    Il ne s'agit PAS encore d'une identite reelle.
    """

    grouped = (
        df.groupby("normalized")
        .agg(
            occurrences=("mention", "size"),
            score_moyen=("score", "mean"),
            score_median=("score", "median"),
            formes=("mention", lambda x: sorted(set(x))),
            fichiers=("file", lambda x: sorted(set(x))),
        )
        .reset_index()
    )

    grouped["candidate_id"] = [
        f"CAND_{i:06d}"
        for i in range(1, len(grouped) + 1)
    ]

    # Le nom canonique initial est la forme qui apparait le plus souvent.
    frequency = (
        df.groupby(["normalized", "mention"])
        .size()
        .reset_index(name="n")
        .sort_values(["normalized", "n", "mention"], ascending=[True, False, True])
    )

    canonical = (
        frequency.drop_duplicates("normalized")
        .set_index("normalized")["mention"]
    )

    grouped["candidate_name"] = grouped["normalized"].map(canonical)

    return grouped


# ============================================================
# INDEX PHONETIQUE
# ============================================================

def build_phonetic_index(candidate_df):
    primary = defaultdict(list)

    for idx, row in candidate_df.iterrows():
        p, s = phonetic_code(row["candidate_name"])

        if p:
            primary[p].append(idx)
        if s and s != p:
            primary[s].append(idx)

    return primary


# ============================================================
# CLUSTERS : FUZZY
# ============================================================

def build_fuzzy_clusters(candidate_df, threshold):
    names = candidate_df["normalized"].tolist()
    n = len(names)
    uf = UnionFind(n)

    for i, name in enumerate(names):
        matches = process.extract(
            name,
            names,
            scorer=fuzz.WRatio,
            limit=20,
            score_cutoff=threshold,
        )

        for match_name, score, j in matches:
            if j <= i:
                continue
            if score >= threshold:
                uf.union(i, j)

    return uf


# ============================================================
# CLUSTERS : PHONETIQUE + FUZZY
# ============================================================

def build_combined_clusters(candidate_df):
    names = candidate_df["normalized"].tolist()
    n = len(names)
    uf = UnionFind(n)

    # Recherche fuzzy initiale
    for i, name in enumerate(names):
        matches = process.extract(
            name,
            names,
            scorer=fuzz.WRatio,
            limit=20,
            score_cutoff=COMBINED_MIN_FUZZY,
        )

        for match_name, fuzzy_score, j in matches:
            if j <= i:
                continue

            if fuzzy_score < COMBINED_MIN_FUZZY:
                continue

            phonetic_score = phonetic_similarity(
                candidate_df.iloc[i]["candidate_name"],
                candidate_df.iloc[j]["candidate_name"],
            )

            combined = 0.45 * fuzzy_score + 0.55 * phonetic_score

            if combined >= COMBINED_THRESHOLD:
                uf.union(i, j)

    return uf


# ============================================================
# CLUSTERS : EMBEDDINGS
# ============================================================

def build_embedding_clusters(candidate_df):
    print("\nChargement du modele d'embeddings...")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )

    print("Device embeddings :", device)
    print("Calcul des embeddings...")

    names = candidate_df["candidate_name"].tolist()

    embeddings = model.encode(
        names,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=128,
    )

    k = min(EMBEDDING_NEIGHBORS + 1, len(names))

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
    )

    nn.fit(embeddings)

    distances, indices = nn.kneighbors(embeddings)

    uf = UnionFind(len(names))

    for i in range(len(names)):
        for pos in range(1, k):
            j = int(indices[i, pos])
            similarity = 1.0 - float(distances[i, pos])

            if similarity >= EMBEDDING_THRESHOLD:
                # Garde-fou : une similarite embedding seule ne suffit pas
                # pour fusionner deux noms clairement tres differents.
                fuzzy_score = fuzz.WRatio(
                    candidate_df.iloc[i]["normalized"],
                    candidate_df.iloc[j]["normalized"],
                )

                phonetic_score = phonetic_similarity(
                    candidate_df.iloc[i]["candidate_name"],
                    candidate_df.iloc[j]["candidate_name"],
                )

                if fuzzy_score >= 55 or phonetic_score >= 70:
                    uf.union(i, j)

    return uf


# ============================================================
# CONVERSION CLUSTERS -> TABLE
# ============================================================

def clusters_to_df(candidate_df, uf, method):
    groups = defaultdict(list)

    for i in range(len(candidate_df)):
        groups[uf.find(i)].append(i)

    rows = []
    cluster_counter = 1

    for members in groups.values():
        member_rows = candidate_df.iloc[members]

        # Canonique = mention la plus frequente du cluster.
        canonical_row = member_rows.sort_values(
            ["occurrences", "score_moyen"],
            ascending=[False, False],
        ).iloc[0]

        cluster_id = f"{method.upper()}_{cluster_counter:06d}"
        cluster_counter += 1

        variants = []
        candidate_ids = []

        total_occurrences = 0

        for _, row in member_rows.iterrows():
            variants.append(row["candidate_name"])
            candidate_ids.append(row["candidate_id"])
            total_occurrences += int(row["occurrences"])

        rows.append({
            "method": method,
            "cluster_id": cluster_id,
            "canonical_candidate": canonical_row["candidate_name"],
            "candidate_ids": " | ".join(candidate_ids),
            "variants": " | ".join(sorted(variants)),
            "nb_variants": len(variants),
            "occurrences": total_occurrences,
        })

    return pd.DataFrame(rows)


# ============================================================
# MATRICE D'ASSOCIATION DES CANDIDATS
# ============================================================

def build_candidate_assignments(candidate_df, cluster_tables):
    assignment = candidate_df[["candidate_id", "candidate_name", "occurrences"]].copy()

    for method, cluster_df in cluster_tables.items():
        mapping = {}

        for _, row in cluster_df.iterrows():
            for cid in row["candidate_ids"].split(" | "):
                mapping[cid] = row["cluster_id"]

        assignment[f"cluster_{method}"] = assignment["candidate_id"].map(mapping)

    return assignment


# ============================================================
# DETECTION DES DESACCORDS
# ============================================================

def build_disagreements(candidate_df, assignments):
    methods = [
        c for c in assignments.columns
        if c.startswith("cluster_")
    ]

    rows = []

    for _, row in assignments.iterrows():
        values = [row[m] for m in methods if pd.notna(row[m])]

        if len(set(values)) > 1:
            rows.append({
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                "occurrences": row["occurrences"],
                **{m: row[m] for m in methods},
            })

    return pd.DataFrame(rows)


# ============================================================
# RESUME METHODES
# ============================================================

def summarize_method(method, clusters_df, candidate_df):
    n_candidates = len(candidate_df)
    n_clusters = len(clusters_df)

    coverage_occurrences = clusters_df["occurrences"].sum()

    multi = clusters_df[clusters_df["nb_variants"] >= 2]

    merged_variants = int(
        multi["nb_variants"].sub(1).clip(lower=0).sum()
    ) if not multi.empty else 0

    max_cluster = int(
        clusters_df["nb_variants"].max()
    ) if not clusters_df.empty else 1

    return {
        "method": method,
        "candidates_initials": n_candidates,
        "clusters_finaux": n_clusters,
        "reduction_percent": round(
            100 * (1 - n_clusters / n_candidates),
            2,
        ) if n_candidates else 0,
        "clusters_avec_variantes": len(multi),
        "variantes_regroupees": merged_variants,
        "plus_gros_cluster_variantes": max_cluster,
        "occurrences_couvertes": int(coverage_occurrences),
    }


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Résolution d'entités exploratoire : normalisation exacte, fuzzy "
            "matching, comparaison phonétique et embeddings. Les regroupements "
            "produits sont des candidats, pas des entités validées."
        )
    )
    parser.add_argument(
        "--ner-results",
        default="results/sauerkraut_gliner",
        help=(
            "Dossier des résultats NER à analyser "
            "(défaut : results/sauerkraut_gliner)"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/linking",
        help="Dossier de sortie (défaut : results/linking)",
    )
    return parser.parse_args()


def main():
    global INPUT_DIR, OUTPUT_DIR

    args = parse_args()
    INPUT_DIR = Path(args.ner_results)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BENCHMARK ENTITY RESOLUTION")
    print("=" * 80)
    print("Source NER :", INPUT_DIR)
    print("Type :", ENTITY_LABEL)

    start_total = time.perf_counter()

    # --------------------------------------------------------
    # Mentions
    # --------------------------------------------------------
    df = load_mentions()

    print("\nOccurrences PERSONNE :", len(df))

    # --------------------------------------------------------
    # Candidats automatiques
    # --------------------------------------------------------
    candidates = build_candidates(df)

    print("Formes textuelles candidates :", len(candidates))

    candidates.to_csv(
        OUTPUT_DIR / "candidate_entities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Exact : chaque forme normalisee = cluster
    # --------------------------------------------------------
    print("\nConstruction exact/normalisation...")

    uf_exact = UnionFind(len(candidates))
    clusters_exact = clusters_to_df(
        candidates,
        uf_exact,
        "exact",
    )

    # --------------------------------------------------------
    # Fuzzy
    # --------------------------------------------------------
    cluster_tables = {
        "exact": clusters_exact,
    }

    summary_rows = []
    summary_rows.append(
        summarize_method("exact", clusters_exact, candidates)
    )

    for threshold in FUZZY_THRESHOLDS:
        method = f"fuzzy_{threshold}"

        print(f"\nConstruction {method}...")

        uf = build_fuzzy_clusters(
            candidates,
            threshold,
        )

        cluster_df = clusters_to_df(
            candidates,
            uf,
            method,
        )

        cluster_tables[method] = cluster_df
        summary_rows.append(
            summarize_method(
                method,
                cluster_df,
                candidates,
            )
        )

    # --------------------------------------------------------
    # Fuzzy + phonétique
    # --------------------------------------------------------
    print("\nConstruction fuzzy + phonétique...")

    uf_combined = build_combined_clusters(candidates)

    clusters_combined = clusters_to_df(
        candidates,
        uf_combined,
        "combined",
    )

    cluster_tables["combined"] = clusters_combined

    summary_rows.append(
        summarize_method(
            "combined",
            clusters_combined,
            candidates,
        )
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------
    print("\nConstruction embeddings...")

    uf_embedding = build_embedding_clusters(candidates)

    clusters_embedding = clusters_to_df(
        candidates,
        uf_embedding,
        "embedding",
    )

    cluster_tables["embedding"] = clusters_embedding

    summary_rows.append(
        summarize_method(
            "embedding",
            clusters_embedding,
            candidates,
        )
    )

    # --------------------------------------------------------
    # Export clusters
    # --------------------------------------------------------
    all_clusters = pd.concat(
        cluster_tables.values(),
        ignore_index=True,
    )

    all_clusters.to_csv(
        OUTPUT_DIR / "clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Assignation par candidat
    # --------------------------------------------------------
    assignments = build_candidate_assignments(
        candidates,
        cluster_tables,
    )

    assignments.to_csv(
        OUTPUT_DIR / "candidate_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Desaccords
    # --------------------------------------------------------
    disagreements = build_disagreements(
        candidates,
        assignments,
    )

    disagreements.to_csv(
        OUTPUT_DIR / "disagreements.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------
    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR / "linking_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Top clusters pour inspection humaine
    # --------------------------------------------------------
    review = (
        clusters_combined[
            clusters_combined["nb_variants"] >= 2
        ]
        .sort_values(
            ["nb_variants", "occurrences"],
            ascending=[False, False],
        )
        .head(1000)
    )

    review.to_csv(
        OUTPUT_DIR / "review_top_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = time.perf_counter() - start_total

    # --------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTATS")
    print("=" * 80)

    print(summary.to_string(index=False))

    print("\nDesaccords entre methodes :", len(disagreements))
    print("Temps total :", round(elapsed, 2), "secondes")

    print("\nFichiers generes :")
    print(OUTPUT_DIR / "candidate_entities.csv")
    print(OUTPUT_DIR / "clusters.csv")
    print(OUTPUT_DIR / "candidate_assignments.csv")
    print(OUTPUT_DIR / "disagreements.csv")
    print(OUTPUT_DIR / "linking_summary.csv")
    print(OUTPUT_DIR / "review_top_clusters.csv")


if __name__ == "__main__":
    main()
