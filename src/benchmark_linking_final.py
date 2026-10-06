#!/usr/bin/env python3
"""
Final exploratory benchmark for entity resolution on PERSON entities.

RhoneFM speech-to-text transcriptions are the case study used for the
experiments, but this script is independent of any specific corpus or
authority file.

Input:
  results/sauerkraut_gliner/*.json
  transcriptions/*.json

The script uses only the `text` field from the original transcription JSONs.
It does NOT require a reference_entities.csv / IdRef export.

It compares candidate generation / ranking strategies:
  1) fuzzy name matching only
  2) fuzzy + phonetic matching
  3) fuzzy + phonetic + context embeddings

The embedding model is BAAI/bge-m3 (multilingual, up to 8192 tokens).
No automatic merge is treated as truth: the output is a candidate proposal
set for later cataloguer validation.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz, process
from metaphone import doublemetaphone
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

NER_DIR = Path("results/sauerkraut_gliner")
TRANSCRIPTIONS_DIR = Path("transcriptions")
OUTPUT_DIR = Path("results/linking_final")

ENTITY_TYPE = "PERSONNE"

# Candidate generation
TOP_K = 5
MIN_FUZZY_FOR_CANDIDATE = 75

# Context extraction: +/- characters around the mention occurrence.
CONTEXT_CHARS = 500
MAX_CONTEXTS_PER_FORM = 3

# Strict proposal threshold for the optional graph/cluster output.
# These are deliberately conservative because there is no gold standard yet.
CLUSTER_MIN_NAME = 90.0
CLUSTER_MIN_PHONETIC = 75.0
CLUSTER_MIN_CONTEXT = 0.70
CLUSTER_MIN_COMBINED = 82.0

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 64


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).casefold().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Civilities / common speech prefixes.
    text = re.sub(
        r"^(m\.?|mr\.?|monsieur|mme\.?|madame|dr\.?|docteur)\s+",
        "",
        text,
    )

    return text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# ============================================================
# PHONETICS
# ============================================================

def phonetic_code(text: str) -> tuple[str, str]:
    words = normalize_text(text).replace("-", " ").split()
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

    return float(max(scores)) if scores else 0.0


# ============================================================
# INPUT LOADING
# ============================================================

def find_transcription(path_name: str) -> Path | None:
    direct = TRANSCRIPTIONS_DIR / path_name
    if direct.exists():
        return direct

    # Fallback by stem, in case the NER result filename was transformed.
    stem = Path(path_name).stem
    matches = list(TRANSCRIPTIONS_DIR.glob(f"{stem}.json"))
    return matches[0] if matches else None


def load_transcription_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError(f"Champ text absent ou invalide: {path}")
    return text


def load_mentions() -> list[dict]:
    rows = []
    result_files = sorted(NER_DIR.glob("*.json"))

    if not result_files:
        raise FileNotFoundError(
            f"Aucun résultat NER trouvé dans {NER_DIR}"
        )

    cache: dict[Path, str] = {}

    for result_path in result_files:
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        filename = result_path.name
        transcription_path = find_transcription(filename)
        if transcription_path is None:
            print(f"ATTENTION: transcription introuvable pour {filename}")
            continue

        if transcription_path not in cache:
            cache[transcription_path] = load_transcription_text(transcription_path)

        full_text = cache[transcription_path]

        for entity in data.get("entities", []):
            if str(entity.get("label", "")).upper() != ENTITY_TYPE:
                continue

            raw = compact_text(entity.get("text", ""))
            if not raw:
                continue

            start = entity.get("start")
            end = entity.get("end")

            if not isinstance(start, int) or not isinstance(end, int):
                continue

            left = max(0, start - CONTEXT_CHARS)
            right = min(len(full_text), end + CONTEXT_CHARS)
            context = compact_text(full_text[left:right])

            rows.append({
                "file": filename,
                "text": raw,
                "normalized": normalize_text(raw),
                "score": float(entity.get("score", 0.0)),
                "start": start,
                "end": end,
                "context": context,
            })

    return rows


# ============================================================
# BUILD UNIQUE FORMS + REPRESENTATIVE CONTEXTS
# ============================================================

def build_form_table(mentions: list[dict]) -> pd.DataFrame:
    grouped: dict[str, dict] = {}

    for row in mentions:
        key = row["normalized"]
        if not key:
            continue

        if key not in grouped:
            grouped[key] = {
                "normalized": key,
                "display": row["text"],
                "occurrences": 0,
                "contexts": [],
            }

        grouped[key]["occurrences"] += 1

        # Keep up to N distinct contexts per form.
        ctx = row["context"]
        if ctx and ctx not in grouped[key]["contexts"]:
            grouped[key]["contexts"].append(ctx)
            grouped[key]["contexts"] = grouped[key]["contexts"][:MAX_CONTEXTS_PER_FORM]

    records = []
    for record in grouped.values():
        records.append({
            "normalized": record["normalized"],
            "display": record["display"],
            "occurrences": record["occurrences"],
            "contexts": record["contexts"],
        })

    return pd.DataFrame(records)


# ============================================================
# EMBEDDINGS
# ============================================================

def build_embedding_inputs(forms: pd.DataFrame) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    owners: list[int] = []

    for idx, row in forms.iterrows():
        contexts = row["contexts"]
        if not contexts:
            texts.append(row["display"])
            owners.append(int(idx))
            continue

        for context in contexts:
            texts.append(context)
            owners.append(int(idx))

    return texts, owners


def load_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement embeddings: {EMBEDDING_MODEL} sur {device}")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )

    return model


def encode_forms(model: SentenceTransformer, forms: pd.DataFrame) -> dict[int, np.ndarray]:
    texts, owners = build_embedding_inputs(forms)

    print(f"Encodage de {len(texts)} contextes...")

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    buckets: dict[int, list[np.ndarray]] = defaultdict(list)
    for emb, owner in zip(embeddings, owners):
        buckets[owner].append(emb)

    # One representative vector per form = mean of context vectors, re-normalized.
    form_embeddings: dict[int, np.ndarray] = {}
    for owner, vectors in buckets.items():
        vec = np.mean(np.vstack(vectors), axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        form_embeddings[owner] = vec

    return form_embeddings


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(forms: pd.DataFrame) -> dict[int, list[int]]:
    choices = forms["normalized"].tolist()
    candidates: dict[int, list[int]] = {}

    for idx, query in enumerate(choices):
        extracted = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=TOP_K + 1,
            score_cutoff=MIN_FUZZY_FOR_CANDIDATE,
        )

        ids = []
        for _, score, match_idx in extracted:
            if match_idx == idx:
                continue
            ids.append(int(match_idx))

        candidates[idx] = ids[:TOP_K]

    return candidates


# ============================================================
# CONTEXT SIMILARITY
# ============================================================

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ============================================================
# SCORE / RANKING
# ============================================================

def combined_score(name_score: float, phonetic_score: float, context_score: float) -> float:
    # Deliberately conservative: spelling remains primary signal,
    # phonetics captures STT errors, context helps disambiguate.
    return (
        0.50 * name_score
        + 0.20 * phonetic_score
        + 0.30 * (context_score * 100.0)
    )


def build_candidate_rows(
    forms: pd.DataFrame,
    candidates: dict[int, list[int]],
    embeddings: dict[int, np.ndarray],
) -> list[dict]:
    rows = []

    for source_idx, target_ids in candidates.items():
        source_name = forms.iloc[source_idx]["display"]

        for rank_name, target_idx in enumerate(target_ids, start=1):
            target_name = forms.iloc[target_idx]["display"]

            name_score = float(
                fuzz.WRatio(
                    forms.iloc[source_idx]["normalized"],
                    forms.iloc[target_idx]["normalized"],
                )
            )

            phon_score = phonetic_similarity(
                source_name,
                target_name,
            )

            if source_idx in embeddings and target_idx in embeddings:
                context_score = cosine(
                    embeddings[source_idx],
                    embeddings[target_idx],
                )
            else:
                context_score = 0.0

            combined = combined_score(
                name_score,
                phon_score,
                context_score,
            )

            rows.append({
                "source_id": source_idx,
                "source": source_name,
                "source_occurrences": int(forms.iloc[source_idx]["occurrences"]),
                "candidate_id": target_idx,
                "candidate": target_name,
                "candidate_occurrences": int(forms.iloc[target_idx]["occurrences"]),
                "fuzzy_name": name_score,
                "phonetic": phon_score,
                "context_cosine": context_score,
                "combined": combined,
                "fuzzy_rank": rank_name,
            })

    return rows


# ============================================================
# METHOD COMPARISON
# ============================================================

def make_rankings(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_df.empty:
        return candidate_df.copy(), pd.DataFrame()

    ranked = []

    for source_id, group in candidate_df.groupby("source_id"):
        group = group.copy()

        for method, score_col in [
            ("fuzzy", "fuzzy_name"),
            ("fuzzy_phonetic", "phonetic"),
            ("combined_context", "combined"),
        ]:
            tmp = group.sort_values(
                [score_col, "fuzzy_name"],
                ascending=False,
            ).copy()
            tmp["method"] = method
            tmp["method_rank"] = range(1, len(tmp) + 1)
            ranked.append(tmp)

    ranking_df = pd.concat(ranked, ignore_index=True)

    top = (
        ranking_df[ranking_df["method_rank"] == 1]
        .copy()
        [[
            "source_id",
            "source",
            "candidate_id",
            "candidate",
            "method",
            "fuzzy_name",
            "phonetic",
            "context_cosine",
            "combined",
        ]]
    )

    pivot = top.pivot_table(
        index=["source_id", "source"],
        columns="method",
        values="candidate",
        aggfunc="first",
    ).reset_index()

    return ranking_df, pivot


# ============================================================
# CONSERVATIVE CLUSTERS
# ============================================================

def build_conservative_clusters(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build candidate clusters only from strong pairwise edges.

    NOTE: These clusters are hypotheses, not truth.
    """
    if candidate_df.empty:
        return pd.DataFrame()

    # Union-Find
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, row in candidate_df.iterrows():
        if (
            row["fuzzy_name"] >= CLUSTER_MIN_NAME
            and row["phonetic"] >= CLUSTER_MIN_PHONETIC
            and row["context_cosine"] >= CLUSTER_MIN_CONTEXT
            and row["combined"] >= CLUSTER_MIN_COMBINED
        ):
            union(int(row["source_id"]), int(row["candidate_id"]))

    groups = defaultdict(list)
    for node in list(parent):
        groups[find(node)].append(node)

    records = []
    for cluster_id, members in enumerate(groups.values(), start=1):
        if len(members) < 2:
            continue
        for member in members:
            records.append({
                "cluster_id": cluster_id,
                "form_id": member,
            })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark final de résolution d'entités : génération de candidats "
            "fuzzy, fuzzy + phonétique, puis fuzzy + phonétique + contexte "
            "(embeddings). Les regroupements sont des propositions à valider."
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
        "--transcriptions",
        default="transcriptions",
        help="Dossier des transcriptions JSON (défaut : transcriptions)",
    )
    parser.add_argument(
        "--output",
        default="results/linking_final",
        help="Dossier de sortie (défaut : results/linking_final)",
    )
    return parser.parse_args()


def main() -> None:
    global NER_DIR, TRANSCRIPTIONS_DIR, OUTPUT_DIR

    args = parse_args()
    NER_DIR = Path(args.ner_results)
    TRANSCRIPTIONS_DIR = Path(args.transcriptions)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BENCHMARK FINAL - ENTITY RESOLUTION")
    print("=" * 80)

    started = time.perf_counter()

    print("\n1) Chargement des mentions PERSONNE")
    mentions = load_mentions()
    print(f"Occurrences chargées : {len(mentions):,}")

    forms = build_form_table(mentions)
    print(f"Formes textuelles uniques : {len(forms):,}")

    forms_path = OUTPUT_DIR / "forms.csv"
    export_forms = forms.copy()
    export_forms["contexts"] = export_forms["contexts"].map(
        lambda xs: " || ".join(xs)
    )
    export_forms.to_csv(
        forms_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n2) Génération des candidats fuzzy")
    candidates = generate_candidates(forms)
    candidate_count = sum(len(v) for v in candidates.values())
    print(f"Paires candidates : {candidate_count:,}")

    print("\n3) Embeddings des contextes")
    embed_model = load_embedding_model()
    form_embeddings = encode_forms(embed_model, forms)

    print("\n4) Calcul des scores candidats")
    candidate_rows = build_candidate_rows(
        forms,
        candidates,
        form_embeddings,
    )
    candidate_df = pd.DataFrame(candidate_rows)

    candidate_path = OUTPUT_DIR / "candidate_suggestions.csv"
    candidate_df.to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n5) Comparaison des stratégies")
    ranking_df, top_pivot = make_rankings(candidate_df)

    ranking_path = OUTPUT_DIR / "method_rankings.csv"
    ranking_df.to_csv(
        ranking_path,
        index=False,
        encoding="utf-8-sig",
    )

    pivot_path = OUTPUT_DIR / "method_top_choices.csv"
    top_pivot.to_csv(
        pivot_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Agreement between top candidates.
    agreement_rows = []
    for _, row in top_pivot.iterrows():
        values = [
            row.get("fuzzy"),
            row.get("fuzzy_phonetic"),
            row.get("combined_context"),
        ]
        values = [v for v in values if pd.notna(v)]
        agreement_rows.append(
            len(values) >= 2 and len(set(values)) == 1
        )

    agreement = float(np.mean(agreement_rows)) if agreement_rows else 0.0

    print("\n6) Clusters très conservateurs")
    cluster_df = build_conservative_clusters(candidate_df)
    cluster_path = OUTPUT_DIR / "conservative_clusters.csv"
    cluster_df.to_csv(
        cluster_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Cluster summary
    if not cluster_df.empty:
        cluster_summary = (
            cluster_df.groupby("cluster_id")
            .agg(
                n_forms=("form_id", "count"),
            )
            .reset_index()
            .sort_values("n_forms", ascending=False)
        )
    else:
        cluster_summary = pd.DataFrame(
            columns=["cluster_id", "n_forms"]
        )

    cluster_summary_path = OUTPUT_DIR / "cluster_summary.csv"
    cluster_summary.to_csv(
        cluster_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    elapsed = time.perf_counter() - started

    summary = pd.DataFrame([
        {
            "occurrences_personne": len(mentions),
            "formes_uniques": len(forms),
            "paires_candidates": len(candidate_df),
            "methodes_comparees": "fuzzy | fuzzy+phonetic | fuzzy+phonetic+context",
            "accord_top_fuzzy_vs_phonetic_context": agreement,
            "clusters_conservateurs": int(cluster_summary.shape[0]),
            "plus_grand_cluster": int(cluster_summary["n_forms"].max()) if not cluster_summary.empty else 0,
            "temps_total_secondes": elapsed,
            "embedding_model": EMBEDDING_MODEL,
        }
    ])

    summary_path = OUTPUT_DIR / "linking_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("RESULTAT")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nFichiers générés :")
    for path in [
        forms_path,
        candidate_path,
        ranking_path,
        pivot_path,
        cluster_path,
        cluster_summary_path,
        summary_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
