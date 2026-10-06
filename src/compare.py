import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODELS = [
    "camembert",
    "moderncamembert",
    "gliner",
    "gliner_large_v25",
    "sauerkraut_gliner",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare les résultats NER des différents modèles et exporte "
            "comparison.csv, entities.csv et summary.csv."
        )
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Dossier racine des résultats (défaut : results)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Sous-ensemble de modèles à comparer (défaut : tous)",
    )
    return parser.parse_args()


ARGS = parse_args()

RESULTS_DIR = Path(ARGS.results)

MODELS = ARGS.models or DEFAULT_MODELS


# ============================================================
# VARIABLES
# ============================================================

stats = []
all_entities = []


# ============================================================
# LECTURE DES RESULTATS
# ============================================================

for model in MODELS:

    model_dir = RESULTS_DIR / model

    if not model_dir.exists():
        print(f"Résultats absents : {model}")
        continue

    json_files = sorted(model_dir.glob("*.json"))

    print(f"{model}: {len(json_files)} fichier(s)")

    for path in json_files:

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities = data.get("entities", [])

        # ----------------------------------------------------
        # Comptage des occurrences
        # ----------------------------------------------------

        counts = Counter(
            entity.get("label", "").upper()
            for entity in entities
        )

        # ----------------------------------------------------
        # Mentions textuelles uniques
        # ----------------------------------------------------
        #
        # On normalise uniquement :
        # - espaces début/fin
        # - majuscules/minuscules
        #
        # "Marie Dupont" et "marie dupont"
        # sont donc considérés comme la même mention.
        #
        # En revanche :
        # "Marie Dupont"
        # "Marie Dupond"
        # restent deux mentions différentes.
        # ----------------------------------------------------

        unique_mentions = {
            entity.get("text", "").strip().casefold()
            for entity in entities
            if entity.get("text", "").strip()
        }

        # ----------------------------------------------------
        # Mentions uniques par type
        # ----------------------------------------------------

        unique_personnes = {
            entity.get("text", "").strip().casefold()
            for entity in entities
            if entity.get("label", "").upper() == "PERSONNE"
            and entity.get("text", "").strip()
        }

        unique_organisations = {
            entity.get("text", "").strip().casefold()
            for entity in entities
            if entity.get("label", "").upper() == "ORGANISATION"
            and entity.get("text", "").strip()
        }

        unique_lieux = {
            entity.get("text", "").strip().casefold()
            for entity in entities
            if entity.get("label", "").upper() == "LIEU"
            and entity.get("text", "").strip()
        }

        unique_evenements = {
            entity.get("text", "").strip().casefold()
            for entity in entities
            if entity.get("label", "").upper() == "EVENEMENT"
            and entity.get("text", "").strip()
        }

        unique_misc = {
            entity.get("text", "").strip().casefold()
            for entity in entities
            if entity.get("label", "").upper() == "MISC"
            and entity.get("text", "").strip()
        }

        # ----------------------------------------------------
        # Statistiques
        # ----------------------------------------------------

        stats.append({
            "model": model,
            "file": path.name,

            # Occurrences
            "total": len(entities),
            "personnes": counts.get("PERSONNE", 0),
            "organisations": counts.get("ORGANISATION", 0),
            "lieux": counts.get("LIEU", 0),
            "evenements": counts.get("EVENEMENT", 0),
            "misc": counts.get("MISC", 0),

            # Mentions textuelles uniques
            "unique_total": len(unique_mentions),
            "unique_personnes": len(unique_personnes),
            "unique_organisations": len(unique_organisations),
            "unique_lieux": len(unique_lieux),
            "unique_evenements": len(unique_evenements),
            "unique_misc": len(unique_misc),

            # Temps
            "elapsed_seconds": float(
                data.get("elapsed_seconds", 0)
            ),

            # Taille du texte
            "text_length": data.get("text_length"),
        })

        # ----------------------------------------------------
        # Export de toutes les occurrences
        # ----------------------------------------------------

        for entity in entities:

            all_entities.append({
                "model": model,
                "file": path.name,
                "text": entity.get("text", ""),
                "label": entity.get("label", ""),
                "score": entity.get("score", 0),
                "start": entity.get("start"),
                "end": entity.get("end"),
            })


# ============================================================
# DATAFRAMES
# ============================================================

df_stats = pd.DataFrame(stats)
df_entities = pd.DataFrame(all_entities)


if df_stats.empty:
    print("\nAucun résultat trouvé.")
    exit(1)


# ============================================================
# EXPORT 1 : COMPARAISON PAR FICHIER
# ============================================================

comparison_path = RESULTS_DIR / "comparison.csv"

df_stats.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# EXPORT 2 : TOUTES LES ENTITES
# ============================================================

entities_path = RESULTS_DIR / "entities.csv"

df_entities.to_csv(
    entities_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# AGREGATION PAR MODELE
# ============================================================

aggregated = (
    df_stats
    .groupby("model")
    .agg({
        "file": "count",

        # Occurrences
        "total": "sum",
        "personnes": "sum",
        "organisations": "sum",
        "lieux": "sum",
        "evenements": "sum",
        "misc": "sum",

        # Mentions uniques
        "unique_total": "sum",
        "unique_personnes": "sum",
        "unique_organisations": "sum",
        "unique_lieux": "sum",
        "unique_evenements": "sum",
        "unique_misc": "sum",

        # Temps
        "elapsed_seconds": "sum",
    })
    .rename(columns={
        "file": "nb_transcriptions",
    })
)


# ============================================================
# RATIOS OCCURRENCES / MENTIONS UNIQUES
# ============================================================

aggregated["ratio_repetition_total"] = (
    aggregated["total"] /
    aggregated["unique_total"]
)

aggregated["ratio_repetition_personnes"] = (
    aggregated["personnes"] /
    aggregated["unique_personnes"]
)

aggregated["ratio_repetition_organisations"] = (
    aggregated["organisations"] /
    aggregated["unique_organisations"]
)

aggregated["ratio_repetition_lieux"] = (
    aggregated["lieux"] /
    aggregated["unique_lieux"]
)

aggregated["ratio_repetition_evenements"] = (
    aggregated["evenements"] /
    aggregated["unique_evenements"].replace(0, pd.NA)
)


# ============================================================
# SCORE MOYEN GLOBAL
# ============================================================

score_mean = (
    df_entities
    .groupby("model")["score"]
    .mean()
    .rename("score_moyen")
)

score_median = (
    df_entities
    .groupby("model")["score"]
    .median()
    .rename("score_median")
)

aggregated = aggregated.join(score_mean)
aggregated = aggregated.join(score_median)


# ============================================================
# NOMBRE D'ENTITES COURTES
# ============================================================

df_entities["text_clean"] = (
    df_entities["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df_entities["text_length"] = (
    df_entities["text_clean"]
    .str.len()
)


short_entities = (
    df_entities[df_entities["text_length"] <= 2]
    .groupby("model")
    .size()
    .rename("entites_1_2_caracteres")
)

aggregated = aggregated.join(short_entities)

aggregated["entites_1_2_caracteres"] = (
    aggregated["entites_1_2_caracteres"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# EXPORT 3 : SYNTHESE PAR MODELE
# ============================================================

summary_path = RESULTS_DIR / "summary.csv"

aggregated.to_csv(
    summary_path,
    encoding="utf-8-sig",
)


# ============================================================
# AFFICHAGE
# ============================================================

print()
print("=" * 100)
print("COMPARAISON DES MODELES")
print("=" * 100)

display_columns = [
    "nb_transcriptions",

    "total",
    "unique_total",

    "personnes",
    "unique_personnes",

    "organisations",
    "unique_organisations",

    "lieux",
    "unique_lieux",

    "evenements",
    "unique_evenements",

    "elapsed_seconds",

    "score_moyen",
    "score_median",

    "entites_1_2_caracteres",
]

print(
    aggregated[display_columns]
    .round(3)
    .to_string()
)


# ============================================================
# RAPPORT COMPLEMENTAIRE
# ============================================================

print()
print("=" * 100)
print("RATIO OCCURRENCES / MENTIONS UNIQUES")
print("=" * 100)

ratio_columns = [
    "ratio_repetition_total",
    "ratio_repetition_personnes",
    "ratio_repetition_organisations",
    "ratio_repetition_lieux",
    "ratio_repetition_evenements",
]

print(
    aggregated[ratio_columns]
    .round(2)
    .to_string()
)


# ============================================================
# FICHIERS GENERES
# ============================================================

print()
print("=" * 100)
print("FICHIERS GENERES")
print("=" * 100)

print(comparison_path)
print(entities_path)
print(summary_path)
