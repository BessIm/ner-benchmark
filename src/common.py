import json
import os
import re
from pathlib import Path


TRANSCRIPTIONS_DIR = Path("transcriptions")
RESULTS_DIR = Path("results")


LABEL_MAP = {
    "PER": "PERSONNE",
    "PERSON": "PERSONNE",
    "PERSONNE": "PERSONNE",

    "ORG": "ORGANISATION",
    "ORGANIZATION": "ORGANISATION",
    "ORGANISATION": "ORGANISATION",

    "LOC": "LIEU",
    "LOCATION": "LIEU",
    "LIEU": "LIEU",

    "MISC": "MISC",
    "EVENT": "EVENEMENT",
    "EVENEMENT": "EVENEMENT",
}


def get_json_files(directory=None):
    """Retourne les fichiers JSON d'un dossier (transcriptions par défaut)."""
    directory = Path(directory) if directory else TRANSCRIPTIONS_DIR
    return sorted(directory.glob("*.json"))


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Pas de champ text valide dans {path}")

    return text


def normalize_label(label):
    label = label.upper()

    # Transformers produit parfois B-PER, I-PER, etc.
    label = re.sub(r"^[BI]-", "", label)

    return LABEL_MAP.get(label, label)


def save_result(model_name, filename, entities, elapsed, output_dir=None):
    output = {
        "file": filename,
        "model": model_name,
        "elapsed_seconds": elapsed,
        "text_length": None,
        "entities": entities,
    }

    if output_dir is None:
        output_dir = RESULTS_DIR / model_name

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def clean_entities(entities):
    """
    Supprime les doublons exacts pouvant apparaître
    à cause du découpage en fenêtres.
    """

    unique = {}

    for entity in entities:
        key = (
            entity["start"],
            entity["end"],
            entity["label"],
            entity["text"].lower(),
        )

        if key not in unique:
            unique[key] = entity
        else:
            # On conserve le score le plus élevé.
            if entity["score"] > unique[key]["score"]:
                unique[key] = entity

    return sorted(
        unique.values(),
        key=lambda x: (x["start"], x["end"])
    )
