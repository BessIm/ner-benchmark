import argparse
import time
import torch

from gliner import GLiNER

from common import (
    get_json_files,
    load_text,
    save_result,
    clean_entities,
)


MODEL_NAME = "VAGOsolutions/SauerkrautLM-GLiNER"
OUTPUT_NAME = "sauerkraut_gliner"

LABELS = [
    "person",
    "organization",
    "location",
    "event",
]

LABEL_MAP = {
    "person": "PERSONNE",
    "organization": "ORGANISATION",
    "location": "LIEU",
    "event": "EVENEMENT",
}

THRESHOLD = 0.50

# GLiNER tronque au-delà de 384 tokens : 1500 caractères restent sous la limite.
CHUNK_SIZE = 1500
OVERLAP = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NER : SauerkrautLM-GLiNER (VAGOsolutions/SauerkrautLM-GLiNER)"
    )
    parser.add_argument(
        "--input",
        default="transcriptions",
        help="Dossier des transcriptions JSON (défaut : transcriptions)",
    )
    parser.add_argument(
        "--output",
        default=f"results/{OUTPUT_NAME}",
        help=f"Dossier de sortie (défaut : results/{OUTPUT_NAME})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Seuil de confiance GLiNER (défaut : {THRESHOLD})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Taille des fenêtres de découpage (défaut : {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help=f"Chevauchement entre fenêtres (défaut : {OVERLAP})",
    )
    return parser.parse_args()


def chunks(text, chunk_size, overlap):
    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        yield start, text[start:end]

        if end >= len(text):
            break

        start = end - overlap


def main():
    args = parse_args()

    print("=" * 70)
    print("SauerkrautLM-GLiNER")
    print("=" * 70)
    print("Modèle :", MODEL_NAME)

    if torch.cuda.is_available():
        device = "cuda"

        print("GPU :", torch.cuda.get_device_name(0))

        print(
            "VRAM :",
            round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3,
                1
            ),
            "GB"
        )
    else:
        device = "cpu"
        print("GPU non disponible")

    print("\nChargement du modèle...")

    model = GLiNER.from_pretrained(
        MODEL_NAME,
    )

    model = model.to(device)
    model.eval()

    print("Modèle chargé.")

    for path in get_json_files(args.input):

        print("\n" + "=" * 70)
        print(path.name)

        text = load_text(path)

        print("Longueur :", len(text), "caractères")

        start_time = time.perf_counter()

        entities = []

        for offset, chunk in chunks(text, args.chunk_size, args.overlap):

            results = model.predict_entities(
                chunk,
                LABELS,
                threshold=args.threshold,
            )

            for entity in results:

                start = offset + entity["start"]
                end = offset + entity["end"]

                label = entity["label"].lower()

                entities.append({
                    "text": text[start:end],
                    "label": LABEL_MAP.get(
                        label,
                        label.upper()
                    ),
                    "score": float(entity["score"]),
                    "start": start,
                    "end": end,
                })

        entities = clean_entities(entities)

        elapsed = time.perf_counter() - start_time

        save_result(
            OUTPUT_NAME,
            path.name,
            entities,
            elapsed,
            output_dir=args.output,
        )

        print("Entités :", len(entities))
        print("Temps :", round(elapsed, 2), "s")


if __name__ == "__main__":
    main()
