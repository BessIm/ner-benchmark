import argparse
import time
import torch
from transformers import pipeline, AutoTokenizer

from common import (
    get_json_files,
    load_text,
    save_result,
    clean_entities,
    normalize_label,
)


MODEL_NAME = "Jean-Baptiste/camembert-ner"
OUTPUT_NAME = "camembert"

CHUNK_SIZE = 5000
OVERLAP = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NER : CamemBERT (Jean-Baptiste/camembert-ner)"
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
        end = min(start + chunk_size, len(text))

        yield start, text[start:end]

        if end == len(text):
            break

        start = end - overlap


def main():
    args = parse_args()

    device = 0 if torch.cuda.is_available() else -1

    print("Chargement :", MODEL_NAME)
    print("Device :", "CUDA" if device == 0 else "CPU")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    ner = pipeline(
        "ner",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )

    for path in get_json_files(args.input):

        print("\n================================")
        print(path.name)

        text = load_text(path)

        start_time = time.perf_counter()

        entities = []

        for offset, chunk in chunks(text, args.chunk_size, args.overlap):

            results = ner(chunk)

            for entity in results:

                start = offset + entity["start"]
                end = offset + entity["end"]

                entities.append({
                    "text": text[start:end],
                    "label": normalize_label(entity["entity_group"]),
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
        print("Temps :", round(elapsed, 2), "secondes")


if __name__ == "__main__":
    main()
