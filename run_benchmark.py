#!/usr/bin/env python3
"""
Lance successivement les modèles NER, puis compare.py, et propose de lancer
la résolution d'entités (benchmark_linking_final.py).

Exemples :
    python run_benchmark.py
    python run_benchmark.py --models sauerkraut gliner
    python run_benchmark.py --linking yes
    python run_benchmark.py --force

Sécurité :
- si des résultats existent déjà pour un modèle, le script demande s'il faut
  les écraser (sauf avec --force) ;
- lorsqu'un écrasement est accepté, le dossier de sortie concerné est vidé
  avant la ré-exécution, afin de ne pas conserver de fichiers obsolètes ;
- l'option --linking contrôle le lancement de la résolution d'entités
  (ask par défaut).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Nom court -> script exécutable + sous-dossier de résultats.
MODELS = {
    "camembert": {
        "script": "src/run_camembert.py",
        "subdir": "camembert",
    },
    "moderncamembert": {
        "script": "src/run_moderncamembert.py",
        "subdir": "moderncamembert",
    },
    "gliner": {
        "script": "src/run_gliner.py",
        "subdir": "gliner",
    },
    "gliner_large": {
        "script": "src/run_gliner_large.py",
        "subdir": "gliner_large_v25",
    },
    "sauerkraut": {
        "script": "src/run_sauerkraut.py",
        "subdir": "sauerkraut_gliner",
    },
}

# Alias acceptés (notamment les noms de dossiers de résultats).
ALIASES = {
    "sauerkraut_gliner": "sauerkraut",
    "gliner_large_v25": "gliner_large",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lance les modèles NER du benchmark, puis compare.py, et propose "
            "de lancer la résolution d'entités."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Modèles à exécuter (défaut : tous). "
            f"Choix possibles : {', '.join(MODELS)}"
        ),
    )
    parser.add_argument(
        "--input",
        default="transcriptions",
        help="Dossier des transcriptions JSON (défaut : transcriptions)",
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Dossier racine des résultats (défaut : results)",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Ne pas lancer compare.py à la fin.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Écraser les résultats existants sans demander confirmation "
            "(le dossier concerné est vidé avant ré-exécution)."
        ),
    )
    parser.add_argument(
        "--linking",
        choices=["ask", "yes", "no"],
        default="ask",
        help=(
            "Lancer la résolution d'entités après compare.py "
            "(ask = demander, yes = toujours, no = jamais ; défaut : ask)"
        ),
    )
    parser.add_argument(
        "--linking-ner",
        default=None,
        help=(
            "Dossier de résultats NER à utiliser pour la résolution d'entités "
            "(défaut : results/sauerkraut_gliner si présent, sinon le premier "
            "dossier disponible)"
        ),
    )
    return parser.parse_args()


def confirm(question, default=False):
    """Pose une question oui/non. Hors terminal interactif, renvoie `default`."""
    if not sys.stdin.isatty():
        return default

    suffix = " [O/n] " if default else " [o/N] "

    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        return default

    if not answer:
        return default

    return answer in ("o", "oui", "y", "yes")


def run_command(command, label):
    result = subprocess.run(command)

    if result.returncode != 0:
        raise SystemExit(
            f"\nArrêt : {label} a échoué (code {result.returncode})."
        )


def resolve_models(names):
    resolved = []

    for name in names:
        key = ALIASES.get(name, name)

        if key not in MODELS:
            raise SystemExit(
                f"Modèle inconnu : {name}. "
                f"Choix possibles : {', '.join(MODELS)}"
            )

        resolved.append(key)

    return resolved


def prepare_output_dir(output_dir, label, force):
    """Décide s'il faut (ré)exécuter et vide le dossier si écrasement accepté.

    Retourne True s'il faut exécuter, False pour ignorer.
    """
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return True

    if not force and not confirm(
        f"[{label}] des résultats existent déjà dans {output_dir}. "
        f"Les écraser ?",
        default=False,
    ):
        print(f"[{label}] conservé (non ré-exécuté).\n")
        return False

    print(f"[{label}] suppression des anciens résultats dans {output_dir} ...")
    shutil.rmtree(output_dir)
    return True


def find_linking_ner(args, results_dir, models):
    """Choisit le dossier de résultats NER utilisé pour la résolution."""
    if args.linking_ner:
        return Path(args.linking_ner)

    preferred = results_dir / MODELS["sauerkraut"]["subdir"]
    if preferred.exists() and any(preferred.glob("*.json")):
        return preferred

    for name in models:
        candidate = results_dir / MODELS[name]["subdir"]
        if candidate.exists() and any(candidate.glob("*.json")):
            return candidate

    return None


def main():
    args = parse_args()

    models = resolve_models(args.models) if args.models else list(MODELS)
    results_dir = Path(args.results)

    print("=" * 70)
    print("BENCHMARK NER - LANCEMENT DES MODELES")
    print("=" * 70)
    print("Modèles :", ", ".join(models))
    print("Transcriptions :", args.input)
    print("Résultats :", results_dir)
    print()

    for name in models:
        spec = MODELS[name]
        output_dir = results_dir / spec["subdir"]

        if not prepare_output_dir(output_dir, name, args.force):
            continue

        print(f"[{name}] exécution de {spec['script']} ...")

        command = [
            sys.executable,
            spec["script"],
            "--input", args.input,
            "--output", str(output_dir),
        ]

        run_command(command, f"le modèle '{name}'")

        print(f"[{name}] terminé.\n")

    if args.skip_compare:
        print("compare.py ignoré (--skip-compare).")
    else:
        print("=" * 70)
        print("COMPARAISON DES RESULTATS (compare.py)")
        print("=" * 70)

        run_command(
            [
                sys.executable,
                "src/compare.py",
                "--results", str(results_dir),
            ],
            "compare.py",
        )

    # ------------------------------------------------------------
    # Résolution d'entités
    # ------------------------------------------------------------
    if args.linking == "no":
        should_run_linking = False
    elif args.linking == "yes":
        should_run_linking = True
    else:
        should_run_linking = confirm(
            "\nLancer la résolution d'entités (benchmark_linking_final.py) ?",
            default=True,
        )

    if not should_run_linking:
        print("Résolution d'entités ignorée.")
        return

    ner_dir = find_linking_ner(args, results_dir, models)

    if ner_dir is None:
        print("Aucun résultat NER trouvé : résolution d'entités ignorée.")
        return

    linking_output = results_dir / "linking_final"

    if not prepare_output_dir(linking_output, "linking_final", args.force):
        return

    print("=" * 70)
    print("RESOLUTION D'ENTITES (benchmark_linking_final.py)")
    print("=" * 70)
    print("Source NER :", ner_dir)

    run_command(
        [
            sys.executable,
            "src/benchmark_linking_final.py",
            "--ner-results", str(ner_dir),
            "--transcriptions", args.input,
            "--output", str(linking_output),
        ],
        "benchmark_linking_final.py",
    )

    print("\nBenchmark terminé.")


if __name__ == "__main__":
    main()

