# Benchmark NER et résolution d'entités

Comparaison de méthodes d'**extraction d'entités nommées** (NER) et de
**rapprochement de mentions** sur des transcriptions issues de la
reconnaissance de la parole (speech-to-text).

Le corpus d'origine est un ensemble de transcriptions utilisé comme **cas d'étude**. Le dépôt est **générique** : il peut être
appliqué à n'importe quel corpus de transcriptions, et l'entité validée peut
ensuite être reliée à n'importe quel référentiel d'autorité (IdRef, Wikidata,
VIAF, référentiel interne ou métier, ...).

---

## 1. Présentation

Ce projet répond à deux questions :

1. **NER** — quelles entités (personnes, organisations, lieux, événements)
   sont mentionnées dans une transcription, et où ?
2. **Résolution d'entités** — plusieurs mentions différentes
   (`Marie Dupont`, `Marie Dupond`, `M. Dupont`) désignent-elles la
   même entité réelle ?

Il compare plusieurs modèles NER et plusieurs stratégies de rapprochement
(comparaison exacte, fuzzy, phonétique, contexte par embeddings).

## 2. Objectif

Fournir un banc d'essai **reproductible** et **documenté** pour :

- mesurer le comportement de différents modèles NER sur des transcriptions ;
- explorer des méthodes de rapprochement de mentions ;
- préparer une future étape de rattachement à un référentiel d'autorité,
  **avec validation humaine**.

## 3. Fonctionnement général

```
Transcriptions (JSON)
        │
        ▼
  NER (plusieurs modèles)      →  results/<modele>/*.json
        │
        ▼
  Comparaison des modèles      →  results/comparison.csv, entities.csv, summary.csv
        │
        ▼
  Résolution d'entités         →  results/linking_final/*.csv
        │
        ▼
  Candidats à valider par un humain
```

Le NER utilise des modèles d'intelligence artificielle. La résolution
d'entités combine des algorithmes classiques (fuzzy, phonétique) et un
modèle d'embeddings. **Aucun regroupement automatique n'est une vérité** :
ce sont des propositions à valider.

### Structure du dépôt

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_benchmark.py            # modèles + compare.py + linking (proposé)
├── src/
│   ├── common.py
│   ├── run_camembert.py
│   ├── run_moderncamembert.py
│   ├── run_gliner.py
│   ├── run_gliner_large.py
│   ├── run_sauerkraut.py
│   ├── compare.py
│   ├── benchmark_linking.py
│   └── benchmark_linking_final.py
├── examples/
│   ├── example_transcription.json
│   └── README.md
├── transcriptions/             # transcriptions d'entrée
├── results/                    # sorties générées
└── docs/
    ├── methodology.md
    ├── ner-benchmark.md
    └── entity-resolution.md
```

## 4. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Les modèles sont **téléchargés automatiquement** au premier lancement depuis
leur source officielle (Hugging Face). Ils ne sont pas inclus dans le dépôt.

Détection automatique du matériel : les scripts utilisent le GPU (CUDA) s'il
est disponible, sinon le CPU.

## 5. Format des données

Un fichier JSON par transcription :

```json
{
  "id": "example_001",
  "original_name": "example.mp3",
  "text": "Marie Dupont s'est rendue à Lausanne ...",
  "segments": []
}
```

Des propriétés supplémentaires (`segments`, `words`, timestamps, scores de
confiance) peuvent exister, mais **le benchmark NER n'utilise que la
propriété `text`**.

Un exemple fictif est fourni dans `examples/`.

## 6. Lancer un modèle

```bash
python src/run_sauerkraut.py \
  --input transcriptions \
  --output results/sauerkraut_gliner
```

Les valeurs par défaut permettent aussi une utilisation simple :

```bash
python src/run_sauerkraut.py
```

Chaque script accepte `--input`, `--output` et, selon le modèle,
`--threshold`, `--chunk-size`, `--overlap`.

## 7. Lancer tous les benchmarks

```bash
python run_benchmark.py
```

Ce script enchaîne :

1. les modèles NER les uns après les autres ;
2. `compare.py` ;
3. **la résolution d'entités** (`benchmark_linking_final.py`), proposée à
   la fin.

Il s'arrête proprement en cas d'erreur et affiche l'étape en cours.

Sécurité des résultats existants :

- si des résultats existent déjà pour un modèle, le script **demande** s'il
  faut les écraser ;
- en cas d'écrasement (réponse « oui », ou option `--force`), le dossier de
  sortie concerné est **vidé** avant la ré-exécution, afin de ne pas
  conserver d'anciens fichiers obsolètes.

Options utiles :

```bash
python run_benchmark.py --models sauerkraut gliner   # limiter les modèles
python run_benchmark.py --linking no                 # ne pas proposer le linking
python run_benchmark.py --linking yes                # lancer le linking sans demander
python run_benchmark.py --skip-compare               # ne pas lancer compare.py
python run_benchmark.py --force                      # écraser sans demander
```

`--linking-ner <dossier>` choisit les résultats NER utilisés pour la
résolution d'entités (défaut : `results/sauerkraut_gliner` si présent, sinon
le premier dossier disponible).

## 8. Comparer les résultats

```bash
python src/compare.py --results results
```

Produit `comparison.csv`, `entities.csv` et `summary.csv` dans
`results/` (voir `results/README.md`).

`compare.py` distingue les **occurrences** (ex. 10 mentions d'une même
personne) des **mentions textuelles uniques** (formes distinctes). Une
mention textuelle unique **n'est pas** une personne unique : seule une
résolution d'entités peut établir ce lien.

## 9. Tester la résolution d'entités

```bash
python src/benchmark_linking_final.py \
  --ner-results results/sauerkraut_gliner \
  --transcriptions transcriptions \
  --output results/linking_final
```

Compare les stratégies fuzzy, fuzzy + phonétique, et
fuzzy + phonétique + contexte (embeddings `BAAI/bge-m3`).

Une expérience antérieure est également disponible :

```bash
python src/benchmark_linking.py \
  --ner-results results/sauerkraut_gliner \
  --output results/linking
```

Tous les regroupements produits sont des **candidats à valider**, jamais des
entités validées. Voir `docs/entity-resolution.md`.

## 10. Comprendre les fichiers générés

- `results/comparison.csv` : statistiques par transcription et par modèle ;
- `results/entities.csv` : toutes les occurrences NER ;
- `results/summary.csv` : statistiques agrégées par modèle ;
- `results/linking_final/*.csv` : candidats de rapprochement et comparaison
  des méthodes.

Détail complet dans `results/README.md`.

## 11. Méthodologie

- `docs/methodology.md` : démarche, découpage en fenêtres, gestion CPU/GPU,
  configuration de référence, reproductibilité, intégrité scientifique ;
- `docs/ner-benchmark.md` : modèles testés, catégories, fichiers produits ;
- `docs/entity-resolution.md` : méthodes de rapprochement, architecture
  cible, référentiels d'autorité, métriques futures.

## 12. Limites

- **Aucun gold standard humain complet** n'était disponible : le nombre
  d'entités détectées n'est pas une mesure de précision, et les scores des
  modèles ne sont pas des précisions mesurées.
- Precision / Recall / F1 ne peuvent pas être calculés sans annotations
  humaines.
- Les regroupements automatiques sont des **candidats**, pas des vérités.
- L'étape de rattachement à un référentiel d'autorité n'est pas encore
  implémentée.

## 13. Référentiels d'autorité

Le dépôt est neutre vis-à-vis du référentiel. L'entité validée pourra être
reliée à IdRef, Wikidata, VIAF, un référentiel interne, ou tout autre
référentiel d'autorité ou métier.

## 14. Reproductibilité

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python run_benchmark.py --input examples      # test avec l'exemple fictif
python src/benchmark_linking_final.py --ner-results results/sauerkraut_gliner --transcriptions examples
```

La configuration expérimentale de référence (OS, versions, GPU) est
documentée dans `docs/methodology.md`, section 6. Ce n'est **pas** une
configuration minimale : le benchmark fonctionne sur d'autres
configurations, éventuellement plus lentement.

## 15. Licences

- **Scripts (code du dépôt)** : MIT (voir `LICENSE`), © 2026 Besim Berisha.
  La licence MIT ne couvre **que** les scripts : ni les poids de modèles, ni
  les modèles tiers, ni les transcriptions et données.
- **Modèles** : chaque modèle conserve sa propre licence. Certaines sont
  confirmées (Apache-2.0 ou MIT) ; pour deux d'entre eux, la licence n'a
  **pas pu être déterminée** — voir `docs/ner-benchmark.md` et
  `docs/entity-resolution.md`.
- **Bibliothèques** : celles de `requirements.txt`.
- **Données / transcriptions** : droits réservés à leurs détenteurs.

Les poids des modèles ne sont **pas** distribués dans le dépôt.

## 16. Citation / auteur

Auteur et méthodologie : **Besim Berisha**.

Aucun système d'intelligence artificielle n'est présenté comme auteur de ce
projet.
