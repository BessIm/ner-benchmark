# Benchmark NER

Cette page documente l'étape d'**extraction d'entités nommées** (Named
Entity Recognition, NER).

## 1. Principe

Chaque transcription est analysée par plusieurs modèles. Pour chaque
modèle, on obtient une liste d'entités (mentions) avec leur catégorie,
leur position dans le texte et un score de confiance.

Les modèles sont appelés par des scripts indépendants :

```bash
python src/run_camembert.py
python src/run_moderncamembert.py
python src/run_gliner.py
python src/run_gliner_large.py
python src/run_sauerkraut.py
```

ou, en une seule commande :

```bash
python run_benchmark.py                     # tous les modèles + compare.py
python run_benchmark.py --models sauerkraut gliner
```

## 2. Modèles testés

Identifiants **exactement** tels qu'utilisés dans les scripts.

| Script | Identifiant du modèle | Auteur / fournisseur | Rôle dans le benchmark | Famille technique | Licence |
|---|---|---|---|---|---|
| `run_camembert.py` | `Jean-Baptiste/camembert-ner` | Jean-Baptiste (Hugging Face) | Modèle supervisé de référence (baseline) | Transformer (CamemBERT, français) | non déterminée |
| `run_moderncamembert.py` | `CATIE-AQ/Moderncamembert_4entities` | CATIE-AQ | Modèle supervisé spécialisé, 4 types d'entités | Transformer (ModernCamemBERT) | non déterminée |
| `run_gliner.py` | `urchade/gliner_multi-v2.1` | urchade (GLiNER) | GLiNER multilingue, détection sans entraînement spécifique (zero-shot) | GLiNER (transformer bidirectionnel) | Apache-2.0 |
| `run_gliner_large.py` | `gliner-community/gliner_large-v2.5` | gliner-community | Variante large de GLiNER | GLiNER | Apache-2.0 |
| `run_sauerkraut.py` | `VAGOsolutions/SauerkrautLM-GLiNER` | VAGOsolutions | GLiNER adapté (corpus germanophones / européens) | GLiNER | MIT |

Pages officielles des modèles :

- https://huggingface.co/Jean-Baptiste/camembert-ner
- https://huggingface.co/CATIE-AQ/Moderncamembert_4entities
- https://huggingface.co/urchade/gliner_multi-v2.1
- https://huggingface.co/gliner-community/gliner_large-v2.5
- https://huggingface.co/VAGOsolutions/SauerkrautLM-GLiNER

Pour `Jean-Baptiste/camembert-ner` et
`CATIE-AQ/Moderncamembert_4entities`, la licence n'a **pas pu être
déterminée** (information indisponible localement, non vérifiable par
l'auteur du benchmark). Elle n'est donc pas affirmée ici.

> **Les poids des modèles ne sont pas distribués dans le dépôt.** Ils sont
> téléchargés depuis leur source officielle par les bibliothèques Python
> (`transformers`, `gliner`) au premier lancement.

## 3. Seuils et fenêtres

Le seuil de confiance dépend du type de modèle :

| Modèle | Seuil (`THRESHOLD`) | Fenêtre (`CHUNK_SIZE`) | Overlap |
|---|---|---|---|
| CamemBERT | — (pipeline `transformers`) | 5000 | 500 |
| ModernCamemBERT | — (pipeline `transformers`) | 5000 | 500 |
| GLiNER multi | 0.4 | 1500 | 200 |
| GLiNER large v2.5 | 0.40 | 1500 | 200 |
| SauerkrautLM-GLiNER | 0.50 | 1500 | 200 |

Les modèles GLiNER reçoivent une liste de libellés à chercher :

- `run_gliner.py` : `personne`, `organisation`, `lieu`, `événement` ;
- `run_gliner_large.py` et `run_sauerkraut.py` : `person`, `organization`,
  `location`, `event`.

Ces seuils sont ceux des expériences d'origine. Les tailles de fenêtres
GLiNER ont été ramenées à `1500 / 200` pour rester sous la limite de
384 tokens des modèles GLiNER (voir `docs/methodology.md`) ; **ce correctif
modifie les résultats** par rapport aux expériences initiales. Tous ces
paramètres restent modifiables en ligne de commande (`--threshold`,
`--chunk-size`, `--overlap`).

## 4. Catégories d'entités et normalisation

Catégories utilisées dans le benchmark :

```
PERSONNE
ORGANISATION
LIEU
EVENEMENT
```

Certains modèles supervisés peuvent en plus produire :

```
MISC
```

Les catégories **ne sont pas nécessairement identiques nativement** d'un
modèle à l'autre (par exemple `PER`, `ORG`, `LOC`, `MISC`). Pour permettre
la comparaison, `src/common.py` normalise les libellés :

```python
LABEL_MAP = {
    "PER": "PERSONNE", "PERSON": "PERSONNE", "PERSONNE": "PERSONNE",
    "ORG": "ORGANISATION", "ORGANIZATION": "ORGANISATION", "ORGANISATION": "ORGANISATION",
    "LOC": "LIEU", "LOCATION": "LIEU", "LIEU": "LIEU",
    "MISC": "MISC",
    "EVENT": "EVENEMENT", "EVENEMENT": "EVENEMENT",
}
```

Les préfixes `B-` / `I-` (utilisés par certains modèles) sont également
supprimés au moment de la normalisation.

## 5. Occurrences et mentions textuelles uniques

Le benchmark **conserve les occurrences**. Une même personne mentionnée
dix fois produit dix occurrences.

`compare.py` distingue clairement :

- **occurrences** : nombre total de détections ;
- **mentions textuelles uniques par transcription** : formes distinctes
  dans un même fichier (comparaison insensible à la casse) ;
- **mentions textuelles uniques** (agrégées par modèle dans `summary.csv`).

> Une « mention textuelle unique » n'est **pas** une « personne unique ».
> `Marie Dupont`, `Marie Dupond` et `M. Dupont` sont trois formes
> textuelles différentes tant qu'une résolution d'entités n'a pas établi
> qu'elles désignent la même personne.

## 6. Fichiers produits

`compare.py` écrit dans le dossier `results/` :

| Fichier | Contenu |
|---|---|
| `entities.csv` | **Occurrences** NER (une ligne par occurrence). |
| `comparison.csv` | **Statistiques par transcription et par modèle**. |
| `summary.csv` | **Statistiques agrégées par modèle**. |

Détail des colonnes principales de `summary.csv` :

- `nb_transcriptions` : nombre de transcriptions traitées ;
- `total` / `unique_total` : occurrences / mentions uniques ;
- `personnes`, `organisations`, `lieux`, `evenements`, `misc` :
  occurrences par catégorie ; les colonnes `unique_*` correspondantes
  donnent les mentions uniques ;
- `elapsed_seconds` : temps de traitement cumulé ;
- `ratio_repetition_*` : rapport occurrences / mentions uniques ;
- `score_moyen`, `score_median` : statistiques sur les scores du modèle ;
- `entites_1_2_caracteres` : nombre d'entités très courtes (1 ou 2
  caractères), utile pour repérer du bruit.

## 7. Limites

Aucun gold standard humain complet n'était disponible. Par conséquent :

- le nombre d'entités détectées n'est **pas** une mesure de précision ;
- le score de confiance du modèle n'est **pas** une précision mesurée ;
- Precision / Recall / F1 ne peuvent pas être calculés sans annotations
  humaines.

L'objectif est d'utiliser progressivement les validations et corrections
des catalogueurs pour construire un gold standard, qui permettra de
calculer Precision, Recall, F1-score, Entity Linking Accuracy et Top-k
candidate recall (voir `docs/entity-resolution.md`).
