# Résultats

Ce dossier reçoit les fichiers **générés** par le benchmark.

`comparison.csv` et `summary.csv` ne contiennent que des valeurs numériques
(comptages, scores, durées). `entities.csv` et les fichiers par modèle
contiennent des **mentions nominatives** issues des transcriptions.

## Contenu des dossiers

Chaque modèle NER écrit dans son propre sous-dossier (`results/<modele>/`),
avec un fichier JSON par transcription :

```json
{
  "file": "transcription1.json",
  "model": "sauerkraut_gliner",
  "elapsed_seconds": 12.34,
  "text_length": null,
  "entities": [
    { "text": "Marie Dupont", "label": "PERSONNE", "score": 0.91, "start": 0, "end": 12 }
  ]
}
```

- `text` : la forme textuelle détectée (mention) ;
- `label` : la catégorie normalisée (`PERSONNE`, `ORGANISATION`, `LIEU`,
  `EVENEMENT`, éventuellement `MISC`) ;
- `score` : score de confiance du modèle (ce n'est **pas** une précision
  mesurée) ;
- `start` / `end` : position de la mention dans le texte.

## Fichiers produits par `compare.py`

`compare.py` lit tous les sous-dossiers de modèles et écrit à la racine de
`results/` :

| Fichier | Contenu |
|---|---|
| `entities.csv` | **Toutes les occurrences** détectées, une ligne par occurrence (`model`, `file`, `text`, `label`, `score`, `start`, `end`). |
| `comparison.csv` | **Statistiques par transcription et par modèle** : occurrences totales et par type, mentions textuelles uniques par transcription, temps d'exécution, longueur du texte. |
| `summary.csv` | **Statistiques agrégées par modèle** : total d'occurrences, mentions uniques, ratios de répétition, score moyen/médian, etc. |

> Une « mention textuelle unique » n'est **pas** une « personne unique ».
> `Marie Dupont`, `Marie Dupond` et `M. Dupont` sont trois formes
> textuelles différentes tant qu'une résolution d'entités n'a pas établi
> qu'elles désignent la même personne.

## Fichiers produits par la résolution d'entités

`benchmark_linking_final.py` (dossier `results/linking_final/`) :

| Fichier | Contenu |
|---|---|
| `forms.csv` | Formes textuelles uniques et leurs contextes. |
| `candidate_suggestions.csv` | Paires candidates avec scores fuzzy, phonétique et contexte. |
| `method_rankings.csv` | Classement des candidats pour chaque méthode. |
| `method_top_choices.csv` | Meilleur candidat par forme selon chaque méthode. |
| `conservative_clusters.csv` | Regroupements très conservateurs (propositions). |
| `cluster_summary.csv` | Taille des regroupements. |
| `linking_summary.csv` | Synthèse de l'expérience. |

`benchmark_linking.py` (dossier `results/linking/`) produit notamment
`candidate_entities.csv`, `clusters.csv`, `candidate_assignments.csv`,
`disagreements.csv`, `review_top_clusters.csv` et `linking_summary.csv`.

> Tous ces regroupements sont des **propositions (candidats)** soumises à
> validation humaine. Ce ne sont jamais des entités validées.
