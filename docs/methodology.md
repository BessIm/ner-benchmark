# Méthodologie

Ce document décrit la démarche générale du benchmark. Il s'adresse aussi
bien à des développeurs qu'à des documentalistes ou des responsables de
projet non informaticiens.

## 1. Objet

Le projet compare plusieurs méthodes pour :

1. **extraire des entités nommées** (NER) dans des transcriptions
   speech-to-text ;
2. **rapprocher ces mentions** entre elles pour proposer des regroupements
   (résolution d'entités).

Le cas d'étude est un corpus de transcriptions speech-to-text, mais
**le dépôt est générique** : il peut être appliqué à n'importe quel
corpus de transcriptions.

## 2. Deux étapes distinctes

| Étape | Question posée | Scripts |
|---|---|---|
| Benchmark NER | « Quelles entités sont mentionnées, et où ? » | `src/run_*.py` |
| Résolution d'entités | « Ces mentions désignent-elles la même entité réelle ? » | `src/benchmark_linking*.py` |

Le rattachement à un référentiel d'autorité externe (IdRef, Wikidata,
VIAF, ...) constitue une troisième étape, appelée **Entity Linking** au
sens documentaire. Elle n'est pas encore implémentée ici.

## 3. Données d'entrée

Chaque transcription est un fichier JSON. Le benchmark NER **n'utilise
que la propriété `text`**. Les propriétés supplémentaires éventuelles
(`segments`, `words`, timestamps, scores de confiance) sont ignorées à
cette étape.

Voir `examples/README.md` pour le format et un exemple fictif.

## 4. Découpage des longues transcriptions

Une transcription peut être trop longue pour être traitée d'un seul coup
par un modèle (limites de longueur des modèles, mémoire GPU). Chaque
script découpe donc le texte en **fenêtres** (`chunks`) de taille fixe.

- `CHUNK_SIZE` : nombre de caractères par fenêtre ;
- `OVERLAP` : nombre de caractères communs entre deux fenêtres
  consécutives.

L'**overlap** (chevauchement) évite de couper une entité située à la
frontière entre deux fenêtres : une mention à cheval sur la limite est vue
en entier dans au moins une des deux fenêtres.

Le chevauchement peut produire des **doublons** : la même entité est
détectée dans deux fenêtres. La fonction `clean_entities()` de
`src/common.py` supprime ces doublons en conservant, pour une même
position et un même libellé, le score le plus élevé.

Valeurs par défaut actuelles :

| Modèle | CHUNK_SIZE | OVERLAP |
|---|---|---|
| CamemBERT | 5000 | 500 |
| ModernCamemBERT | 5000 | 500 |
| GLiNER multi | 1500 | 200 |
| GLiNER large v2.5 | 1500 | 200 |
| SauerkrautLM-GLiNER | 1500 | 200 |

Ces valeurs sont modifiables en ligne de commande (`--chunk-size`,
`--overlap`).

## 5. Gestion CPU / GPU

Les scripts détectent automatiquement la disponibilité de CUDA :

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Aucun modèle de GPU n'est codé en dur. Le benchmark fonctionne donc aussi
sur CPU, simplement plus lentement.

## 6. Configuration expérimentale de référence

Les expériences de référence ont été exécutées sur la configuration
suivante. Il s'agit de la configuration utilisée pour produire les
résultats initiaux, **et non d'une configuration minimale requise**.

```
OS      : Ubuntu / Linux
Python  : 3.14.4
PyTorch : 2.14.0+cu132
CUDA    : disponible
GPU     : NVIDIA RTX PRO 5000 Blackwell
VRAM    : 47.3 GB
```

Versions des principales bibliothèques lors de ces expériences :

| Bibliothèque | Version |
|---|---|
| torch | 2.14.0+cu132 |
| transformers | 5.16.1 |
| gliner | 0.2.29 |
| sentence-transformers | 6.1.0 |
| huggingface_hub | 1.33.0 |
| sentencepiece | 0.2.2 |
| protobuf | 7.36.2 |
| tiktoken | 0.14.0 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| rapidfuzz | 3.14.6 |
| Metaphone | 0.6 |
| scikit-learn | 1.9.1 |
| tqdm | 4.70.1 |
| accelerate | 1.15.0 |

Le fichier `requirements.txt` ne fige pas ces versions : il liste les
dépendances nécessaires. Les versions ci-dessus servent de référence pour
reproduire les expériences initiales.

## 7. Reproductibilité

1. Créer un environnement Python vierge.
2. Installer les dépendances : `python -m pip install -r requirements.txt`.
3. Placer des transcriptions JSON dans `transcriptions/` (ou utiliser
   `examples/`).
4. Lancer les modèles, la comparaison et (au choix) la résolution d'entités :

   ```bash
   python run_benchmark.py
   ```

   À la fin, le script **propose** de lancer `benchmark_linking_final.py`
   (répondre oui/non). Ce comportement peut être forcé :

   ```bash
   python run_benchmark.py --linking yes    # lancer le linking sans demander
   python run_benchmark.py --linking no     # ne pas lancer le linking
   ```

5. Alternativement, lancer la résolution d'entités seule :

   ```bash
   python src/benchmark_linking_final.py
   ```

Toutes les commandes du `README.md` utilisent des chemins **relatifs** et
des valeurs par défaut : aucune adaptation à une machine particulière
n'est nécessaire.

## 8. Intégrité scientifique

Les scripts d'origine ont été généralisés (ajout de `argparse`, chemins
paramétrables) **sans modifier** :

- les modèles testés ;
- les libellés d'entités ;
- les seuils expérimentaux ;
- les méthodes de comparaison ;
- les formules de score ;
- la logique des expériences.

### Point d'attention documenté

Le champ `text_length` des fichiers de résultats JSON est actuellement
**toujours `null`** (il n'est jamais renseigné par `save_result()`). Ce
comportement a été conservé tel quel : il n'a pas été corrigé
silencieusement. `compare.py` lit ce champ mais `comparison.csv` conserve
donc une colonne `text_length` vide.

## 9. Limites

Aucun gold standard humain complet n'était disponible pendant le
benchmark exploratoire (voir `docs/ner-benchmark.md`, section « Limites »).

En conséquence :

- le nombre d'entités détectées **n'est pas** une mesure de précision ;
- le score de confiance d'un modèle **n'est pas** une précision mesurée ;
- on ne peut pas calculer Precision / Recall / F1 sans annotations
  humaines.

Les regroupements produits sont des **candidats** à valider, jamais des
vérités.
