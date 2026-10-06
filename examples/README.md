# Exemples

Ce dossier contient un exemple de transcription **entièrement fictif** :

- `example_transcription.json`

Il sert à montrer le format attendu en entrée et à permettre de tester le
benchmark sans publier de données réelles.

## Format des données d'entrée

Les scripts du benchmark lisent des fichiers JSON, un fichier par
transcription. Le format minimal est le suivant :

```json
{
  "id": "example_001",
  "original_name": "example.mp3",
  "text": "Marie Dupont s'est rendue à Lausanne ...",
  "segments": []
}
```

Une transcription réelle produite par un système de reconnaissance de la
parole (speech-to-text) peut contenir **d'autres propriétés** : `segments`,
`words`, des horodatages (timestamps), des scores de confiance, etc.

**Important : le benchmark NER n'utilise que la propriété JSON `text`.**

Les autres propriétés sont ignorées par les scripts `run_*.py`. Elles
peuvent en revanche être utilisées par d'autres traitements (par exemple
extraire le contexte d'une mention pour la résolution d'entités).

## Utilisation

```bash
python run_benchmark.py --input examples
```

ou directement :

```bash
python src/run_sauerkraut.py --input examples --output results/exemple
```
