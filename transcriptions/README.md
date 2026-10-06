# Transcriptions

Ce dossier accueille les **transcriptions d'entrée** du benchmark : un
fichier JSON par transcription.

Aucune transcription réelle n'est fournie. Un exemple entièrement fictif
est disponible dans `../examples/`.

## Format attendu

```json
{
  "id": "identifiant",
  "original_name": "fichier.mp3",
  "text": "texte complet de la transcription",
  "segments": []
}
```

Le benchmark NER n'utilise **que** la propriété `text`.

## Utilisation

```bash
# Déposer les transcriptions ici, puis :
python run_benchmark.py --input transcriptions
```
