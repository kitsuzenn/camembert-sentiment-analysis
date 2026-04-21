# camembert-sentiment-analysis
CamemBERT fine-tuné en 3 classes (Positif/Négatif/Neutre) pour l'analyse de sentiments en français
# Analyse de Sentiment FR — CamemBERT 3 classes

Modèle CamemBERT fine-tuné pour l'analyse de sentiments en français, avec 3 classes : **Positif**, **Négatif** et **Neutre**.

## Démo

Application déployée sur Hugging Face Spaces :
[Tester le modèle ici](https://huggingface.co/spaces/Dahren/camembert-sentiment-analysis)

## Le défi

Les modèles de sentiment français existants travaillent quasi exclusivement en 2 classes (positif/négatif). Le dataset Allociné de référence exclut les avis 3 étoiles, éliminant toute notion de neutralité.

Ce projet comble ce manque en construisant un dataset 3 classes avec une classe neutre fonctionnelle.

## Approche

### Dataset custom (9 000 exemples)
- **3 000 positifs** — critiques Allociné 4-5 étoiles
- **3 000 négatifs** — critiques Allociné 1-2 étoiles
- **3 000 neutres** — phrases factuelles sur le cinéma (réalisateurs, dates, durées, castings)

### Itérations
Le dataset neutre a nécessité **3 versions** avant de fonctionner :
- **v1** : neutres synthétiques générés par IA → vocabulaire subtilement négatif → le modèle confondait neutre et négatif
- **v2** : neutres par templates combinatoires → même problème, les formulations "correct mais sans éclat" restaient négatives
- **v3** : phrases purement factuelles (style Wikipedia/fiche technique) → le modèle distingue enfin l'absence d'opinion

### Entraînement
- Modèle : `camembert-base` (110M paramètres)
- Fine-tuning : 3 epochs, batch size 16, ~40 min sur GPU T4
- Split : 80% train / 20% test
- Validation loss : 0.114

## Résultats

| Type de phrase | Exemple | Prédiction | Confiance |
|---------------|---------|------------|-----------|
| Positif | "Ce film est absolument génial" | Positif | 99.6% |
| Négatif | "Film nul, une perte de temps" | Négatif | 99.7% |
| Neutre | "Le film dure 1h47 et sort le 15 mars" | Neutre | 99.9% |
| Neutre | "Le réalisateur est né en 1965 à Lyon" | Neutre | 99.9% |
| Ambigu | "Un film correct, sans plus" | Négatif | 73.6% |

## Stack technique
- Python
- Hugging Face Transformers
- PyTorch
- Gradio
- Google Colab (GPU T4)

## Structure du projet
├── README.md
├── train.py          # Code d'entraînement
├── app.py            # Application Gradio
├── requirements.txt  # Dépendances
└── dataset/          # Scripts de construction du dataset

## Auteur
**Dahren Smith** — Projet réalisé seul dans le cadre d'une formation ML Engineer. 
Projet réalisé au mois 5 d'une formation intensive ML Engineer (24 mois)
