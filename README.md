# Climate Stress Testing Project

Ce projet vise à réaliser des stress tests climatiques sur un portefeuille de crédit en utilisant des données de ratings, des variables macroéconomiques et des données du nGFS (Network for Greening the Financial System).

## Structure du Projet

```
.
├── data/                  # Dossier contenant les données
│   ├── credit_ratings/    # Données des ratings de crédit
│   ├── macro_data/       # Variables macroéconomiques
│   └── ngfs_data/        # Données NGFS
├── notebooks/            # Jupyter notebooks pour l'analyse
├── src/                 # Code source Python
└── requirements.txt     # Dépendances du projet
```

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
```

2. Activer l'environnement virtuel :
- Windows :
```bash
venv\Scripts\activate
```
- Unix/MacOS :
```bash
source venv/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Activer l'environnement virtuel (voir ci-dessus)
2. Lancer Jupyter Notebook :
```bash
jupyter notebook
```

## Données

- `credit_ratings.xlsx`: Contient les données historiques des ratings de crédit du portefeuille
- `macro_variables.xlsx`: Contient les variables macroéconomiques
- `ngfs_scenarios.xlsx`: Contient les données des scénarios NGFS

## Modification des fichiers

Parfois, lorsque vous avez déjà imorté un fichier (par exemple, import importing), si vous modifiez ce fichier et que vous reimportez ce fichier modifié, le kernela enregistré l'ancienne version du fichier et ne lemet pasà jour. Vous devrez exécuter ces lignes de code :

import importlib
importlib.reload(nom_fichier.py)