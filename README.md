# Climate Stress Testing Project

Ce projet vise à réaliser des stress tests climatiques sur un portefeuille de crédit en utilisant des données de ratings, des variables macroéconomiques et des données du nGFS (Network for Greening the Financial System). Vous devrez mobiliser des modèles G-VAR (et autres) à l'aide des tableaux entrée-sorties. Ces tableaux que l'on nommera TES montrent les échanges que peuvent se faire entre différents secteurs et différents pays. Le fichier excel `ReadMe_ICIO_small.xlsx` vous aidera à mieux comprendre. Bon courage :) N'hésitez pas à contacter Gaëtan Blécon pour toute question sur le code.

## Pour les novices de Git :

GitHub sert à mettre des données sur un cloud et pouvoir travailler de manière inteligente, en modifiant le code source sans changer le code qui est en production. Dans la plupart des entreprises, il est essentiel de savoir le manipuler. L'idée c'est que plusieurs personnes peuvent travailler en même temps sur des sujets parallèles sans compromettre le travail des autres.

Voici un site qui permet de comprendre une bonne quantité des options envisageables : https://learngitbranching.js.org/?locale=fr_FR

Pour débuter, créez vous un compte et chercher à cloner mon "repository". Je pense qu'il y a pas de grande difficulté avec l'aide d'internet. Si vous rencontrez des difficultés, contactez encore une fois Gaëtan Blécon.

Vous devrez bien penser à rajouter les données. (Elles ne sont pas directemment intégrées car le clonage serait un peu trop lourd.) 

## Structure du Projet

```
.
├── data/                  # Dossier contenant les données
│   ├── credit_ratings/    # Données des ratings de crédit
│   ├── international_TES  # Données des Tableaux entrées sorties
│   ├── macro_data/        # Variables macroéconomiques
│   └── ngfs_data/         # Données NGFS
├── notebooks/             # Jupyter notebooks pour l'analyse
├── src/                   # Code source Python
└── requirements.txt       # Dépendances du projet
```

## Installation

Sous VS Code, ouvrez un nouveau terminal et exécutez les commandes suivantes

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
2. Lancer le notebook : main.ipynb et exécutez les différentes cellules.

## Données

- `data_rating_corporate.xlsx`: Contient les données historiques des ratings de crédit du portefeuille
- `Données_macro_hist_v2.xlsx`: Contient les variables macroéconomiques
- `NiGEM_data.xlsx`: Contient les données des scénarios du NGFS
- `20XX_SML.csv`: Contient les tableaux TES
- `NACE 38 - 88 detaille vf.xlsx`: Contient les descriptions des différents agrégats sectoriels
- `ReadMe_ICIO_small.xlsx`: Contient des détails sur le tableau TES (A noter que la plupart des infos ne seront pas utilisées)

## Modification des fichiers

Parfois, lorsque vous avez déjà imorté un fichier (par exemple, import importing), si vous modifiez ce fichier et que vous réimportez ce fichier modifié, le kernel a enregistré l'ancienne version du fichier et ne le met pas à jour. Vous devrez exécuter ces lignes de code :

import importlib # A faire uniquement une fois évidemment

importlib.reload(nom_fichier.py)