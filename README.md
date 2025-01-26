# Développement d’un Modèle Prédictif pour Estimation des Primes et Analyse de la Dérive

## Contexte et Objectif

Ce projet vise à résoudre deux problématiques clés dans le domaine de l'assurance automobile :

1. **Estimation des Primes Commerciales** : Construire un modèle prédictif pour estimer précisément les primes commerciales en fonction des caractéristiques des conducteurs, des véhicules, et d'autres facteurs.
2. **Analyse de la Dérive (Drift)** : Identifier et analyser les changements dans les données (data drift) et les relations entre variables (concept drift) pour comprendre et résoudre la dégradation des performances du modèle.

Le projet met en avant une **approche modulaire**, **maintenable**, et **réutilisable**, en suivant les bonnes pratiques de structuration et de documentation pour les pipelines de machine learning.

---

## Structure du Projet

### Organisation des Dossiers

- **`config/`** : Contient le fichier `config.yaml`, qui centralise les configurations (chemins des fichiers, hyperparamètres des modèles, etc.).
- **`data/`** : Regroupe les données utilisées dans le projet :
  - `X_train.csv` : Données d'entraînement.
  - `X_test.csv` : Données de test.
  - `X_drift.csv` : Données pour l'analyse de dérive.
  - `X_train_cleaned.csv` : Données nettoyées après prétraitement.
- **`models/`** : Contient les modèles sauvegardés (ex. : `Model.pkl`) pour une réutilisation future.
- **`output/`** : Stocke les résultats finaux (ex. : `results.csv` pour les prédictions).
- **`__pycache__/`** : Fichiers Python compilés, ignorés via `.gitignore`.

### Scripts et Notebooks

- **`common_function.py`** : Regroupe les fonctions utilitaires pour automatiser et structurer le pipeline (prétraitement, visualisation, gestion des modèles, évaluation, etc.).
- **`Partiel1.ipynb`** : Notebook dédié à l’analyse exploratoire et à la construction du modèle prédictif.
- **`Partiel2.ipynb`** : Notebook pour l’analyse de la dérive des données et l’évaluation des performances du modèle.
- **`.gitignore`** : Liste les fichiers à ignorer par Git (par ex. : fichiers temporaires, données sensibles).

---

## Fonctionnalités Clés

### 1. Prétraitement des Données
- Détection des valeurs manquantes avec `detect_missing_values`.
- Gestion des valeurs aberrantes via `cap_outliers`.
- Transformations avancées : transformation logarithmique (`log_transform`) et mise à l’échelle robuste (`scale_features`).
- Encodage des variables catégoriques avec `preprocess_modeling`.

### 2. Modélisation
- Entraînement et évaluation de modèles variés :
  - Régressions linéaires : Ridge, Lasso, ElasticNet.
  - Méthodes d'ensemble : Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost.
  - Réseau de neurones (MLP).
- Optimisation des hyperparamètres via validation croisée.
- Sauvegarde et chargement des modèles avec `joblib`.

### 3. Analyse Exploratoire
- Visualisation des distributions avec `plot_feature_distribution`.
- Matrice de corrélation des variables avec `plot_correlation_matrix`.
- Relation entre variables : graphiques de dispersion (`plot_scatter`) et box plots (`plot_box_plot`).

### 4. Analyse de la Dérive
- Analyse des dérives des distributions : tests KS et Chi-square pour les variables continues et catégoriques (`analyze_data_drift`).
- Divergence de Jensen-Shannon pour évaluer les changements dans les distributions (`calculate_jsd_categorical`).
- Comparaison des matrices de corrélation pour identifier les évolutions dans les relations entre variables (`comparer_matrices_correlation`).

### 5. Évaluation
- Métriques sur validation croisée : MSE, R², MAE avec `evaluation_metrics`.
- Évaluation sur ensemble de validation avec `evaluate_on_validation_set`.

---

## Configuration

Le fichier `config.yaml` centralise la configuration du pipeline :
- **Données** : Chemins des fichiers d’entraînement, de test, et de dérive.
- **Caractéristiques** :
  - Listes des colonnes numériques et catégoriques.
  - Colonnes nécessitant des transformations spécifiques (logarithmique, capping).
- **Modèles** : Hyperparamètres pour les modèles linéaires, les méthodes d'ensemble, et les réseaux de neurones.
- **Évaluation** : Configuration de la validation croisée et des métriques de performance (`neg_mean_squared_error`).

---

## Bonnes Pratiques Suivies

1. **Modularité** : Toutes les fonctions utilitaires sont centralisées dans `common_function.py`, facilitant la réutilisabilité.
2. **Centralisation des configurations** : Le fichier `config.yaml` sépare les paramètres de configuration du code.
3. **Traçabilité** : Les données, modèles, et résultats sont organisés dans des dossiers dédiés pour simplifier le suivi.
4. **Gestion de la dérive** : Utilisation de méthodes robustes pour analyser les changements dans les distributions et relations des données.
5. **Visualisation claire** : Graphiques variés pour mieux comprendre les données et les résultats.

---
