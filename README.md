# Développement d’un Modèle Prédictif pour Estimation des Primes et Analyse de la Dérive

## Contexte et Objectif: 

Ce projet a été conçu pour répondre à deux problématiques clés dans le domaine de l'assurance automobile :

  - Estimation des Primes Commerciales : Construire un modèle prédictif performant pour estimer avec précision la prime commerciale payée par les assurés, en utilisant des données variées (caractéristiques des conducteurs, des véhicules, et d'autres facteurs).
  - Analyse de la Dérive : Identifier et analyser les changements survenus dans les données (data drift) et les relations entre variables (concept drift) pour expliquer la dégradation éventuelle des performances du modèle sur des données récentes.


Le projet met en avant une approche modulaire, réutilisable et maintenable, en suivant les bonnes pratiques de structuration et de documentation pour les pipelines de machine learning.

## Structure du Projet:

Le projet est organisé de manière à favoriser la clarté, la traçabilité, et la réutilisabilité des composants.

### 1. Organisation des Dossiers: 

- config/ : Contient le fichier config.yaml, qui centralise les configurations (chemins des fichiers, hyperparamètres des modèles, etc.).
- data/ : Regroupe les données utilisées dans le projet :
- X_train.csv : Données d'entraînement.
- X_test.csv : Données de test.
- X_drift.csv : Données pour l'analyse de dérive.
- X_train_cleaned.csv : Données nettoyées après prétraitement.
- models/ : Contient les modèles sauvegardés (ex. : Model.pkl) pour une réutilisation future.
- output/ : Stocke les résultats finaux (ex. : results.csv pour les prédictions).
- catboost_info/ : Dossier généré par CatBoost contenant les informations liées à l'entraînement.
- __pycache__/ : Fichiers Python compilés, ignorés via .gitignore.


### 2. Scripts et Notebooks :

- common_function.py : Regroupe les fonctions utilitaires pour automatiser et structurer le pipeline (prétraitement, visualisation, gestion des modèles, évaluation, etc.).

- Partiel1.ipynb : Notebook dédié à l'analyse exploratoire et à la construction du modèle prédictif (Partie I).
- Partiel2.ipynb : Notebook pour l'analyse de la dérive des données et l'évaluation des performances du modèle (Partie II).
- .gitignore : Liste les fichiers à ignorer par Git (par ex. : fichiers temporaires, données sensibles).
- README.md : Documentation détaillée du projet.
### 3. Configuration du Projet: 

Le fichier config.yaml permet de centraliser et de simplifier la configuration du pipeline :

- Données : Chemins des fichiers d'entraînement, de test, et de dérive.
- Caractéristiques : Listes des colonnes numériques, catégoriques, et transformations spécifiques (logarithmique, capping).
- Modèles : Hyperparamètres pour les modèles linéaires, les méthodes d'ensemble (Random Forest, XGBoost, LightGBM, CatBoost), et les réseaux de neurones (MLP).
- Évaluation : Configuration de la validation croisée et des métriques de performance (neg_mean_squared_error).
### Fonctionnalités Clés
- 1. Prétraitement des Données
Détection des valeurs manquantes avec detect_missing_values.
Gestion des valeurs aberrantes via detect_outliers et cap_outliers.
Transformations avancées : Transformation logarithmique (log_transform) et mise à l’échelle robuste (scale_features).
Encodage des variables catégoriques avec preprocess_modeling.
- 2. Modélisation
Entraînement et évaluation de modèles variés :
Régressions linéaires (Ridge, Lasso, ElasticNet).
Méthodes d'ensemble (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost).
Réseau de neurones (MLP).
Optimisation des hyperparamètres via validation croisée.
Sauvegarde et chargement des modèles avec joblib.
- 3. Analyse Exploratoire
Visualisation des distributions des caractéristiques avec plot_feature_distribution.
Corrélation entre variables via plot_correlation_matrix.
Relation entre variables : Graphiques de dispersion (plot_scatter) et box plots (plot_box_plot).
- 4. Analyse de la Dérive
Dérive des distributions (data drift) : Tests KS et Chi-square pour les variables continues et catégoriques (analyze_data_drift).
Divergence de Jensen-Shannon pour évaluer les changements dans les distributions (calculate_jsd_categorical).
Comparaison des matrices de corrélation pour identifier les évolutions entre ensembles de données (comparer_matrices_correlation).
- 5. Évaluation
Métriques sur validation croisée : MSE, R², MAE avec evaluation_metrics.
Évaluation sur ensemble de validation avec evaluate_on_validation_set.
Bonnes Pratiques Suivies
Modularité : Toutes les fonctions utilitaires sont centralisées dans common_function.py, facilitant la réutilisabilité et la maintenance.
Centralisation des configurations : Utilisation de config.yaml pour séparer les paramètres de configuration du code.
Visualisation : Graphiques clairs et variés pour mieux comprendre les données et les résultats.
Traçabilité : Les données, modèles, et résultats sont organisés dans des dossiers dédiés pour simplifier le suivi.
Gestion de la dérive : Méthodes robustes pour analyser les changements dans les distributions et relations des données.