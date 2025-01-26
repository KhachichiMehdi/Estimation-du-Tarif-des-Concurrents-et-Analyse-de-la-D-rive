import warnings
warnings.filterwarnings("ignore")
import yaml
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from scipy.stats import zscore
from scipy.stats import ks_2samp ,chi2_contingency
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator
import os
import joblib


def load_config(path: str) -> Dict[str, Any]:
    """
    Charge les paramètres de configuration depuis un fichier YAML.

    Args:
        path (str): Chemin d'accès au fichier de configuration YAML.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant les paramètres de configuration.
    """      
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config




def load_data(path: str) -> pd.DataFrame:
          
    """
    Charge les données depuis un fichier CSV.

    Args:
        path (str): Chemin d'accès au fichier CSV.

    Returns:
        pd.DataFrame: Un DataFrame contenant les données.
    """
    return pd.read_csv(path)


def load_model(model_path: str) -> BaseEstimator:
    """
    Charge un modèle pré-entraîné depuis le chemin spécifié.

    Paramètres:
    model_path (str): Chemin vers le fichier du modèle sauvegardé.

    Retourne:
    BaseEstimator: Instance du modèle chargé.
    """
    model = joblib.load(model_path)
    return model


def save_model_with_directory(model: BaseEstimator, model_dir: str, model_filename: str) -> None:
    """
    Crée le répertoire spécifié s'il n'existe pas, puis sauvegarde le modèle dans ce répertoire.

    Paramètres:
    model (BaseEstimator): Le modèle à sauvegarder.
    model_dir (str): Le chemin du répertoire où sauvegarder le modèle.
    model_filename (str): Le nom du fichier pour sauvegarder le modèle (par défaut : 'model.pkl').
    """
    # Crée le répertoire s'il n'existe pas
    os.makedirs(model_dir, exist_ok=True)
    
    # Crée le chemin complet du modèle
    model_path = os.path.join(model_dir, model_filename)
    
    # Sauvegarde le modèle
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé avec succès dans {model_path}")


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte et retourne le total des valeurs manquantes par colonne.
    
    Args:
        df (pd.DataFrame): Le DataFrame à vérifier pour les valeurs manquantes.
    
    Retourne:
        pd.DataFrame: Un DataFrame avec les colonnes 'Column' et 'Missing Values' montrant
                      le nombre de valeurs manquantes pour chaque colonne.
    """
    missing_values = df.isna().sum()
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values
    })
    
    # Filtre pour afficher uniquement les colonnes avec des valeurs manquantes
    missing_df = missing_df[missing_df['Missing Values'] > 0].reset_index(drop=True)
    
    return missing_df




def plot_feature_distribution(df: pd.DataFrame, col: str):

    """
    Trace la distribution d'une caractéristique .

    Args
        df (pd.DataFrame): DataFrame contenant les données.
        col (str): Nom de la colonne de la caractéristique.
    Returns:
        None
    """
    sns.histplot(data=df, x=col, kde=True,color='salmon')
    plt.title(f'Distribution de {col} ')
    plt.xlabel(col)
    plt.ylabel('Fréquence')
    plt.grid(True)       


def plot_scatter(df:pd.DataFrame, col:str, target:str):
    """ Génère un graphique de dispersion pour visualiser la relation entre deux variables.
    Args
    df : DataFrame
        Les données contenant les colonnes à visualiser.
    col : str
        Nom de la colonne à utiliser pour l'axe des abscisses.
    target: str
        Nom de la colonne à utiliser pour l'axe des ordonnées.
    

    returns:
    --------
    None  """

    sns.regplot(x=col, y=target, data=df, scatter_kws={'color': 'skyblue'}, line_kws={'color': 'red'}  )
    plt.title(f"{target} vs {col} ")
    plt.xlabel(col)
    plt.ylabel(target)
    plt.grid(True)

def plot_correlation_matrix(df:pd.DataFrame):
    """Affiche une matrice de corrélation pour les colonnes numériques spécifiées.
    Args
    df : DataFrame
        Les données contenant les colonnes à visualiser.
    retuns:
    --------
    None
    
    """
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()






def plot_box_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str = ''):
    """
    Génère un box plot pour visualiser l'influence d'une variable catégorielle sur une variable numérique.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Nom de la colonne catégorielle pour l'axe des x.
        y_col (str): Nom de la colonne numérique pour l'axe des y.
        title (str, optionnel): Titre du graphique. Par défaut, il est vide.
        
    Returns:
        None: Affiche le box plot.
    """
    sns.boxplot(x=x_col, y=y_col, data=df, palette='Set2')
    plt.title(title if title else f'Influence de {x_col} sur {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)




def detect_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Détecte les valeurs aberrantes dans une colonne spécifique d'un DataFrame 
    en utilisant l'intervalle interquartile (IQR).
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col (str): Le nom de la colonne pour laquelle détecter les valeurs aberrantes.
    
    Returns:
        pd.DataFrame: Un DataFrame contenant uniquement les lignes avec des valeurs aberrantes dans la colonne spécifiée.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]






def detect_outliers_summary(df: pd.DataFrame, columns: list) -> dict:
    """
    Détecte et résume le nombre de valeurs aberrantes pour chaque colonne spécifiée.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        columns (list): La liste des colonnes pour lesquelles détecter les valeurs aberrantes.
    
    Returns:
        dict: Un dictionnaire avec le nom de chaque colonne en clé et le nombre de valeurs aberrantes en valeur.
    """
    outliers_summary = {}
    for col in columns:
        outliers = detect_outliers(df, col)
        outliers_summary[col] = len(outliers)
    return outliers_summary







def cap_outliers(df:pd.DataFrame, col: str)->pd.DataFrame:

    """
    Limite les valeurs aberrantes dans une colonne donnée d'un DataFrame en utilisant la méthode de l'IQR (Intervalle Interquartile).
    
    Cette méthode ajuste les valeurs aberrantes en les limitant aux bornes inférieure et supérieure calculées,
    sans les supprimer du DataFrame.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame contenant les données à traiter.
    
    col : str
        Le nom de la colonne dans laquelle détecter et ajuster les valeurs aberrantes.
    
    Retour:
    -------
    pd.DataFrame
        Le DataFrame avec les valeurs aberrantes ajustées pour la colonne spécifiée.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


def log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applique une transformation logarithmique (log1p) aux colonnes asymétriques spécifiées.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données initiales.
        columns (list): Liste des colonnes à transformer par log.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes transformées.
    """
    for col in columns:
        df[f"{col}_log"] = np.log1p(df[col])
    return df

def scale_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applique RobustScaler pour mettre à l'échelle les colonnes spécifiées, réduisant l'impact des valeurs aberrantes.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        columns (list): Liste des colonnes à mettre à l'échelle.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes mises à l'échelle.
    """
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df



def preprocess_modeling(df: pd.DataFrame, target_column: str=None, drop_columns: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les données pour la modélisation en appliquant l'encodage One-Hot avec OneHotEncoder aux variables catégorielles,
    en supprimant les colonnes inutiles et en séparant les caractéristiques de la cible.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée contenant les données.
        target_column (str): La colonne cible.
        drop_columns (list): Liste des colonnes à supprimer (par défaut à None).

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Un tuple contenant les caractéristiques (X) et la cible (y).
    """
    # Suppression des colonnes inutiles
    if drop_columns:
        df = df.drop(columns=drop_columns)
    
    # selectioner categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    if not categorical_cols.empty:  
        encoded_columns = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_columns, 
                                  columns=encoder.get_feature_names_out(categorical_cols), 
                                  index=df.index)

        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y  # Return both X and y
    
    # If no target_column, return only X
    return df



def evaluation_metrics(model: BaseEstimator, X: pd.DataFrame, y:pd.Series, cv: int) -> Dict[str, Any]:
    """
    Evaluates a model and returns Mean Squared Error (MSE), R2, and Mean Absolute Error (MAE).
    
    Args:
        model: The machine learning model to be evaluated.
        X (pd.DataFrame or np.ndarray): Features for evaluation.
        y (pd.Series or np.ndarray): Target variable for evaluation.
        cv (int): Number of cross-validation folds.
        
    Returns:
        dict: A dictionary containing 'Mean MSE', 'Mean R2', and 'Mean MAE' scores.
    """
    # cross-validation for each metric
    mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    
    # Calculate metrics
    metrics = {
        'Mean MSE': -np.mean(mse_scores),
        'Mean R2': np.mean(r2_scores),
        'Mean MAE': -np.mean(mae_scores)
    }
    
    return metrics


def evaluate_on_validation_set(model: BaseEstimator, y_val: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    """
    Évalue un modèle sur un ensemble de validation et renvoie les scores MSE, R2, MAE, et RMSE.
    
    Paramètres:
    ----------
    model : BaseEstimator
        Le modèle de machine learning à évaluer.
    y_val : pd.Series
        La variable cible réelle pour l'évaluation.
    y_pred : pd.Series
        Les prédictions du modèle sur l'ensemble de validation.
        
    Retourne:
    --------
    Dict[str, Any]
        Un dictionnaire contenant les scores suivants :
        - 'Validation MSE' : L'erreur quadratique moyenne (Mean Squared Error).
        - 'Validation R2' : Le coefficient de détermination (R²).
        - 'Validation MAE' : L'erreur absolue moyenne (Mean Absolute Error).
        - 'Validation RMSE' : La racine de l'erreur quadratique moyenne (Root Mean Squared Error).
    """
    # Calcul des métriques sur le jeu de validation
    metrics = {
        'Validation MSE': mean_squared_error(y_val, y_pred),
        'Validation R2': r2_score(y_val, y_pred),
        'Validation MAE': mean_absolute_error(y_val, y_pred),
        'Validation RMSE': np.sqrt(mean_squared_error(y_val, y_pred))
    }

    return metrics

def preprocess_columns(
    X: pd.DataFrame, 
    expected_columns: List[str]
) -> pd.DataFrame:
    """
    Prétraite un DataFrame pour gérer les colonnes supplémentaires et manquantes,
    afin de correspondre aux colonnes attendues.

    Paramètres :
        X (pd.DataFrame) : Le DataFrame d'entrée.
        expected_columns (List[str]) : Liste des noms des colonnes attendues.

    Retourne :
        pd.DataFrame : Un DataFrame contenant uniquement les colonnes attendues, 
        avec les colonnes manquantes ajoutées (remplies avec 0).
    """
    # Identifier les colonnes supplémentaires et manquantes
    extra_columns: Set[str] = set(X.columns) - set(expected_columns)
    missing_columns: Set[str] = set(expected_columns) - set(X.columns)

    print("Colonnes supplémentaires dans X :", extra_columns)
    print("Colonnes manquantes dans X :", missing_columns)

    # Supprimer les colonnes supplémentaires
    X = X.drop(columns=list(extra_columns))
    
    # Ajouter les colonnes manquantes avec des valeurs remplies à 0
    for col in missing_columns:
        X[col] = 0

    # Réorganiser les colonnes pour correspondre à l'ordre attendu
    X = X[expected_columns]
    
    return X





def analyze_data_drift(data_train:pd.DataFrame,data_drift:pd.DataFrame,col: str):
    """
    Analyse la dérive de la colonne spécifiée entre data_train et data_drift.
    Affiche les résultats du test KS pour les variables continues et du Chi-Square pour les catégorielles.
    """
    if data_train[col].dtype == 'object':  # Variable catégorielle
        contingency_table = pd.crosstab(data_train[col], data_drift[col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-Square Test pour '{col}': Chi2 = {chi2}, p-value = {p}")
    else:  # Variable continue
        ks_stat, p_value = ks_2samp(data_train[col], data_drift[col])
        print(f"KS Test pour '{col}': Statistique = {ks_stat}, p-value = {p_value}")


def calculate_jsd_categorical(
    data_train: pd.DataFrame, 
    data_drift: pd.DataFrame, 
    col: str
) -> float:
    """
    Calcule la divergence de Jensen-Shannon pour une colonne catégorielle.

    Arguments :
        data_train (pd.DataFrame) : DataFrame contenant les données d'entraînement.
        data_drift (pd.DataFrame) : DataFrame contenant les données pour détecter un drift.
        col (str) : Nom de la colonne sur laquelle calculer la divergence.

    Retourne :
        float : Valeur de la divergence de Jensen-Shannon.
    """
    # Calculer les distributions de fréquences pour chaque catégorie
    train_freq = data_train[col].value_counts(normalize=True)  # Normaliser pour obtenir des probabilités
    drift_freq = data_drift[col].value_counts(normalize=True)

    # Aligner les catégories pour s'assurer qu'elles correspondent
    all_categories = set(train_freq.index).union(set(drift_freq.index))
    train_dist = np.array([train_freq.get(cat, 0) for cat in all_categories])
    drift_dist = np.array([drift_freq.get(cat, 0) for cat in all_categories])

    # Calculer la divergence de Jensen-Shannon
    js_divergence = jensenshannon(train_dist, drift_dist)
    return float(js_divergence)



def div_jen_shanon(data_train, data_drift, col):
    """
    Calcule la divergence de Jensen-Shannon entre deux distributions pour une colonne donnée.

    Arguments :
        data_train (pd.DataFrame) : DataFrame contenant les données d'entraînement.
        data_drift (pd.DataFrame) : DataFrame contenant les données pour détecter un drift.
        col (str) : Nom de la colonne sur laquelle calculer la divergence.

    Retourne :
        None
    """
    # Calculer les histogrammes pour la colonne dans les deux ensembles de données
    train_dist, _ = np.histogram(data_train[col], bins=20, density=True)
    drift_dist, _ = np.histogram(data_drift[col], bins=20, density=True)
    
    # Calculer la divergence de Jensen-Shannon
    js_divergence = jensenshannon(train_dist, drift_dist)
    
    # Afficher le résultat
    print(f"Divergence de Jensen-Shannon pour `{col}` : {js_divergence}")

def comparer_matrices_correlation(
    data_train: pd.DataFrame,
    data_drift: pd.DataFrame,
    colonnes_numeriques: List[str]
) -> float:
    """
    Compare les matrices de corrélation des variables numériques entre deux ensembles de données.

    Paramètres :
    ----------
    data_train : pd.DataFrame
        DataFrame contenant les données d'entraînement.
    data_drift : pd.DataFrame
        DataFrame contenant les nouvelles données avec une possible dérive.
    colonnes_numeriques : List[str]
        Liste des noms des colonnes numériques à comparer.

    Retour :
    -------
    mse_correlation : float
        Erreur Quadratique Moyenne (MSE) entre les deux matrices de corrélation.

    Visualisations :
    ---------------
    - Matrice de corrélation pour les données d'entraînement.
    - Matrice de corrélation pour les nouvelles données.
    - Différence entre les deux matrices de corrélation.
    """
    # Étape 2 : Calcul des matrices de corrélation
    correlation_train = data_train[colonnes_numeriques].corr()
    correlation_drift = data_drift[colonnes_numeriques].corr()

    # Étape 3 : Calcul de la différence entre les matrices
    correlation_diff = correlation_drift - correlation_train

    # Étape 4 : Calcul de la MSE entre les deux matrices
    mse_correlation = np.mean((correlation_diff.values) ** 2)

    # Étape 5 : Visualisation des matrices de corrélation et de leurs différences
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 5.1 Matrice de corrélation des données d'entraînement
    sns.heatmap(correlation_train, cmap="coolwarm", annot=True, fmt=".2f", center=0, ax=axes[0])
    axes[0].set_title("Matrice de Corrélation (Données d'Entraînement)")

    # 5.2 Matrice de corrélation des nouvelles données
    sns.heatmap(correlation_drift, cmap="coolwarm", annot=True, fmt=".2f", center=0, ax=axes[1])
    axes[1].set_title("Matrice de Corrélation (Nouvelles Données)")

    # 5.3 Différences entre les matrices
    sns.heatmap(correlation_diff, cmap="coolwarm", annot=True, fmt=".2f", center=0, ax=axes[2])
    axes[2].set_title("Différences entre les Matrices de Corrélation")

    # Affichage des graphiques
    plt.suptitle("Comparaison des Matrices de Corrélation", fontsize=16)
    plt.show()

    # Étape 6 : Afficher la MSE
    print(f"MSE entre les deux matrices de corrélation : {mse_correlation:.4f}")
    
    return mse_correlation

