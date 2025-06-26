import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


def add_interaction_feature(df: DataFrame) -> DataFrame:
    """
    Add an interaction feature between driver age and vehicle age.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: DataFrame with the new interaction feature.
    """
    df["Driver_Vehicle_Age_Interaction"] = (
        df["AgeConducteur_log"] * df["AgeVehicule_log"]
    )
    return df


def bin_age_conducteur(df: DataFrame) -> DataFrame:
    """
    Bin the 'AgeConducteur_log' column into three categories: 'Young', 'Mature', and '
    Senior'.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: DataFrame with a new binned age column.
    """
    df["AgeConducteur_Binned"] = pd.cut(
        df["AgeConducteur_log"],
        bins=[-float("inf"), pd.np.log1p(25), pd.np.log1p(50), float("inf")],
        labels=["Young", "Mature", "Senior"],
    )
    return df


def create_bonus_age_difference(df: DataFrame) -> DataFrame:
    """
    Create a new feature by subtracting driver age from BonusMalus.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: DataFrame with the new difference feature.
    """
    df["Bonus_Age_Difference"] = df["BonusMalus_log"] - df["AgeConducteur_log"]
    return df


def encode_binned_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'AgeConducteur_Binned' column.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: DataFrame with one-hot encoded binned age columns.
    """
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_columns = encoder.fit_transform(df[["AgeConducteur_Binned"]])
    encoded_df = pd.DataFrame(
        encoded_columns, columns=encoder.get_feature_names_out(["AgeConducteur_Binned"])
    )
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=["AgeConducteur_Binned"], inplace=True)
    return df


def add_polynomial_features(df: DataFrame, columns: list, degree: int = 2) -> DataFrame:
    """
    Generate polynomial features for the specified columns.

    Args:
        df (DataFrame): The input DataFrame.
        columns (list): List of column names to generate polynomial features for.
        degree (int): The degree of the polynomial features.

    Returns:
        DataFrame: DataFrame with added polynomial features.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[columns])
    poly_feature_names = poly.get_feature_names_out(columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = pd.concat([df, poly_df], axis=1)
    return df


def add_feature_interaction(
    df: DataFrame, input_features: list, cats: list
) -> DataFrame:
    """
    Add interaction features between specified input features and categorical variables.

    Args:
        df (DataFrame): The input DataFrame.
        input_features (list): List of input feature names.
        cats (list): List of categorical variable names.

    Returns:
        DataFrame: DataFrame with added interaction features.
    """
    for cat in cats:
        for feature in input_features:
            df[f"{feature}_x_{cat}"] = df[feature] * df[cat]
    return df
