"""
Fitbit Dataset Imputation Pipeline

This script provides a comprehensive pipeline for imputing missing values in Fitbit datasets.
It includes derived feature engineering, KNN-based imputation, and visualization of changes before and after imputation.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# === Feature Engineering Functions ===

def add_derived_features(df):
    """
    Adds derived features to the DataFrame.

    Derived Features:
    - active_minutes: Sum of very, fairly, and lightly active minutes.
    - heart_rate_variability: Difference between max and min heart rate.
    - sleep_quality: Percentage of total sleep time relative to time in bed.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional derived features.
    """
    df['active_minutes'] = (
        df['veryactiveminutes'] + df['fairlyactiveminutes'] + df['lightlyactiveminutes']
    )
    df['heart_rate_variability'] = df['max_heart_rate'] - df['min_heart_rate']
    df['sleep_quality'] = (
        df['totalminutesasleep'] / (df['totaltimeinbed'].replace(0, pd.NA) + 1)
    ) * 100  # Avoid division by zero
    return df


# === Imputation Functions ===

def knn_impute_missing_values(df, features, n_neighbors=25):
    """
    Imputes missing values using KNNImputer for the specified features.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.
        features (list): List of feature names to impute.
        n_neighbors (int): Number of neighbors to use for imputation.

    Returns:
        pd.DataFrame: DataFrame with imputed values for the specified features.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[features] = imputer.fit_transform(df[features])
    return df


# === Visualization Functions ===

def plot_distributions(df_before, df_after, features, title_prefix, bins=20):
    """
    Plots the distributions of features before and after imputation.

    Args:
        df_before (pd.DataFrame): DataFrame before imputation.
        df_after (pd.DataFrame): DataFrame after imputation.
        features (list): List of feature names to plot.
        title_prefix (str): Prefix for the plot title.
        bins (int): Number of bins for the histograms.

    Returns:
        None
    """
    num_features = len(features)
    rows = (num_features + 2) // 3  # 3 plots per row
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(
            df_before[feature].dropna(), bins=bins, kde=True, color="blue", label="Before", ax=ax
        )
        sns.histplot(
            df_after[feature].dropna(), bins=bins, kde=True, color="orange", label="After", ax=ax
        )
        ax.set_title(f"{title_prefix}: {feature}")
        ax.legend()

    # Turn off extra axes
    for ax in axes[len(features):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# === Imputation Pipeline ===

def imputation_pipeline(input_path, output_path, raw_features, derived_features, n_neighbors=25):
    """
    Performs the imputation pipeline:
    - Loads the dataset from input_path.
    - Adds derived features.
    - Imputes missing values using KNNImputer.
    - Plots distributions before and after imputation.
    - Saves the imputed dataset to output_path.

    Args:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the imputed dataset.
        features_for_imputation (list): List of features to impute.
        n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: Fully processed and imputed DataFrame.
    """
    # Step 1: Load the cleaned dataset
    df = pd.read_csv(input_path)

    # Step 2: Add derived features
    df = add_derived_features(df)

    # Step 3: Save a copy before imputation for comparison
    df_before_imputation = df.copy()

    # Step 4: Impute missing values for raw features
    df = knn_impute_missing_values(df, raw_features, n_neighbors=n_neighbors)

    # Step 5: Recalculate derived features after raw feature imputation
    df = add_derived_features(df)

    # Step 6: Identify missing features after imputation
    print("Validating missing values after imputation...")
    missing_features_after = df.columns[df.isnull().any()].tolist()
    if not missing_features_after:
        print("No missing values detected after imputation.")
    else:
        print(f"Warning: Missing features after imputation: {missing_features_after}")

    # Step 7: Plot distributions before and after imputation
    print("Plotting feature distributions before and after imputation...")
    plot_distributions(df_before_imputation, df, raw_features + derived_features, "Feature Distributions")


    # Step 8: Summary statistics
    print("\n===== Summary Statistics =====")
    print(df.describe())

    # Step 9: Save the imputed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Imputed dataset saved to {output_path}")

    return df



# === Main Execution ===

if __name__ == "__main__":
    # Define file paths
    daily_cleaned_path = "data/processed/daily_cleaned.csv"
    df_imputed_path = "data/processed/df_imputed.csv"

    # Define raw and derived features
    raw_features = [
        'totalsleeprecords', 'totalminutesasleep', 'totaltimeinbed', 
        'avg_intensity_night', 'avg_intensity_morning', 'avg_intensity_afternoon', 
        'avg_intensity_evening', 'avg_intensity_late_night', 'max_intensity_night', 
        'max_intensity_morning', 'max_intensity_afternoon', 'max_intensity_evening', 
        'max_intensity_late_night', 'avg_mets', 'max_mets', 'min_mets', 
        'avg_heart_rate', 'max_heart_rate', 'min_heart_rate', 
        'veryactiveminutes', 'fairlyactiveminutes', 'lightlyactiveminutes'
    ]

    derived_features = [
        'active_minutes', 'heart_rate_variability', 'sleep_quality'
    ]

    # Run the imputation pipeline
    imputation_pipeline(daily_cleaned_path, df_imputed_path, raw_features, derived_features)
