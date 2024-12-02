"""
Fitbit Dataset Preprocessing Pipeline

This script provides a comprehensive preprocessing pipeline for Fitbit health and activity data.
It combines multiple daily-level datasets, aggregates hourly intensities, METs, and heart rate data,
and creates a merged, cleaned dataset ready for analysis.

"""

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi



# === Data Download ===

def download_dataset(dataset, download_path):
    """
    Downloads and unzips a dataset from Kaggle.

    Args:
        dataset (str): The Kaggle dataset identifier (e.g., 'arashnic/fitbit').
        download_path (str): The local directory path where the dataset will be downloaded.
    """

    api = KaggleApi()
    api.authenticate()
    os.makedirs(download_path, exist_ok=True)
    api.dataset_download_files(dataset, path=download_path, unzip=True)


# === Cleaning and Preprocessing Functions ===

def clean_and_standardize(df, time_column=None, date_format=None):
    """
    Standardizes column names and parses specified time columns to datetime objects.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        time_column (str, optional): Column to parse as datetime.
        date_format (str, optional): Date format to use for parsing.

    Returns:
        pd.DataFrame: Cleaned and standardized DataFrame.
    """
    # Rename columns: lowercase and replace spaces with underscores
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Map potential alternate date column names to 'activitydate'
    date_column_mapping = {
        "activityday": "activitydate",
        "date": "activitydate",
        "sleepday": "activitydate",
    }
    df.rename(columns=date_column_mapping, inplace=True)

    # Parse time column to datetime if specified
    if time_column and time_column in df.columns:
        df[time_column] = pd.to_datetime(
            df[time_column], format=date_format, errors="coerce"
        ).dt.normalize()

    return df

def simplify_duplicate_columns(df):
    """
    Simplifies the DataFrame by resolving duplicate columns with suffixes (_x, _y).
    If the columns are identical, keeps one and removes the suffix.

    Args:
        df (pd.DataFrame): The DataFrame to simplify.

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates resolved.
    """
    for col in df.columns:
        if col.endswith('_x'):
            base_col = col[:-2]
            col_x = col
            col_y = f"{base_col}_y"
            if col_y in df.columns:
                if df[col_x].equals(df[col_y]):
                    df.rename(columns={col_x: base_col}, inplace=True)
                    df.drop(columns=[col_y], inplace=True)
    return df


# === Aggregation Functions ===

def aggregate_hourly_data(hourly_intensities):
    """
    Aggregates hourly intensities into average and max by time period.

    Args:
        hourly_intensities (pd.DataFrame): DataFrame with hourly intensities.

    Returns:
        pd.DataFrame: Aggregated DataFrame with average and max intensity per time period.
    """
    # Parse 'ActivityHour' to datetime and extract hour and date
    hourly_intensities['ActivityHour'] = pd.to_datetime(hourly_intensities['ActivityHour'])
    hourly_intensities['Hour'] = hourly_intensities['ActivityHour'].dt.hour
    hourly_intensities['Date'] = hourly_intensities['ActivityHour'].dt.date

    # Create time periods based on hour of the day
    hourly_intensities['TimePeriod'] = pd.cut(
        hourly_intensities['Hour'],
        bins=[-1, 5, 11, 16, 21, 24],
        labels=['night', 'morning', 'afternoon', 'evening', 'late_night']
    )

    # Aggregate hourly data to get average and max intensity per time period
    aggregated = (
        hourly_intensities.groupby(['Id', 'Date', 'TimePeriod'])
        .agg(
            avg_intensity=('TotalIntensity', 'mean'),
            max_intensity=('TotalIntensity', 'max')
        )
        .reset_index()
        .pivot(index=['Id', 'Date'], columns='TimePeriod')
        .reset_index()
    )

    # Flatten MultiIndex columns
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
    aggregated.rename(columns={'Id_': 'id', 'Date_': 'activitydate'}, inplace=True)

    return aggregated

def aggregate_daily_mets(minute_mets):
    """
    Aggregates METs data to get daily average, max, and min METs.

    Args:
        minute_mets (pd.DataFrame): DataFrame with minute-level METs data.

    Returns:
        pd.DataFrame: Daily aggregated METs data.
    """
    minute_mets['ActivityMinute'] = pd.to_datetime(minute_mets['ActivityMinute'])
    minute_mets['Date'] = minute_mets['ActivityMinute'].dt.date
    
    daily_mets = (
        minute_mets.groupby(['Id', 'Date'])
        .agg(
            avg_mets=('METs', 'mean'),
            max_mets=('METs', 'max'),
            min_mets=('METs', 'min')
        )
        .reset_index()
    )
    daily_mets.rename(columns={'Id': 'id', 'Date': 'activitydate'}, inplace=True)
    
    return daily_mets

def aggregate_heart_rate(heart_rate):
    """
    Aggregates heart rate data to get daily average, max, and min heart rate.

    Args:
        heart_rate (pd.DataFrame): DataFrame with heart rate data.

    Returns:
        pd.DataFrame: Daily aggregated heart rate data.
    """
    heart_rate['Time'] = pd.to_datetime(heart_rate['Time'])
    heart_rate['Date'] = heart_rate['Time'].dt.date
    
    heart_rate_daily = (
        heart_rate.groupby(['Id', 'Date'])
        .agg(
            avg_heart_rate=('Value', 'mean'),
            max_heart_rate=('Value', 'max'),
            min_heart_rate=('Value', 'min')
        )
        .reset_index()
    )
    heart_rate_daily.rename(columns={'Id': 'id', 'Date': 'activitydate'}, inplace=True)
    
    return heart_rate_daily

# === Preprocessing Pipeline ===

def preprocessing_pipeline(fitbit_dir, output_path):
    """
    Preprocesses Fitbit datasets by merging, cleaning, and aggregating data.

    Args:
        fitbit_dir (str): Path to the directory containing raw Fitbit data.
        output_path (str): Path to save the preprocessed dataset.
    """
    # List of daily-level datasets
    daily_files = {
        "Sleep Day": "sleepDay_merged.csv",
        "Daily Activity": "dailyActivity_merged.csv",
        "Daily Calories": "dailyCalories_merged.csv",
        "Daily Intensities": "dailyIntensities_merged.csv",
        "Daily Steps": "dailySteps_merged.csv",
    }

    # Step 1: Load and clean all daily datasets
    daily_dataframes = {}
    for dataset_name, file_name in daily_files.items():
        file_path = os.path.join(fitbit_dir, file_name)
        df = pd.read_csv(file_path)
        df_cleaned = clean_and_standardize(df, time_column="activitydate")
        daily_dataframes[dataset_name] = df_cleaned


    # Step 2: Merge all datasets using Sleep Day as the base
    daily_merged = daily_dataframes["Sleep Day"]
    for dataset_name, df in daily_dataframes.items():
        if dataset_name != "Sleep Day":
            daily_merged = pd.merge(
                daily_merged, df, on=["id", "activitydate"], how="outer"
            )

    # Step 3: Simplify duplicate columns
    daily_cleaned = simplify_duplicate_columns(daily_merged)

    # Step 4: Aggregate additional datasets
    hourly_patterns = aggregate_hourly_data(
        pd.read_csv(os.path.join(fitbit_dir, "hourlyIntensities_merged.csv"))
    )
    daily_mets = aggregate_daily_mets(
        pd.read_csv(os.path.join(fitbit_dir, "minuteMETsNarrow_merged.csv"))
    )
    heart_rate_daily = aggregate_heart_rate(
        pd.read_csv(os.path.join(fitbit_dir, "heartrate_seconds_merged.csv"))
    )

    # Step 5: Ensure all date columns are datetime objects
    date_columns = [hourly_patterns, daily_mets, heart_rate_daily, daily_cleaned]
    for df in date_columns:
        df['activitydate'] = pd.to_datetime(df['activitydate'])

    # Step 6: Merge all aggregated datasets
    daily_cleaned = pd.merge(
        daily_cleaned, hourly_patterns, on=['id', 'activitydate'], how='left'
    )
    daily_cleaned = pd.merge(
        daily_cleaned, daily_mets, on=['id', 'activitydate'], how='left'
    )
    daily_cleaned = pd.merge(
        daily_cleaned, heart_rate_daily, on=['id', 'activitydate'], how='left'
    )
    # Step 7: Save final dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_cleaned.to_csv(output_path, index=False)
    

# === Main Execution ===
def main():
    """
    Main execution function for the Fitbit data preprocessing script.
    """
    fitbit_dir = "data/raw/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16"
    output_path = "data/processed/daily_cleaned.csv"
    preprocessing_pipeline(fitbit_dir, output_path)

if __name__ == "__main__":
    main()