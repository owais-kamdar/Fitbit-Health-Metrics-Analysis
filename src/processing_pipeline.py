"""
Fitbit Health Metrics Processing Pipeline

This script implements a comprehensive data processing pipeline for Fitbit health and activity data.
It integrates clinical standards, calculates derived metrics such as sleep quality,
activity intensity, and recovery scores, and prepares the data for advanced analysis and modeling.

"""



import os
import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from typing import Dict, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.clinical_standards import ClinicalStandards


class HealthMetricsPipeline:
    """
    Comprehensive and detailed health metrics processing pipeline.
    This pipeline calculates various health-related metrics, normalizes them,
    and organizes the data for analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the pipeline with a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing raw health data.
        """
        self.df = df.copy()
        self.standards = ClinicalStandards()
        self._validate_required_columns()

    def _validate_required_columns(self):
        """
        Validate the presence of required columns in the DataFrame.
        
        Raises:
            ValueError: If any required columns are missing.
        """
        required_columns = {
            'id', 'activitydate', 'totalsteps', 'totalminutesasleep', 'totaltimeinbed',
            'veryactiveminutes', 'fairlyactiveminutes', 'lightlyactiveminutes',
            'sedentaryminutes', 'calories', 'min_heart_rate', 'max_heart_rate',
            'avg_heart_rate', 'heart_rate_variability'
        }
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def run_pipeline(self):
        """
        Execute the health metrics processing pipeline.
        
        Returns:
            pd.DataFrame: Processed DataFrame with calculated metrics.
        """
        self._preprocess_data()
        self._calculate_sleep_metrics()
        self._calculate_activity_metrics()
        self._calculate_heart_metrics()
        self._calculate_recovery_metrics()
        self._calculate_composite_scores()
        self._derive_risk_flags()
        self._calculate_temporal_metrics()
        self._organize_features()
        return self.df

    # === Preprocessing Methods ===

    def _preprocess_data(self):
        """
        Preprocess and clean data:
        - Convert date columns to datetime.
        - Sort values by 'id' and 'activitydate'.
        """
        self.df['activitydate'] = pd.to_datetime(self.df['activitydate'])
        self.df.sort_values(['id', 'activitydate'], inplace=True)
        self.df['week'] = self.df['activitydate'].dt.isocalendar().week
        self.df['day_of_week'] = self.df['activitydate'].dt.day_name()

    # === Normalization Methods ===

    def _normalize_sleep_quality(self, minutes_asleep, sleep_efficiency):
        """
        Calculate sleep quality score based on duration and efficiency.
        
        Args:
            minutes_asleep (float): Total minutes asleep.
            sleep_efficiency (float): Sleep efficiency percentage.
        
        Returns:
            float: Weighted sleep quality score (0-10).
        """
        # Sleep duration score (0-10)
        if minutes_asleep >= self.standards.SLEEP['duration']['optimal']:
            duration_score = 10
        elif minutes_asleep <= self.standards.SLEEP['duration']['poor']:
            duration_score = 0
        else:
            duration_score = (
                (minutes_asleep - self.standards.SLEEP['duration']['poor']) /
                (self.standards.SLEEP['duration']['optimal'] - self.standards.SLEEP['duration']['poor']) * 10
            )

        # Sleep efficiency score (0-10)
        if sleep_efficiency >= self.standards.SLEEP['efficiency']['excellent']:
            efficiency_score = 10
        elif sleep_efficiency <= self.standards.SLEEP['efficiency']['poor']:
            efficiency_score = 0
        else:
            efficiency_score = (
                (sleep_efficiency - self.standards.SLEEP['efficiency']['poor']) /
                (self.standards.SLEEP['efficiency']['excellent'] - self.standards.SLEEP['efficiency']['poor']) * 10
            )

        # Weighted sleep quality score
        return duration_score * 0.6 + efficiency_score * 0.4

    def _normalize_activity_intensity(self, activity_intensity):
        """
        Calculate activity intensity score based on MET thresholds.
        
        Args:
            activity_intensity (float): Total MET minutes.
        
        Returns:
            float: Normalized activity intensity score (0-10).
        """
        # Convert MET thresholds to MET-minutes
        light = self.standards.ACTIVITY['intensity']['light'] * 30
        moderate = self.standards.ACTIVITY['intensity']['moderate'] * 30
        vigorous = self.standards.ACTIVITY['intensity']['vigorous'] * 30

        # Progressive scoring based on activity levels
        if activity_intensity >= vigorous:
            return 10
        elif activity_intensity <= light:
            return (activity_intensity / light) * 3  # Scale up to 3
        elif activity_intensity <= moderate:
            return 3 + ((activity_intensity - light) / (moderate - light)) * 3  # Scale 3-6
        else:
            return 6 + ((activity_intensity - moderate) / (vigorous - moderate)) * 4  # Scale 6-10

    def _normalize_calories_per_active_minute(self, value):
        """
        Normalize calories per active minute using thresholds.
        """
        light = self.standards.CALORIC_EFFICIENCY['calories_per_active_minute']['light']
        vigorous = self.standards.CALORIC_EFFICIENCY['calories_per_active_minute']['vigorous']

        if value >= vigorous:
            return 10
        elif value <= light:
            return 0
        else:
            return (value - light) / (vigorous - light) * 10

    def _normalize_calories_per_step(self, value):
        """
        Normalize calories per step using thresholds.
        """
        sedentary = self.standards.CALORIC_EFFICIENCY['calories_per_step']['sedentary']
        active = self.standards.CALORIC_EFFICIENCY['calories_per_step']['active']

        if value >= active:
            return 10
        elif value <= sedentary:
            return 0
        else:
            return (value - sedentary) / (active - sedentary) * 10

    def _normalize_heart_rate_reserve(self, value):
        """
        Normalize heart rate reserve using thresholds.
        """
        poor = self.standards.HEART_RATE['reserve']['poor']
        excellent = self.standards.HEART_RATE['reserve']['excellent']

        if value >= excellent:
            return 10
        elif value <= poor:
            return 0
        else:
            return (value - poor) / (excellent - poor) * 10

    def _normalize_hr_variability(self, value):
        """
        Normalize heart rate variability using thresholds.
        """
        low = self.standards.HEART_RATE['variability']['low']
        excellent = self.standards.HEART_RATE['variability']['excellent']

        if value >= excellent:
            return 10
        elif value <= low:
            return 0
        else:
            return (value - low) / (excellent - low) * 10

    def _calculate_sleep_metrics(self):
        """
        Calculate detailed sleep-related metrics, including:
        - Sleep efficiency percentage.
        - Sleep quality score (0-10).
        - Sleep debt and variability metrics.
        """

        # Sleep efficiency (%)
        self.df['sleep_efficiency_%'] = (
            self.df['totalminutesasleep'] / (self.df['totaltimeinbed'] + 1e-5) * 100
        ).clip(0, 100)

        # Sleep quality score (0-10)
        self.df['sleep_quality_score_0_10'] = self.df.apply(
            lambda row: self._normalize_sleep_quality(row['totalminutesasleep'], row['sleep_efficiency_%']),
            axis=1
        )

        # Sleep debt over 7 days
        optimal_sleep = self.standards.SLEEP['duration']['optimal']  # 480 minutes
        self.df['sleep_debt_min_7day'] = self.df.groupby('id')['totalminutesasleep'].transform(
            lambda x: (optimal_sleep - x).clip(lower=0).rolling(7, min_periods=1).sum()
        )

        # Sleep debt score (0-10)
        self.df['sleep_debt_score_0_10'] = self.df['sleep_debt_min_7day'].apply(
            lambda x: 10 if x >= 360 else x / 360 * 10
        )

        # Sleep variability over 7 days
        self.df['sleep_variability_7day'] = self.df.groupby('id')['totalminutesasleep'].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )

        # Sleep variability score (0-10)
        self.df['sleep_variability_7day_score_0_10'] = self.df['sleep_variability_7day'].apply(
            lambda x: 10 if x >= 180 else x / 180 * 10
        )

        # Circadian rhythm disruption score
        self.df['circadian_disruption_score_0_10'] = self.df.apply(
            lambda row: self._calculate_circadian_disruption(
                row['veryactiveminutes'], row['fairlyactiveminutes'], row['lightlyactiveminutes']
            ),
            axis=1
        )

        # Cumulative weekly sleep quality score
        self.df['cumulative_weekly_sleep_quality_score_0_10'] = self.df.groupby(['id', 'week'])[
            'sleep_quality_score_0_10'
        ].transform('sum').apply(
            lambda x: 10 if x >= 70 else x / 70 * 10
        )

    def _calculate_activity_metrics(self):
        """
        Calculate detailed activity-related metrics, including:
        - Activity intensity score (0-10).
        - Steps per active minute.
        - Calories per active minute and per step.
        """
        # Total active minutes
        self.df['active_minutes'] = (
            self.df['veryactiveminutes'] +
            self.df['fairlyactiveminutes'] +
            self.df['lightlyactiveminutes']
        )

        # Activity intensity
        self.df['activity_intensity'] = (
            self.df['veryactiveminutes'] * self.standards.ACTIVITY['intensity']['vigorous'] +
            self.df['fairlyactiveminutes'] * self.standards.ACTIVITY['intensity']['moderate'] +
            self.df['lightlyactiveminutes'] * self.standards.ACTIVITY['intensity']['light']
        )

        # Activity intensity score (0-10)
        self.df['activity_intensity_score_0_10'] = self.df['activity_intensity'].apply(
            self._normalize_activity_intensity
        )

        # Steps per active minute score
        self.df['steps_per_active_minute'] = self.df['totalsteps'] / (self.df['active_minutes']+ 1e-5)
        self.df['steps_per_active_minute_score_0_10'] = self.df['steps_per_active_minute'].apply(
            lambda x: 10 if x >= 50 else (x - 10) / (50 - 10) * 10 if x >= 10 else 0
        )

        # Calories per active minute
        self.df['calories_per_active_minute'] = self.df['calories'] / (self.df['active_minutes']+ 1e-5)
        self.df['calories_per_active_minute_score_0_10'] = self.df['calories_per_active_minute'].apply(
            self._normalize_calories_per_active_minute
        )

        # Calories per step
        self.df['calories_per_step'] = self.df['calories'] / (self.df['totalsteps']+ 1e-5)
        self.df['calories_per_step_score_0_10'] = self.df['calories_per_step'].apply(
            self._normalize_calories_per_step
        )

        # Step variability over 7 days
        self.df['step_variability_7day'] = self.df.groupby('id')['totalsteps'].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )
        self.df['step_variability_7day_score_0_10'] = self.df['step_variability_7day'].apply(
            lambda x: 10 if x >= 3000 else x / 3000 * 10
        )

        # Cumulative weekly steps
        self.df['cumulative_weekly_steps'] = self.df.groupby(['id', 'week'])['totalsteps'].transform('sum')
        self.df['cumulative_weekly_steps_score_0_10'] = self.df['cumulative_weekly_steps'].apply(
            lambda x: 10 if x >= 105000 else (x - 35000) / (105000 - 35000) * 10 if x >= 35000 else 0
        )

    def _calculate_heart_metrics(self):
        """
        Calculate heart-related metrics, including:
        - Heart rate reserve and its score.
        - Heart rate variability score.
        """
        # Heart rate reserve
        self.df['heart_rate_reserve'] = self.df['max_heart_rate'] - self.df['min_heart_rate']

        # Heart rate reserve score
        self.df['heart_rate_reserve_score_0_10'] = self.df['heart_rate_reserve'].apply(
            self._normalize_heart_rate_reserve
        )

        # Heart rate variability score
        self.df['hr_variability_score_0_10'] = self.df['heart_rate_variability'].apply(
            self._normalize_hr_variability
        )

    def _calculate_recovery_metrics(self):
        """
        Calculate recovery-related metrics using:
        - Sleep quality.
        - Heart rate variability.
        - Activity contribution based on intensity.
        """
        # Recovery score with activity contribution based on intensity level
        def calculate_recovery_score(row):
            # Base recovery components
            sleep_component = row['sleep_quality_score_0_10'] * 0.4
            hrv_component = row['hr_variability_score_0_10'] * 0.3
            
            # Dynamic activity contribution based on intensity
            activity_intensity = row['activity_intensity_score_0_10']
            
            if activity_intensity <= 3:  # Low intensity
                activity_component = activity_intensity * 0.1  # Slight positive contribution
            elif activity_intensity <= 7:  # Moderate intensity
                activity_component = activity_intensity * 0.2  # Moderate positive contribution
            else:  # High intensity
                # Diminishing returns for very high intensity
                activity_component = (7 * 0.2) + (activity_intensity - 7) * 0.05
                
            # Rest deficit adjustment
            if row['sleep_debt_min_7day'] > 120:  # More than 2 hours sleep debt
                sleep_penalty = min(row['sleep_debt_min_7day'] / 480, 0.3)  # Max 30% penalty
                recovery_score = (sleep_component + hrv_component + activity_component) * (1 - sleep_penalty)
            else:
                recovery_score = sleep_component + hrv_component + activity_component
                
            return np.clip(recovery_score, 0, 10)

        # Calculate recovery score
        self.df['recovery_score_0_10'] = self.df.apply(calculate_recovery_score, axis=1)

        # Calculate recovery-strain balance with improved formula
        def calculate_recovery_strain_balance(row):
            recovery = row['recovery_score_0_10']
            strain = row['activity_intensity_score_0_10']
            
            if strain == 0:
                return recovery  # If no strain, balance is just recovery
                
            # Balance considers both magnitude and ratio
            balance = (recovery / (strain + 1e-5)) * np.sqrt(recovery)
            return np.clip(balance, 0, 10)

        # Recovery strain balance
        self.df['recovery_strain_balance_score_0_10'] = self.df.apply(
            calculate_recovery_strain_balance, axis=1
        )

    def _calculate_composite_scores(self):
        """
        Calculate composite health and fitness scores using weighted components.
        - Health score: Focus on overall wellbeing.
        - Fitness score: Emphasis on physical performance.
        """
        # Define score components and weights
        score_components = {
            'health': {
                'sleep_quality_score_0_10': 0.35,      # Sleep quality
                'recovery_score_0_10': 0.35,           # Recovery
                'activity_intensity_score_0_10': 0.20,  # Activity
                'hr_variability_score_0_10': 0.10      # Heart rate variability
            },
            'fitness': {
                'heart_rate_reserve_score_0_10': 0.35,  # Cardiovascular fitness
                'activity_intensity_score_0_10': 0.35,  # Activity level
                'recovery_score_0_10': 0.20,            # Recovery capability
                'sleep_quality_score_0_10': 0.10        # Sleep support
            }
        }
    
        # Calculate weighted scores
        for score_type, components in score_components.items():
            score = sum(
                self.df[metric] * weight 
                for metric, weight in components.items()
                if metric in self.df.columns
            )
            # Normalize in case some components are missing
            total_weight = sum(
                weight for metric, weight in components.items() 
                if metric in self.df.columns
            )
            self.df[f'{score_type}_score_0_10'] = (score / total_weight).clip(0, 10)

    def _derive_risk_flags(self):
        """
        Derive cardiovascular disease (CVD) risk metrics:
        - CVD risk score (0-10).
        - Risk category (Low, Moderate, High).
        """
        self.df['cvd_risk_score_0_10'] = (
            (10 - self.df['recovery_score_0_10']) * 0.4 +
            (10 - self.df['health_score_0_10']) * 0.3 +
            (10 - self.df['fitness_score_0_10']) * 0.3
        ).clip(0, 10)

        def classify_cvd_risk(score):
            if score > 7.5:
                return "High Risk"
            elif score > 5.5:
                return "Moderate Risk"
            else:
                return "Low Risk"

        self.df['cvd_risk_category'] = self.df['cvd_risk_score_0_10'].apply(classify_cvd_risk)


    def _calculate_temporal_metrics(self):
        """
        Calculate rolling averages and trends for key metrics over a 7-day window.
        """
        metrics = [
            'totalsteps', 'calories', 'active_minutes',
            'sleep_quality_score_0_10', 'recovery_score_0_10'
        ]
        
        for metric in metrics:
            if metric in self.df.columns:
                # Calculate 7-day rolling average with proper handling of missing values
                rolling_mean = (
                    self.df.groupby('id')[metric]
                    .transform(lambda x: x.rolling(window=7, min_periods=1, center=True).mean())
                )
                # Handle any remaining NaN values
                self.df[f'{metric}_7day_avg'] = (
                    rolling_mean
                    .ffill()  # Forward fill first
                    .bfill()  # Then backward fill any remaining NaNs
                )
                
                # Calculate 7-day trend using robust method
                def calculate_trend(series):
                    if len(series) < 2:
                        return 0
                    
                    # Use exponential weighted mean to reduce noise
                    recent = series.ewm(span=3).mean().iloc[-1]
                    previous = series.ewm(span=3).mean().iloc[0]
                    
                    if previous == 0:
                        return 0
                    
                    trend = ((recent - previous) / previous) * 100
                    return np.clip(trend, -100, 100)  # Clip extreme values
                
                # Apply trend calculation to rolling 7-day windows
                self.df[f'{metric}_7day_trend_%'] = (
                    self.df.groupby('id')[metric]
                    .transform(lambda x: x.rolling(window=7, min_periods=1)
                    .apply(calculate_trend))
                    .fillna(0)  # Fill NA with 0 for trend
                )
    def _calculate_circadian_disruption(self, very_active, fairly_active, lightly_active):
        """
        Calculate circadian rhythm disruption based on activity timing.
        
        Args:
            very_active (float): Very active minutes.
            fairly_active (float): Fairly active minutes.
            lightly_active (float): Lightly active minutes.
        
        Returns:
            float: Circadian disruption score (0-10).
        """
        disruption = abs(very_active * 0.4 - fairly_active * 0.3) + \
                     abs(fairly_active * 0.3 - lightly_active * 0.5)
        return 10 if disruption >= 200 else disruption / 200 * 10

    def _organize_features(self):
        """
        Organize features into logical groups for clarity in the final DataFrame.
        """
        column_order = [
            # Identifiers
            'id', 'activitydate', 'week', "day_of_week",

            # Sleep metrics
            'totalminutesasleep', 'totaltimeinbed', 'sleep_efficiency_%',
            'sleep_quality_score_0_10', 'sleep_debt_min_7day', 'sleep_debt_score_0_10',
            'sleep_variability_7day', 'sleep_variability_7day_score_0_10',
            'circadian_disruption_score_0_10', 'cumulative_weekly_sleep_quality_score_0_10',

            # Activity metrics
            'totalsteps', 'active_minutes', 'activity_intensity', 'activity_intensity_score_0_10',
            'steps_per_active_minute', 'steps_per_active_minute_score_0_10',
            'calories', 'calories_per_active_minute', 'calories_per_active_minute_score_0_10',
            'calories_per_step', 'calories_per_step_score_0_10',
            'step_variability_7day', 'step_variability_7day_score_0_10',
            'cumulative_weekly_steps', 'cumulative_weekly_steps_score_0_10',

            # Heart metrics
            'min_heart_rate', 'max_heart_rate', 'avg_heart_rate',
            'heart_rate_reserve', 'heart_rate_reserve_score_0_10',
            'heart_rate_variability', 'hr_variability_score_0_10',

            # Temporal metrics
            'totalsteps_7day_avg', 'totalsteps_7day_trend_%',
            'calories_7day_avg', 'calories_7day_trend_%',
            'active_minutes_7day_avg', 'active_minutes_7day_trend_%',
            'sleep_quality_score_0_10_7day_avg', 'sleep_quality_score_0_10_7day_trend_%',
            'recovery_score_0_10_7day_avg', 'recovery_score_0_10_7day_trend_%',

            # Recovery and composite scores
            'recovery_score_0_10', 'recovery_strain_balance_score_0_10',
            'health_score_0_10', 'fitness_score_0_10',
            'cvd_risk_score_0_10', 'cvd_risk_category',
        ]
        # Ensure all columns exist before reordering
        existing_columns = [col for col in column_order if col in self.df.columns]
        self.df = self.df[existing_columns]


if __name__ == "__main__":
    # File paths
    input_file = "data/processed/df_imputed.csv"
    output_file = "data/processed/df_processed.csv"

    # Load the imputed dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")

    df_imputed = pd.read_csv(input_file)

    # Initialize and run the pipeline
    pipeline = HealthMetricsPipeline(df_imputed)
    processed_df = pipeline.run_pipeline()

    # Save the processed dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    print(f"Processed health metrics saved to {output_file}")
