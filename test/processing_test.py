"""
Unit tests for processing_pipeline.py

Tests health metrics calculations and processing functionality.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.processing_pipeline import HealthMetricsPipeline

class TestHealthMetricsPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test data with realistic values"""
        self.test_data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'activitydate': pd.date_range(start='2024-01-01', periods=4),
            'totalsteps': [8000, 10000, 7500, 9500],
            'totalminutesasleep': [420, 380, 440, 400],
            'totaltimeinbed': [480, 420, 500, 450],
            'veryactiveminutes': [30, 45, 20, 35],
            'fairlyactiveminutes': [60, 75, 45, 65],
            'lightlyactiveminutes': [180, 160, 200, 170],
            'sedentaryminutes': [720, 700, 750, 710],
            'calories': [2200, 2400, 2000, 2300],
            'min_heart_rate': [55, 58, 52, 56],
            'max_heart_rate': [165, 172, 158, 168],
            'avg_heart_rate': [72, 75, 68, 73],
            'heart_rate_variability': [45, 42, 48, 44]
        })
        
        # Ensure all numeric columns are float64 to prevent type issues
        numeric_columns = self.test_data.select_dtypes(include=[np.number]).columns
        self.test_data[numeric_columns] = self.test_data[numeric_columns].astype(np.float64)
        
        self.pipeline = HealthMetricsPipeline(self.test_data)

    def test_temporal_metrics(self):
        """Test temporal metric calculations"""
        processed_df = self.pipeline.run_pipeline()
        
        # Check required temporal metrics exist
        temporal_metrics = [
            'totalsteps_7day_avg',
            'calories_7day_avg',
            'sleep_quality_score_0_10_7day_avg'
        ]
        for metric in temporal_metrics:
            self.assertIn(metric, processed_df.columns)
        
        # Verify moving averages exist and are within reasonable ranges
        for idx, row in processed_df.iterrows():
            self.assertFalse(
                pd.isna(row['totalsteps_7day_avg']),
                f"Row {idx}: Step average should not be NaN"
            )
            self.assertGreaterEqual(
                row['totalsteps_7day_avg'],
                0,
                f"Row {idx}: Step average {row['totalsteps_7day_avg']} should be non-negative"
            )
            self.assertFalse(
                pd.isna(row['calories_7day_avg']),
                f"Row {idx}: Calorie average should not be NaN"
            )
            self.assertGreaterEqual(
                row['calories_7day_avg'],
                0,
                f"Row {idx}: Calorie average {row['calories_7day_avg']} should be non-negative"
            )
        
        # Verify averages are within reasonable ranges of the original data
        self.assertTrue(
            all(processed_df['totalsteps_7day_avg'] >= processed_df['totalsteps'].min() * 0.5),
            "Step averages should not be less than 50% of minimum steps"
        )
        self.assertTrue(
            all(processed_df['totalsteps_7day_avg'] <= processed_df['totalsteps'].max() * 1.5),
            "Step averages should not exceed 150% of maximum steps"
        )

    def test_activity_metrics(self):
        """Test activity-related metric calculations"""
        processed_df = self.pipeline.run_pipeline()
        
        # Check required activity metrics exist
        activity_metrics = [
            'activity_intensity_score_0_10',
            'steps_per_active_minute',
            'calories_per_active_minute',
            'calories_per_step'
        ]
        for metric in activity_metrics:
            self.assertIn(metric, processed_df.columns)
        
        # Test activity calculations
        row_idx = 0
        total_active = (
            self.test_data.loc[row_idx, 'veryactiveminutes'] +
            self.test_data.loc[row_idx, 'fairlyactiveminutes'] +
            self.test_data.loc[row_idx, 'lightlyactiveminutes']
        )
        expected_steps_per_minute = self.test_data.loc[row_idx, 'totalsteps'] / total_active
        
        self.assertAlmostEqual(
            processed_df.loc[row_idx, 'steps_per_active_minute'],
            expected_steps_per_minute,
            places=1,
            msg=f"Expected {expected_steps_per_minute}, got {processed_df.loc[row_idx, 'steps_per_active_minute']}"
        )

    def test_heart_metrics(self):
        """Test heart rate metric calculations"""
        processed_df = self.pipeline.run_pipeline()
        
        # Check required heart metrics exist
        heart_metrics = [
            'heart_rate_reserve_score_0_10',
            'hr_variability_score_0_10'
        ]
        for metric in heart_metrics:
            self.assertIn(metric, processed_df.columns)
        
        # Test heart rate reserve calculation
        row_idx = 0
        expected_reserve = (
            self.test_data.loc[row_idx, 'max_heart_rate'] -
            self.test_data.loc[row_idx, 'min_heart_rate']
        )
        
        self.assertAlmostEqual(
            processed_df.loc[row_idx, 'heart_rate_reserve'],
            expected_reserve,
            places=1,
            msg=f"Expected {expected_reserve}, got {processed_df.loc[row_idx, 'heart_rate_reserve']}"
        )

    def test_sleep_metrics(self):
        """Test sleep-related metric calculations"""
        processed_df = self.pipeline.run_pipeline()
        
        # Check required sleep metrics exist
        sleep_metrics = [
            'sleep_efficiency_%',
            'sleep_quality_score_0_10',
            'sleep_debt_min_7day',
            'sleep_variability_7day'
        ]
        for metric in sleep_metrics:
            self.assertIn(metric, processed_df.columns)
        
        # Test sleep efficiency calculation
        row_idx = 0
        expected_efficiency = (
            self.test_data.loc[row_idx, 'totalminutesasleep'] /
            self.test_data.loc[row_idx, 'totaltimeinbed'] * 100
        )
        
        self.assertAlmostEqual(
            processed_df.loc[row_idx, 'sleep_efficiency_%'],
            expected_efficiency,
            places=1,
            msg=f"Expected {expected_efficiency}, got {processed_df.loc[row_idx, 'sleep_efficiency_%']}"
        )

    def test_composite_scores(self):
        """Test composite health and fitness score calculations"""
        processed_df = self.pipeline.run_pipeline()
        
        # Check required composite scores exist
        composite_scores = [
            'health_score_0_10',
            'fitness_score_0_10',
            'recovery_score_0_10',
            'cvd_risk_score_0_10'
        ]
        for score in composite_scores:
            self.assertIn(score, processed_df.columns)
            self.assertTrue(
                all(0 <= s <= 10 for s in processed_df[score]),
                f"{score} should be between 0 and 10"
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)