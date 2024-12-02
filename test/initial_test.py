"""
Unit tests for initial_pipeline.py

Tests the data loading, cleaning, and preprocessing functionality.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.initial_pipeline import (
    clean_and_standardize,
    simplify_duplicate_columns,
    aggregate_hourly_data,
    aggregate_daily_mets,
    aggregate_heart_rate
)

class TestInitialPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Sample daily activity data with proper date format
        self.daily_data = pd.DataFrame({
            'Id': [1, 2],
            'ActivityDate': ['4/12/2016', '4/13/2016'],  # Match your data format
            'Total Steps': [10000, 12000],
            'Calories': [2000, 2200]
        })
        
        # Sample hourly data
        self.hourly_data = pd.DataFrame({
            'Id': [1, 1, 1, 2],
            'ActivityHour': [
                '4/12/2016 10:00:00',
                '4/12/2016 11:00:00',
                '4/12/2016 14:00:00',
                '4/12/2016 10:00:00'
            ],
            'TotalIntensity': [20, 30, 25, 15]
        })
        
        # Sample METs data
        self.mets_data = pd.DataFrame({
            'Id': [1, 1, 2],
            'ActivityMinute': [
                '4/12/2016 10:00:00',
                '4/12/2016 10:01:00',
                '4/12/2016 10:00:00'
            ],
            'METs': [4.5, 5.0, 3.5]
        })
        
        # Sample heart rate data
        self.heart_rate_data = pd.DataFrame({
            'Id': [1, 1, 2],
            'Time': [
                '4/12/2016 10:00:00',
                '4/12/2016 10:01:00',
                '4/12/2016 10:00:00'
            ],
            'Value': [75, 80, 70]
        })

    def test_clean_and_standardize(self):
        """Test data cleaning and standardization"""
        cleaned_df = clean_and_standardize(
            self.daily_data.copy(),
            time_column='ActivityDate',
            date_format='%m/%d/%Y'  # Specify the date format
        )
        
        # Check column names are standardized
        self.assertIn('activitydate', cleaned_df.columns)
        self.assertIn('total_steps', cleaned_df.columns)
        
        # Check date parsing
        try:
            pd.to_datetime(cleaned_df['activitydate'])
            date_parsing_successful = True
        except:
            date_parsing_successful = False
        self.assertTrue(date_parsing_successful, "Date parsing should succeed")
        
        # Check Id is lowercase
        self.assertIn('id', cleaned_df.columns)
        
        # Print debug information if test fails
        if not date_parsing_successful:
            print("Debug info:")
            print("Original activitydate values:", self.daily_data['ActivityDate'].tolist())
            print("Cleaned activitydate values:", cleaned_df['activitydate'].tolist())

    def test_simplify_duplicate_columns(self):
        """Test duplicate column resolution"""
        # Create test data with duplicate columns
        df_with_dupes = self.daily_data.copy()
        df_with_dupes['steps_x'] = [10000, 12000]
        df_with_dupes['steps_y'] = [10000, 12000]
        
        simplified_df = simplify_duplicate_columns(df_with_dupes)
        
        # Check duplicates are resolved
        self.assertNotIn('steps_x', simplified_df.columns)
        self.assertNotIn('steps_y', simplified_df.columns)
        
        # If values were identical, should have single column
        if 'steps' in simplified_df.columns:
            self.assertEqual(simplified_df['steps'].tolist(), [10000, 12000])

    def test_aggregate_hourly_data(self):
        """Test hourly data aggregation"""
        aggregated = aggregate_hourly_data(self.hourly_data)
        
        # Check output structure
        self.assertIn('id', aggregated.columns)
        self.assertIn('activitydate', aggregated.columns)
        
        # Check time period aggregations exist
        time_periods = ['morning', 'afternoon', 'evening', 'night', 'late_night']
        for period in time_periods:
            self.assertTrue(
                any(period in col for col in aggregated.columns)
            )

    def test_aggregate_daily_mets(self):
        """Test METs aggregation"""
        daily_mets = aggregate_daily_mets(self.mets_data)
        
        # Check required columns exist
        self.assertIn('avg_mets', daily_mets.columns)
        self.assertIn('max_mets', daily_mets.columns)
        self.assertIn('min_mets', daily_mets.columns)
        
        # Verify calculations
        self.assertEqual(daily_mets['max_mets'].iloc[0], 5.0)
        self.assertEqual(daily_mets['min_mets'].iloc[0], 4.5)

    def test_aggregate_heart_rate(self):
        """Test heart rate aggregation"""
        heart_rate_daily = aggregate_heart_rate(self.heart_rate_data)
        
        # Check required columns exist
        self.assertIn('avg_heart_rate', heart_rate_daily.columns)
        self.assertIn('max_heart_rate', heart_rate_daily.columns)
        self.assertIn('min_heart_rate', heart_rate_daily.columns)
        
        # Verify calculations
        self.assertEqual(heart_rate_daily['max_heart_rate'].iloc[0], 80)
        self.assertEqual(heart_rate_daily['min_heart_rate'].iloc[0], 75)

if __name__ == '__main__':
    unittest.main(verbosity=2)