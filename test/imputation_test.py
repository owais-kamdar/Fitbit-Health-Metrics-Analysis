"""
Unit tests for imputation_pipeline.py

Tests data imputation and feature engineering functionality.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.imputation_pipeline import (
    add_derived_features,
    knn_impute_missing_values,
    imputation_pipeline
)

class TestImputationPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test data with deliberate missing values"""
        self.test_data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'activitydate': ['4/12/2016', '4/13/2016', '4/12/2016', '4/13/2016'],
            'veryactiveminutes': [30.0, np.nan, 45.0, 25.0],
            'fairlyactiveminutes': [45.0, 50.0, np.nan, 40.0],
            'lightlyactiveminutes': [180.0, 160.0, 200.0, np.nan],
            'totalminutesasleep': [360.0, np.nan, 420.0, 380.0],
            'totaltimeinbed': [400.0, 420.0, np.nan, 410.0],
            'min_heart_rate': [55.0, 58.0, np.nan, 54.0],
            'max_heart_rate': [165.0, np.nan, 158.0, 162.0],
            'avg_heart_rate': [72.0, 75.0, 68.0, np.nan]
        })

    def test_add_derived_features(self):
        """Test derived feature calculations"""
        df_with_features = add_derived_features(self.test_data.copy())
        
        # Check if derived columns were created
        expected_columns = [
            'active_minutes',
            'heart_rate_variability',
            'sleep_quality'
        ]
        for col in expected_columns:
            self.assertIn(col, df_with_features.columns)
        
        # Test active_minutes calculation for complete row
        expected_active = 30.0 + 45.0 + 180.0  # Row 0 values
        self.assertEqual(
            df_with_features.loc[0, 'active_minutes'],
            expected_active
        )
        
        # Test heart_rate_variability calculation
        expected_hrv = 165.0 - 55.0  # Row 0 values
        self.assertEqual(
            df_with_features.loc[0, 'heart_rate_variability'],
            expected_hrv
        )
        
        # Test sleep_quality calculation with more flexible assertion
        expected_sleep_quality = (360.0 / (400.0 + 1)) * 100  # Row 0 values, account for +1 in denominator
        self.assertAlmostEqual(
            df_with_features.loc[0, 'sleep_quality'],
            expected_sleep_quality,
            places=2,
            msg=f"Expected {expected_sleep_quality}, got {df_with_features.loc[0, 'sleep_quality']}"
        )

    def test_knn_impute_missing_values(self):
        """Test KNN imputation functionality"""
        features_to_impute = [
            'veryactiveminutes',
            'fairlyactiveminutes',
            'lightlyactiveminutes',
            'totalminutesasleep',
            'totaltimeinbed'
        ]
        
        # Original missing value counts
        original_missing = self.test_data[features_to_impute].isna().sum().sum()
        self.assertGreater(original_missing, 0, "Test data should have missing values")
        
        # Perform imputation
        imputed_df = knn_impute_missing_values(
            self.test_data.copy(),
            features_to_impute,
            n_neighbors=2
        )
        
        # Check if all missing values were imputed
        imputed_missing = imputed_df[features_to_impute].isna().sum().sum()
        self.assertEqual(imputed_missing, 0, "All values should be imputed")
        
        # Check if imputed values are within reasonable ranges
        self.assertTrue(
            all(imputed_df['veryactiveminutes'] >= 0),
            "Imputed active minutes should be non-negative"
        )
        self.assertTrue(
            all(imputed_df['totalminutesasleep'] >= 0),
            "Imputed sleep minutes should be non-negative"
        )

    def test_imputation_pipeline_end_to_end(self):
        """Test the complete imputation pipeline"""
        # Define test features
        test_raw_features = [
            'veryactiveminutes', 'fairlyactiveminutes', 'lightlyactiveminutes',
            'totalminutesasleep', 'totaltimeinbed',
            'min_heart_rate', 'max_heart_rate', 'avg_heart_rate'
        ]
        test_derived_features = [
            'active_minutes', 'heart_rate_variability', 'sleep_quality'
        ]
        
        # Save test data to temporary file
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'test_input.csv')
            output_path = os.path.join(tmpdir, 'test_output.csv')
            
            self.test_data.to_csv(input_path, index=False)
            
            # Run pipeline
            try:
                result_df = imputation_pipeline(
                    input_path,
                    output_path,
                    test_raw_features,
                    test_derived_features
                )
                
                # Verify results
                self.assertIsInstance(result_df, pd.DataFrame)
                
                # Check if all specified features exist and have no missing values
                all_features = test_raw_features + test_derived_features
                for feature in all_features:
                    self.assertIn(feature, result_df.columns)
                    self.assertEqual(
                        result_df[feature].isna().sum(),
                        0,
                        f"Feature {feature} should have no missing values"
                    )
                
            except Exception as e:
                self.fail(f"Pipeline execution failed: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)