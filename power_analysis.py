"""
Power Analysis for Health Metrics Dataset
Analyzes statistical power for key health metric relationships
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower, FTestPower
import pandas as pd
import matplotlib.pyplot as plt

def calculate_sample_sizes(effect_sizes=[0.2, 0.5, 0.8], power=0.8, alpha=0.05):
    """
    Calculate required sample sizes for different effect sizes.
    
    Args:
        effect_sizes (list): List of Cohen's d effect sizes to analyze. Default [0.2, 0.5, 0.8]
            representing small, medium, and large effects.
        power (float): Desired statistical power (1 - β). Default 0.8 (80% power).
        alpha (float): Significance level (Type I error rate). Default 0.05 (5%).
    
    Returns:
        dict: Required sample sizes for each effect size.
            Keys are effect sizes, values are required sample sizes.
    """
    analysis = TTestPower()
    sample_sizes = {}
    
    for effect in effect_sizes:
        n = analysis.solve_power(effect_size=effect, power=power, alpha=alpha)
        sample_sizes[effect] = int(np.ceil(n))
    
    return sample_sizes

def analyze_existing_power(df, column1, column2, alpha=0.05):
    """
    Analyze statistical power between two groups in a dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column1 (str): Column name for the first group.
        column2 (str): Column name for the second group.
        alpha (float, optional): Significance level (default is 0.05).

    Returns:
        dict: Dictionary with keys 'effect_size', 'sample_size', and 'achieved_power'.
    """
    group1 = df[column1].dropna()
    group2 = df[column2].dropna()
    
    effect_size = np.abs(group1.mean() - group2.mean()) / \
                 np.sqrt((group1.var() + group2.var()) / 2)
    
    analysis = TTestPower()
    power = analysis.solve_power(
        effect_size=effect_size,
        nobs=min(len(group1), len(group2)),
        alpha=alpha
    )
    
    return {
        'effect_size': effect_size,
        'sample_size': min(len(group1), len(group2)),
        'achieved_power': power
    }

def run_power_analysis(df):
    """
    Perform power analysis for predefined metric pairs in the dataset.

    Args:
        df (pd.DataFrame): Dataset containing the metrics to analyze.

    Returns:
        dict: Contains:
            - 'required_sample_sizes': Calculated sample sizes for different effect sizes.
            - 'actual_power_analysis': Achieved power for each metric pair.
            - 'total_samples': Total number of samples in the dataset.
    """
    # Required sample sizes for different effect sizes
    sample_sizes = calculate_sample_sizes()
    
    # More relevant metric pairs based on health relationships
    metric_pairs = [
        # Activity relationships
        ('totalsteps', 'calories'),
        ('veryactiveminutes', 'calories'),
        ('totalsteps', 'veryactivedistance'),
        
        # Sleep patterns
        ('totalminutesasleep', 'totaltimeinbed'),
        ('totalminutesasleep', 'sedentaryminutes'),
        
        # Heart rate patterns
        ('avg_heart_rate', 'veryactiveminutes'),
        ('heart_rate_variability', 'avg_heart_rate'),
        
        # METs relationships
        ('avg_mets', 'calories'),
        ('max_mets', 'veryactiveminutes')
    ]
    
    power_results = {}
    for col1, col2 in metric_pairs:
        if col1 in df.columns and col2 in df.columns:
            power_results[f'{col1}_vs_{col2}'] = analyze_existing_power(df, col1, col2)
    
    return {
        'required_sample_sizes': sample_sizes,
        'actual_power_analysis': power_results,
        'total_samples': len(df)
    }

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/processed/daily_cleaned.csv")
    
    # Run power analysis
    results = run_power_analysis(df)
    
    print("\nPower Analysis Results:")
    print(f"\nTotal samples in dataset: {results['total_samples']}")
    
    print("\nRequired sample sizes for different effect sizes (power=0.8, alpha=0.05):")
    for effect, size in results['required_sample_sizes'].items():
        print(f"Effect size {effect}: {size} samples required")
    
    print("\nActual power analysis results:")
    for comparison, result in results['actual_power_analysis'].items():
        print(f"\n{comparison}:")
        print(f"Effect size: {result['effect_size']:.3f}")
        print(f"Sample size: {result['sample_size']}")
        print(f"Achieved power: {result['achieved_power']:.3f}")