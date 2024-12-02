# clinical_standards.py
from dataclasses import dataclass

@dataclass
class ClinicalStandards:
    """
    Comprehensive clinical standards for health metrics with accurate thresholds.
    These standards are used for normalizing and interpreting health data.

    Sources:
        - HEART_RATE: American Heart Association (AHA), National Institutes of Health (NIH)
        - ACTIVITY: World Health Organization (WHO), American College of Sports Medicine (ACSM)
        - SLEEP: National Sleep Foundation (NSF), Centers for Disease Control and Prevention (CDC)
        - RECOVERY: WHOOP Labs, American Journal of Sports Medicine (AJSM)
        - CALORIC_EFFICIENCY: Compendium of Physical Activities (CPA)
    """
    HEART_RATE = {
        'zones': {  # Source: AHA, NIH
            'bradycardia': 60,   # Resting HR < 60 bpm
            'normal_low': 60,    # Normal range lower bound
            'normal_high': 100,  # Normal range upper bound
            'tachycardia': 100   # Resting HR > 100 bpm
        },
        'variability': {  # Source: WHOOP Labs, NIH
            'low': 20,
            'moderate': 40,
            'good': 60,
            'excellent': 80
        },
        'reserve': {  # Source: AHA
            'poor': 50,    # Adjusted for realism
            'fair': 75,
            'good': 100,
            'excellent': 125
        }
    }

    ACTIVITY = {
        'steps': {  # Source: WHO, ACSM
            'sedentary': 5000,
            'low': 7500,
            'moderate': 10000,
            'active': 12500,
            'highly_active': 15000
        },
        'intensity': {  # Source: CPA
            'light': 1.5,
            'moderate': 3.0,
            'vigorous': 6.0
        }
    }

    SLEEP = {
        'duration': {  # Source: NSF, CDC
            'poor': 360,
            'minimum': 420,
            'optimal': 480,
            'maximum': 540
        },
        'efficiency': {  # Source: NSF
            'poor': 75,
            'good': 90,
            'excellent': 95
        }
    }

    RECOVERY = {
        'strain': {  # Source: WHOOP Labs
            'low': 25,
            'moderate': 50,
            'high': 75,
            'extreme': 90
        },
        'scores': {  # Source: WHOOP Labs
            'poor': 33,
            'fair': 66,
            'good': 85,
            'excellent': 95
        }
    }

    CALORIC_EFFICIENCY = {
        'calories_per_step': {  # Source: CPA
            'sedentary': 0.1,
            'moderate': 0.2,
            'active': 0.35
        },
        'calories_per_active_minute': {  # Source: CPA
            'light': 3.5,
            'moderate': 5.0,
            'vigorous': 7.5
        }
    }
