import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_ups_dataset(n_samples=15000):
    """
    Generate synthetic UPS dataset with realistic fault scenarios
    """
    data = []
    
    # Define normal operating ranges
    normal_ranges = {
        'Vin': (220, 240),      # Input voltage
        'Vout': (220, 230),     # Output voltage
        'Vbat': (12.0, 13.8),  # Battery voltage
        'Iload': (0.5, 8.0),   # Load current
        'Ichg': (0.0, 2.0),    # Charging current
        'Temp_bat': (20, 35),  # Battery temperature
        'Temp_tx': (25, 50),   # Transformer temperature
        'mains_ok': 1,         # Mains status
        'mode_bat': 0          # Battery mode
    }
    
    # Define fault scenarios
    fault_scenarios = {
        'normal': {
            'weight': 0.6,  # 60% normal operation
            'conditions': normal_ranges
        },
        'battery_low': {
            'weight': 0.08,
            'conditions': {
                'Vin': (220, 240),
                'Vout': (200, 220),
                'Vbat': (10.5, 11.5),  # Low battery
                'Iload': (0.5, 8.0),
                'Ichg': (0.0, 0.2),
                'Temp_bat': (20, 40),
                'Temp_tx': (25, 55),
                'mains_ok': 1,
                'mode_bat': 1  # On battery
            }
        },
        'charging_fault': {
            'weight': 0.06,
            'conditions': {
                'Vin': (220, 240),
                'Vout': (220, 230),
                'Vbat': (11.0, 12.5),
                'Iload': (0.5, 8.0),
                'Ichg': (0.0, 0.1),    # No charging
                'Temp_bat': (20, 45),
                'Temp_tx': (25, 60),
                'mains_ok': 1,
                'mode_bat': 0
            }
        },
        'mains_fail': {
            'weight': 0.08,
            'conditions': {
                'Vin': (0, 180),       # Low/no input voltage
                'Vout': (210, 230),
                'Vbat': (11.5, 13.0),
                'Iload': (0.5, 8.0),
                'Ichg': (0.0, 0.0),
                'Temp_bat': (20, 40),
                'Temp_tx': (25, 55),
                'mains_ok': 0,         # Mains failed
                'mode_bat': 1          # On battery
            }
        },
        'overload': {
            'weight': 0.06,
            'conditions': {
                'Vin': (220, 240),
                'Vout': (200, 220),    # Voltage drop under load
                'Vbat': (12.0, 13.5),
                'Iload': (8.5, 15.0),  # High load current
                'Ichg': (0.0, 1.0),
                'Temp_bat': (30, 50),  # Higher temperature
                'Temp_tx': (50, 80),   # Higher transformer temp
                'mains_ok': 1,
                'mode_bat': 0
            }
        },
        'short_circuit': {
            'weight': 0.04,
            'conditions': {
                'Vin': (220, 240),
                'Vout': (0, 100),      # Very low output
                'Vbat': (10.0, 13.0),
                'Iload': (15.0, 25.0), # Very high current
                'Ichg': (0.0, 0.0),
                'Temp_bat': (40, 70),
                'Temp_tx': (70, 100),  # Very high temp
                'mains_ok': 1,
                'mode_bat': 0
            }
        },
        'transformer_overheat': {
            'weight': 0.08,
            'conditions': {
                'Vin': (220, 240),
                'Vout': (210, 225),
                'Vbat': (12.0, 13.5),
                'Iload': (6.0, 10.0),
                'Ichg': (0.5, 2.0),
                'Temp_bat': (35, 55),
                'Temp_tx': (80, 120),  # Overheated transformer
                'mains_ok': 1,
                'mode_bat': 0
            }
        }
    }
    
    # Calculate number of samples per class
    class_samples = {}
    for fault, config in fault_scenarios.items():
        class_samples[fault] = int(n_samples * config['weight'])
    
    # Adjust to ensure total equals n_samples
    total_assigned = sum(class_samples.values())
    class_samples['normal'] += n_samples - total_assigned
    
    print(f"Generating {n_samples} samples:")
    for fault, count in class_samples.items():
        print(f"  {fault}: {count} samples ({count/n_samples*100:.1f}%)")
    
    # Generate data for each class
    for fault_type, num_samples in class_samples.items():
        conditions = fault_scenarios[fault_type]['conditions']
        
        for _ in range(num_samples):
            sample = {}
            
            # Generate features based on fault conditions
            for feature, value_range in conditions.items():
                if feature in ['mains_ok', 'mode_bat']:
                    # Binary features
                    if isinstance(value_range, (int, float)):
                        sample[feature] = value_range
                    else:
                        sample[feature] = random.choice([0, 1])
                else:
                    # Continuous features with some noise
                    min_val, max_val = value_range
                    base_val = np.random.uniform(min_val, max_val)
                    noise = np.random.normal(0, (max_val - min_val) * 0.05)  # 5% noise
                    sample[feature] = max(0, base_val + noise)
            
            # Add some realistic correlations and constraints
            # Battery voltage should decrease when on battery mode
            if sample['mode_bat'] == 1:
                sample['Vbat'] *= np.random.uniform(0.85, 0.95)
            
            # Output voltage drops with high load
            if sample['Iload'] > 8:
                sample['Vout'] *= np.random.uniform(0.9, 0.98)
            
            # Temperature correlation with load
            if sample['Iload'] > 6:
                sample['Temp_tx'] += np.random.uniform(5, 15)
                sample['Temp_bat'] += np.random.uniform(2, 8)
            
            # Add fault label
            sample['fault_type'] = fault_type
            
            data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Round numerical values for realism
    numerical_cols = ['Vin', 'Vout', 'Vbat', 'Iload', 'Ichg', 'Temp_bat', 'Temp_tx']
    for col in numerical_cols:
        df[col] = df[col].round(2)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating UPS fault detection dataset...")
    dataset = generate_ups_dataset(15000)
    
    # Save to CSV
    dataset.to_csv('ups_synthetic.csv', index=False)
    print(f"\nDataset saved as 'ups_synthetic.csv'")
    print(f"Dataset shape: {dataset.shape}")
    
    # Display class distribution
    print("\nClass distribution:")
    print(dataset['fault_type'].value_counts().sort_index())
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(dataset.head())
    
    # Display basic statistics
    print("\nDataset statistics:")
    print(dataset.describe())
