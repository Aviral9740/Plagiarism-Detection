import pandas as pd
import numpy as np
from collections import defaultdict


def generate_plagiarism_dataset(num_entries=1200):
    """
    Generate a plagiarism detection dataset with realistic patterns and outliers
    """
    np.random.seed(42)  # For reproducibility

    # Base patterns from the original data
    groups = ['g0p', 'g1p', 'g2p', 'g3p', 'g4p', 'g5p', 'g6p', 'g7p', 'g8p', 'g9p']
    participants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    tasks = ['a', 'b', 'c', 'd', 'e']
    categories = ['non', 'cut', 'light', 'heavy']

    # Category probabilities (weighted distribution)
    cat_probs = {
        'non': 0.4,  # 40% non-plagiarized
        'cut': 0.25,  # 25% cut (low plagiarism)
        'light': 0.2,  # 20% light plagiarism
        'heavy': 0.15  # 15% heavy plagiarism
    }

    # Task-specific tendencies (which tasks tend to have more plagiarism)
    task_tendencies = {
        'a': {'non': 0.5, 'cut': 0.3, 'light': 0.15, 'heavy': 0.05},
        'b': {'non': 0.4, 'cut': 0.25, 'light': 0.2, 'heavy': 0.15},
        'c': {'non': 0.3, 'cut': 0.3, 'light': 0.25, 'heavy': 0.15},
        'd': {'non': 0.35, 'cut': 0.25, 'light': 0.2, 'heavy': 0.2},
        'e': {'non': 0.45, 'cut': 0.25, 'light': 0.2, 'heavy': 0.1}
    }

    data = []

    # Generate regular entries
    for i in range(num_entries):
        group = np.random.choice(groups)
        participant = np.random.choice(participants)
        task = np.random.choice(tasks)

        # Create filename
        filename = f"{group}{participant}_task{task}.txt"

        # Determine category based on task tendencies with some randomness
        rand_val = np.random.random()
        cumulative_prob = 0

        # Use task-specific probabilities with some noise
        probs = task_tendencies[task].copy()
        # Add some randomness to probabilities
        for cat in probs:
            probs[cat] *= np.random.uniform(0.8, 1.2)
        # Normalize
        total = sum(probs.values())
        for cat in probs:
            probs[cat] /= total

        category = None
        for cat, prob in probs.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                category = cat
                break

        # Ensure we always get a category
        if category is None:
            category = np.random.choice(categories, p=[cat_probs[cat] for cat in categories])

        data.append({
            'File': filename,
            'Task': task,
            'Category': category
        })

    # Add some outliers (5% of data)
    num_outliers = int(num_entries * 0.05)
    outlier_indices = np.random.choice(len(data), num_outliers, replace=False)

    for idx in outlier_indices:
        # Outliers have unusual category distributions
        if np.random.random() < 0.7:
            # Extreme cases: either very high or very low plagiarism
            if np.random.random() < 0.5:
                # High plagiarism outlier
                data[idx]['Category'] = np.random.choice(['heavy', 'heavy', 'light'], p=[0.7, 0.2, 0.1])
            else:
                # Low plagiarism outlier
                data[idx]['Category'] = np.random.choice(['non', 'non', 'cut'], p=[0.8, 0.15, 0.05])
        else:
            # Random unusual pattern
            data[idx]['Category'] = np.random.choice(categories, p=[0.1, 0.1, 0.1, 0.7])

    # Add some pattern-based outliers (specific groups with unusual behavior)
    problematic_groups = ['g5p', 'g8p']
    for entry in data:
        if any(p_group in entry['File'] for p_group in problematic_groups):
            if np.random.random() < 0.6:  # 60% chance of unusual pattern
                if 'g5p' in entry['File']:
                    # g5p group tends to have more heavy plagiarism
                    entry['Category'] = np.random.choice(categories, p=[0.1, 0.2, 0.3, 0.4])
                elif 'g8p' in entry['File']:
                    # g8p group tends to have very little plagiarism
                    entry['Category'] = np.random.choice(categories, p=[0.7, 0.2, 0.1, 0.0])

    return pd.DataFrame(data)


def add_special_patterns(df):
    """
    Add some special patterns to make EDA more interesting
    """
    # Pattern 1: Specific task-participant combinations that always have heavy plagiarism
    heavy_pattern = df[(df['Task'] == 'c') & (df['File'].str.contains('pC'))].index
    for idx in heavy_pattern:
        if np.random.random() < 0.8:  # 80% chance
            df.at[idx, 'Category'] = 'heavy'

    # Pattern 2: Friday participants (pF) tend to have less plagiarism
    friday_pattern = df[df['File'].str.contains('pF')].index
    for idx in friday_pattern:
        if np.random.random() < 0.7:  # 70% chance
            df.at[idx, 'Category'] = 'non'

    return df


def analyze_dataset(df):
    """
    Provide basic analysis of the generated dataset
    """
    print("Dataset Overview:")
    print(f"Total entries: {len(df)}")
    print("\nCategory Distribution:")
    print(df['Category'].value_counts())
    print(f"\nPercentage distribution:")
    print(df['Category'].value_counts(normalize=True) * 100)

    print("\nTask Distribution:")
    print(df['Task'].value_counts())

    # Check for unique groups/participants
    files = df['File'].str.extract(r'(g\d+p[A-J])')[0].unique()
    print(f"\nUnique group-participant combinations: {len(files)}")


# Generate the dataset
print("Generating plagiarism detection dataset...")
df = generate_plagiarism_dataset(1200)
df = add_special_patterns(df)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Analyze the dataset
analyze_dataset(df)

# Save to CSV
output_file = 'plagiarism_dataset.csv'
df.to_csv(output_file, index=False)
print(f"\nDataset saved to '{output_file}'")

# Display first 20 rows
print("\nFirst 20 rows of the dataset:")
print(df.head(20))

# Additional analysis for EDA preparation
print("\n" + "=" * 50)
print("ADDITIONAL ANALYSIS FOR EDA:")
print("=" * 50)

# Cross-tabulation of Task vs Category
print("\nTask vs Category Cross-tabulation:")
cross_tab = pd.crosstab(df['Task'], df['Category'], margins=True)
print(cross_tab)

# Group-level analysis
print("\nSample of group patterns (first 10 groups):")
df['Group'] = df['File'].str.extract(r'(g\d+p[A-J])')[0]
group_samples = df['Group'].value_counts().head(10)
print(group_samples)