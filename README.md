# Master's Thesis

This repository contains the source code and experimental environment for the Master's thesis titled: **"Selected methods of removing data inconsistencies and their impact on the classification process"**

## Overview

This project investigates the impact of different methods for removing data inconsistencies on classification performance. The experiments compare three data variants (baseline, quantitative inconsistency removal, and drastic inconsistency removal) using Decision Tree and Random Forest classifiers across four datasets.

## Repository Structure

- `skrypt_badawczy.py` - Main research script that runs experiments and generates results
- `skrypt_wykres.py` - Visualization script for creating charts from experimental results
- `README.md` - This documentation file

## Dependencies

The project requires the following Python packages:

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Datasets

The experiments use four datasets:

1. **Breast Cancer** - From scikit-learn's built-in datasets
2. **Mushroom** - UCI Machine Learning Repository (agaricus-lepiota.data)
3. **Adult** - UCI Machine Learning Repository (adult.data)
4. **Car** - UCI Machine Learning Repository (car.data)

All datasets are automatically downloaded from their respective sources when running the script.

## Methodology

### Data Preprocessing and Inconsistency Injection

Each dataset undergoes specific transformations to introduce inconsistencies:

- **Breast Cancer**: Reduced to 3 features with discretization (qcut, 4 bins)
- **Mushroom**: Reduced to 4 features with category merging (mod 3 mapping)
- **Adult**: Created binned features (age_bin, hours_bin) with one-hot encoding
- **Car**: Removed 1 feature (buying) with mild category merging (mod 6)

### Data Variants

Three variants are created for each dataset:

1. **Bazowy (Baseline)** - Original modified dataset with inconsistencies
2. **Ilościowy (Quantitative)** - Removes rows that are quantitatively inconsistent (non-majority class labels in duplicate feature blocks)
3. **Drastyczny (Drastic)** - Removes all rows involved in inconsistent blocks

### Experimental Setup

- **Models**: Decision Tree Classifier and Random Forest Classifier (100 estimators)
- **Cross-Validation**: Stratified K-Fold (number of folds adapted to smallest class size, default: 10)
- **Metrics**: Accuracy (mean ± std), average tree depth, average number of nodes
- **Random State**: 42 (for reproducibility)

## Usage

### Running the Main Research Script

Execute the main experiment script:

```bash
python skrypt_badawczy.py
```

This script will:

1. Load and preprocess all four datasets
2. Inject inconsistencies according to dataset-specific transformations
3. Create three variants (baseline, quantitative, drastic) for each dataset
4. Run cross-validation experiments with both classifiers
5. Display results in a formatted table
6. Generate LaTeX code for the results table

**Output**: The script prints:

- Progress information for each dataset
- Summary statistics (number of inconsistent blocks, rows removed)
- Complete results table with all metrics
- LaTeX code ready for inclusion in thesis document

### Generating Visualizations

Run the visualization script:

```bash
python skrypt_wykres.py
```

This script creates two bar charts:

1. **Accuracy Comparison** - Shows accuracy (with error bars) for Decision Tree and Random Forest across all three variants
2. **Model Complexity** - Shows the reduction in number of nodes for Decision Tree across variants

**Note**: The visualization script currently uses hardcoded data from the Adult dataset. To visualize results from other datasets or your own experimental results, modify the data arrays at the top of the script.

## Results

The experimental results include:

- **Accuracy**: Mean and standard deviation across cross-validation folds
- **Model Complexity**: Average tree depth and number of nodes (for Decision Tree)
- **Inconsistency Statistics**: Number of inconsistent blocks and rows affected

Results are formatted for easy inclusion in academic papers, with LaTeX table generation support.

## Code Structure

### `skrypt_badawczy.py`

- **Data Loading Functions**: `load_and_preprocess_*()` for each dataset
- **Inconsistency Injection**: `inject_inconsistency()` - applies dataset-specific transformations
- **Variant Creation**: `find_inconsistent_blocks()` and `get_data_variants()` - identifies and creates data variants
- **Experiment Execution**: `run_experiment()` - runs cross-validation experiments
- **Main Function**: Orchestrates all experiments and generates output

### `skrypt_wykres.py`

- Creates matplotlib/seaborn visualizations
- Configures styling for publication-quality figures
- Generates accuracy comparison and complexity reduction charts

## Configuration

Key parameters can be adjusted in `skrypt_badawczy.py`:

- `RANDOM_STATE = 42` - Random seed for reproducibility
- `BASE_N_SPLITS = 10` - Default number of CV folds (automatically adjusted based on class distribution)

## Notes

- The script automatically handles missing values and categorical encoding
- Cross-validation fold count is dynamically adjusted to ensure StratifiedKFold works correctly with small classes
- Warnings are suppressed for cleaner output
- All datasets are downloaded automatically from UCI ML Repository (except Breast Cancer which comes from scikit-learn)

## License

This code is part of a Master's thesis research project.
