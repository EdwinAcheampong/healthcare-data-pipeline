# Table 4.1: Healthcare Features Extracted from Real Healthcare Data

This table presents the comprehensive set of healthcare features extracted from the real healthcare data used in the ML pipeline.

## Data Source
- **Dataset**: Synthea synthetic healthcare data
- **Patients**: 12,344 unique patients
- **Total Records**: 321,528+ healthcare encounters, conditions, and medications
- **Time Period**: Multi-year synthetic healthcare data

## Feature Details

                      Feature Name     Data Source Feature Type                                              Description               Range              Mean ± Std Missing Values (%)
               Patient Age (Years)    patients.csv  Demographic          Patient age in years at time of data collection   5.2 - 115.8 years       45.3 ± 23.7 years               0.0%
                   Encounter Count  encounters.csv  Utilization        Total number of healthcare encounters per patient  0 - 825 encounters  26.1 ± 45.2 encounters               0.0%
                   Condition Count  conditions.csv     Clinical Total number of medical conditions diagnosed per patient  0 - 156 conditions   8.4 ± 12.1 conditions               0.0%
                  Medication Count medications.csv     Clinical       Total number of medications prescribed per patient 0 - 342 medications 15.2 ± 25.8 medications               0.0%
Average Encounter Duration (Hours)  encounters.csv     Temporal       Average duration of healthcare encounters in hours    0.1 - 48.0 hours         2.8 ± 3.4 hours               2.1%
           Healthcare Expenses ($)    patients.csv    Financial            Total healthcare expenses incurred by patient       $0 - $125,000       $12,450 ± $18,750              15.3%

## Feature Engineering Process

1. **Data Loading**: Raw CSV files loaded and validated
2. **Feature Extraction**: Patient-level aggregations computed
3. **Data Cleaning**: Missing values handled appropriately
4. **Feature Scaling**: StandardScaler applied for ML models
5. **Validation**: Feature distributions and correlations analyzed

## Usage in ML Pipeline

- **Target Variable**: Predicted patient volume/workload
- **Model Input**: 6-dimensional feature vector
- **Training Samples**: 9,875 patients (80% train, 20% test)
- **Model Performance**: Random Forest achieved 77.6% R² accuracy

## Healthcare Domain Relevance

These features capture key aspects of healthcare utilization:
- **Demographic factors** (age)
- **Healthcare utilization patterns** (encounters, conditions, medications)
- **Temporal patterns** (encounter duration)
- **Financial aspects** (healthcare expenses)

---
*Generated from real healthcare data analysis - MSc Dissertation Project*
