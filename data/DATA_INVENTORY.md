# Data Inventory - MSc Healthcare Project

## Overview

This document provides an inventory of all data files and their organization within the project.

## Data Structure

```
data/
├── synthea/          # Synthea synthetic healthcare data (CSV format)
├── ccda/            # C-CDA XML documents
├── raw/             # Raw/unprocessed data files
└── processed/       # Processed and cleaned data files
```

## Synthea Dataset (data/synthea/)

### Core Tables

| File             | Records     | Description                                   |
| ---------------- | ----------- | --------------------------------------------- |
| patients.csv     | 12,344      | Patient demographic and basic information     |
| conditions.csv   | ~50,000+    | Medical conditions and diagnoses              |
| encounters.csv   | ~150,000+   | Healthcare encounters (visits, admissions)    |
| observations.csv | ~1,000,000+ | Laboratory results, vital signs, measurements |
| medications.csv  | ~200,000+   | Medication prescriptions and administrations  |
| procedures.csv   | ~100,000+   | Medical procedures performed                  |

### Supporting Tables

| File                  | Description                                  |
| --------------------- | -------------------------------------------- |
| organizations.csv     | Healthcare organizations and facilities      |
| providers.csv         | Healthcare providers (doctors, nurses, etc.) |
| payers.csv            | Insurance payers and coverage information    |
| payer_transitions.csv | Changes in insurance coverage                |
| allergies.csv         | Patient allergies and adverse reactions      |
| careplans.csv         | Care plans and treatment protocols           |
| devices.csv           | Medical devices used                         |
| immunizations.csv     | Vaccination records                          |
| supplies.csv          | Medical supplies used                        |
| imaging_studies.csv   | Medical imaging studies                      |

### Analysis Documents

| File                           | Description                   |
| ------------------------------ | ----------------------------- |
| Synthea COVID-19 Analysis.html | Pre-generated analysis report |

## C-CDA Documents (data/ccda/)

**Format**: XML (C-CDA Release 2.1)  
**Count**: 109 documents  
**Description**: Clinical Document Architecture (CDA) documents containing structured clinical information

### Sample Documents

- Patient clinical summaries
- Continuity of Care Documents (CCD)
- Discharge summaries
- Clinical notes

## Data Characteristics

### Patient Demographics

- **Total Patients**: 12,344
- **Geographic Coverage**: Massachusetts, USA
- **Age Range**: Newborn to elderly
- **Gender Distribution**: Mixed male/female
- **Race/Ethnicity**: Diverse representation

### Healthcare Data Scope

- **COVID-19 Focus**: Includes pandemic-related conditions and treatments
- **Time Period**: Multi-year synthetic data
- **Clinical Domains**:
  - Primary care
  - Specialty care
  - Emergency visits
  - Hospitalizations
  - Laboratory tests
  - Medications
  - Procedures

## Data Quality Notes

### Synthea Data

- ✅ **Synthetic**: No real patient data (PHI-safe)
- ✅ **Structured**: Well-formatted CSV files
- ✅ **Linked**: Proper foreign key relationships
- ✅ **Comprehensive**: Full healthcare journey simulation

### C-CDA Data

- ✅ **Standards-compliant**: Follows HL7 C-CDA specifications
- ✅ **Structured**: XML format with defined schemas
- ⚠️ **Complex**: Requires specialized parsing tools

## Usage Guidelines

### For Analysis

1. **Start with patients.csv** for demographic analysis
2. **Link datasets** using patient IDs and encounter IDs
3. **Filter by date ranges** for temporal analysis
4. **Aggregate carefully** to avoid patient re-identification

### For Machine Learning

1. **Feature engineering** from multiple tables
2. **Temporal modeling** using encounter sequences
3. **Outcome prediction** using conditions and procedures
4. **Risk stratification** using demographic and clinical factors

### For FHIR Development

1. **Map Synthea to FHIR** resources
2. **Use C-CDA documents** for document reference patterns
3. **Validate against FHIR profiles**
4. **Test interoperability** scenarios

## Privacy and Ethics

- ✅ All data is **synthetic** - no real patients
- ✅ Safe for research and development
- ✅ No IRB approval required for synthetic data
- ⚠️ Still follow data handling best practices
- ⚠️ Be mindful of synthetic data limitations

## Next Steps

1. **Data Profiling**: Run automated data quality checks
2. **EDA**: Perform exploratory data analysis
3. **ETL Pipeline**: Design extraction, transformation, loading processes
4. **FHIR Mapping**: Convert to FHIR resources
5. **ML Preparation**: Feature engineering and model-ready datasets

---

**Last Updated**: 2025-01-15  
**Data Version**: Synthea COVID-19 10k sample  
**Contact**: Muhammad Yekini - MSc Project

