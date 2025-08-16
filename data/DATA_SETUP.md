# Healthcare Data Setup Guide

## 📊 **Required Data for This Project**

This project requires **651.4 MB** of healthcare data which is **too large for GitHub**. Follow the instructions below to set up the data locally.

---

## 🚀 **Quick Setup (Automated)**

The easiest way to get the data is using our automated setup script:

```bash
# Run the setup script - it will guide you through data acquisition
python scripts/setup_environment.py
```

---

## 📥 **Manual Data Setup**

### **1. Synthea COVID-19 Dataset**

You need to place **Synthea synthetic healthcare data** in the `data/synthea/` directory.

**Required Files:**

- `patients.csv` (12,352 patients)
- `conditions.csv` (114,544 conditions)
- `encounters.csv` (321,528 encounters)
- `observations.csv` (1,659,750 observations)
- `medications.csv` (431,262 medications)
- `procedures.csv` (100,427 procedures)
- `organizations.csv` (5,499 organizations)
- `providers.csv` (31,764 providers)
- `allergies.csv` (5,417 allergies)
- `careplans.csv` (37,715 care plans)
- `immunizations.csv` (16,481 immunizations)
- `devices.csv` (2,360 devices)
- `supplies.csv` (143,110 supplies)
- `imaging_studies.csv` (4,504 imaging studies)
- `payers.csv` (10 payers)
- `payer_transitions.csv` (insurance transitions)

### **2. C-CDA XML Documents**

You also need **109 C-CDA XML documents** in the `data/ccda/` directory for healthcare interoperability testing.

---

## 📦 **Data Sources**

### **⭐ Option 1: Download from Google Drive (Recommended)**

**Direct Access**: [https://drive.google.com/drive/folders/1tpU0jugYL3w6cji3zRACPV1qyLB3fAOj?usp=sharing](https://drive.google.com/drive/folders/1tpU0jugYL3w6cji3zRACPV1qyLB3fAOj?usp=sharing)

**Steps**:
1. Click the Google Drive link above
2. Right-click on the folder → "Download" (creates zip file)
3. Extract the downloaded zip to your project's `data/` directory
4. Run validation: `python scripts/validate_data.py`

### **Option 2: Generate with Synthea**

```bash
# Install Synthea
git clone https://github.com/synthetichealth/synthea.git
cd synthea

# Generate COVID-19 focused data
./run_synthea -p 12352 --exporter.fhir.export false --exporter.ccda.export true Massachusetts
```

### **Option 3: Alternative Data Sources**

If you prefer to use your own Synthea datasets, the project expects:

- **12,352+ patients** for meaningful analysis
- **COVID-19 module enabled** for pandemic research
- **Multi-year data range** (2010-2020) for temporal analysis

---

## 🔍 **Data Validation**

After setting up your data, validate it:

```bash
# Validate data structure and integrity
python scripts/validate_data.py

# Verify complete setup
python scripts/verify_setup.py
```

**Expected Output:**

```
✅ Data Validation Complete!
📊 Data Summary:
- CSV Files: 15/15 found
- Patients: 12,352
- XML Documents: 109
- Total Data Size: ~651.4 MB
```

---

## 📁 **Expected Directory Structure**

After setup, your data structure should look like:

```
data/
├── synthea/
│   ├── patients.csv
│   ├── conditions.csv
│   ├── encounters.csv
│   ├── observations.csv
│   ├── medications.csv
│   ├── procedures.csv
│   ├── organizations.csv
│   ├── providers.csv
│   ├── allergies.csv
│   ├── careplans.csv
│   ├── immunizations.csv
│   ├── devices.csv
│   ├── supplies.csv
│   ├── imaging_studies.csv
│   ├── payers.csv
│   ├── payer_transitions.csv
│   └── Synthea COVID-19 Analysis.html
│
├── ccda/
│   ├── [109 XML files]
│   └── ...
│
├── processed/
│   └── [Generated during analysis]
│
└── raw/
    └── [Additional raw data as needed]
```

---

## ⚠️ **Important Notes**

### **Privacy & Ethics**

- ✅ **100% Synthetic Data**: No real patient information
- ✅ **Research Safe**: No IRB approval required
- ✅ **Educational Use**: Perfect for learning and development
- ⚠️ **Not for Production**: Synthetic data only, not real healthcare decisions

### **Data Size Considerations**

- **Total Size**: ~651.4 MB (too large for GitHub)
- **CSV Files**: ~500 MB (patient records)
- **XML Files**: ~150 MB (clinical documents)
- **Processing**: Requires 8GB+ RAM for full analysis

### **Academic Use**

- **Citation Required**: Acknowledge Synthea in academic work
- **Reproducibility**: Document data generation parameters
- **Sharing**: Provide data setup instructions for peer review

---

## 🆘 **Troubleshooting**

### **Data Not Found Error**

```bash
# If you get "data files not found" errors
python scripts/verify_setup.py
# Follow the prompts to fix missing data
```

### **Size Warnings**

```bash
# If analysis is slow due to data size
# Use sample data for initial development
# Full dataset for final analysis
```

### **Format Issues**

```bash
# If CSV files have encoding issues
# Ensure UTF-8 encoding
# Verify column names match expected format
```

---

## 📞 **Need Help?**

- **Setup Issues**: Run `python scripts/verify_setup.py`
- **Data Questions**: Check [Data Inventory](DATA_INVENTORY.md)
- **GitHub Issues**: [Report problems](https://github.com/EdwinAcheampong/healthcare-data-pipeline/issues)

---

**Remember**: This project uses synthetic data for safety and educational purposes. Always follow your institution's data handling guidelines for real healthcare projects!
