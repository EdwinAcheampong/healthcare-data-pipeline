# Healthcare Dataset Hosting Guide

## 🎯 **Recommended: Zenodo Academic Repository**

[Zenodo](https://zenodo.org) is the **best choice for academic MSc projects** because it provides:
- ✅ **Permanent DOI**: Citable dataset with academic credibility
- ✅ **Free Hosting**: Up to 50GB per dataset
- ✅ **Version Control**: Track dataset versions
- ✅ **Academic Recognition**: Integrated with CERN, EU research
- ✅ **Long-term Preservation**: Data preserved indefinitely

### **📤 Steps to Upload to Zenodo**

1. **Create Account**: https://zenodo.org (use institutional email)
2. **New Upload**: Click "Upload" → "New upload"
3. **Upload Files**: Add your data files (all 651MB)
4. **Add Metadata**:
   ```
   Title: Synthea COVID-19 Healthcare Dataset for MSc Research
   Description: Comprehensive synthetic healthcare dataset containing 12,352 patients...
   Keywords: healthcare, synthea, covid19, FHIR, synthetic-data, medical-informatics
   License: Creative Commons Attribution 4.0
   ```
5. **Publish**: Get permanent DOI (e.g., `10.5281/zenodo.XXXXXX`)

### **🔗 Integration with Repository**

Once uploaded, update your project documentation with:
```markdown
## Data Access
Download the complete dataset (651MB) from:
**DOI:** [10.5281/zenodo.XXXXXX](https://doi.org/10.5281/zenodo.XXXXXX)
```

---

## 🌐 **Alternative: Google Drive (Quick Setup)**

For immediate sharing while setting up Zenodo:

1. **Upload to Google Drive**
2. **Set Sharing**: "Anyone with link can view"
3. **Get Direct Download Link**:
   ```
   Original: https://drive.google.com/file/d/FILE_ID/view
   Direct:   https://drive.google.com/uc?export=download&id=FILE_ID
   ```

---

## 📋 **Data Package Contents**

Whatever platform you choose, include these files:

### **Core Dataset (CSV Files)**
- `patients.csv` (12,352 patients, ~3.4MB)
- `encounters.csv` (321,528 encounters, ~95MB)
- `conditions.csv` (114,544 conditions, ~13.5MB)
- `observations.csv` (1,659,750 observations, ~239MB)
- `medications.csv` (431,262 medications, ~97MB)
- `procedures.csv` (100,427 procedures, ~16MB)
- `organizations.csv` (5,499 organizations, ~0.8MB)
- `providers.csv` (31,764 providers, ~5.4MB)
- `allergies.csv` (5,417 allergies, ~0.6MB)
- `careplans.csv` (37,715 care plans, ~7.2MB)
- `immunizations.csv` (16,481 immunizations, ~2.1MB)
- `devices.csv` (2,360 devices, ~0.5MB)
- `supplies.csv` (143,110 supplies, ~19.6MB)
- `imaging_studies.csv` (4,504 studies, ~1MB)
- `payers.csv` (10 payers, ~0.003MB)
- `payer_transitions.csv` (transitions, ~0.1MB)

### **Clinical Documents (XML Files)**
- 109 C-CDA XML files (~150MB total)

### **Documentation**
- Dataset methodology
- Data dictionary
- Usage guidelines

---

## 📚 **Academic Citation**

Once hosted on Zenodo, users can cite your dataset:

```bibtex
@dataset{yekini2024synthea,
  author = {Yekini, Muhammad and Acheampong, Edwin},
  title = {{Synthea COVID-19 Healthcare Dataset for MSc Research}},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXX},
  url = {https://doi.org/10.5281/zenodo.XXXXXX}
}
```

---

## 🔄 **Automated Download Script**

Create a download script that users can run:

```python
# scripts/download_data.py
import requests
import zipfile
from pathlib import Path

def download_dataset():
    """Download dataset from hosted location."""
    zenodo_url = "https://zenodo.org/record/XXXXXX/files/healthcare-dataset.zip"
    # Implementation for automated download
```

This makes data acquisition seamless for researchers!
