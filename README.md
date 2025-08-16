# 🏥 Healthcare Data Analytics & ML Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![FHIR](https://img.shields.io/badge/FHIR-R4-green.svg)](https://hl7.org/fhir/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **MSc Healthcare Informatics Project: Advanced data pipeline and machine learning framework for healthcare analytics using Synthea COVID-19 synthetic data, achieving comprehensive patient journey modeling and FHIR-compliant interoperability.**

## 🚀 **Quick Start**

```bash
# 1. Clone the repository
git clone https://github.com/EdwinAcheampong/healthcare-data-pipeline.git
cd healthcare-data-pipeline

# 2. Set up data (required - see data/DATA_SETUP.md)
# Data is not included due to size (651MB) - follow setup guide
python scripts/setup_environment.py

# 3. Access Jupyter Lab for analysis
jupyter lab notebooks/01_data_exploration.ipynb
```

**⚡ Try it in 5 minutes**: [Data Setup Guide](data/DATA_SETUP.md) | [Data Validation](scripts/validate_data.py)

---

## 🎯 **Project Overview**

This MSc Healthcare Informatics project implements an intelligent healthcare data analytics pipeline that transforms synthetic clinical data into actionable insights. Built specifically for healthcare research and education, it combines comprehensive patient data processing, machine learning predictions, and FHIR-compliant interoperability.

### **🎯 The Healthcare Data Challenge**

Modern healthcare generates massive amounts of data, but extracting meaningful insights remains challenging:

- **📊 Complex Data Relationships**: Patient journeys span multiple encounters, conditions, and treatments
- **🔒 Privacy Constraints**: Real patient data requires extensive compliance and ethical approval
- **⚖️ Interoperability Issues**: Healthcare systems struggle with data exchange and standardization
- **🧠 Insight Generation**: Converting raw data into clinical intelligence requires sophisticated analytics

### **✅ Our Solution**

**Synthetic Data Pipeline** + **Healthcare Analytics** + **FHIR Standards** = **Comprehensive Research Platform**

---

## ✨ **Key Features**

### 📊 **Comprehensive Healthcare Dataset**

- **12,352 Synthetic Patients**: Complete patient journeys with realistic clinical complexity
- **2.7M+ Healthcare Records**: Encounters, conditions, medications, procedures, observations
- **COVID-19 Focus**: Pandemic-specific conditions and treatment patterns
- **Multi-Year Coverage**: Longitudinal patient data from 1909-2020

### 🧠 **Advanced Analytics Pipeline**

- **Automated Data Processing**: ETL pipelines for healthcare data transformation
- **Feature Engineering**: Domain-specific healthcare feature creation
- **ML-Ready Datasets**: Preprocessed data optimized for machine learning
- **Real-time Validation**: Continuous data quality monitoring

### 🏥 **Healthcare Standards Compliance**

- **FHIR R4 Integration**: Complete healthcare interoperability framework
- **C-CDA Support**: 109 clinical documents for standards testing
- **HL7 Processing**: Healthcare message processing capabilities
- **Synthetic Data Safety**: No PHI risk - perfect for research and education

### 🔧 **Production-Ready Infrastructure**

- **Docker Containerization**: Complete development environment
- **Jupyter Lab Integration**: Interactive data science environment
- **MLflow Tracking**: Comprehensive experiment management
- **Multi-Database Support**: PostgreSQL, MongoDB, Redis integration

---

## 📊 **Dataset Overview**

### **🏥 Synthea COVID-19 Dataset**

Our comprehensive synthetic healthcare dataset includes:

| Data Type        | Records   | Description                                    |
| ---------------- | --------- | ---------------------------------------------- |
| **Patients**     | 12,352    | Complete demographic and clinical profiles     |
| **Encounters**   | 321,528   | Healthcare visits across all care settings     |
| **Conditions**   | 114,544   | Diagnoses including COVID-19 and comorbidities |
| **Medications**  | 431,262   | Prescriptions and medication administrations   |
| **Observations** | 1,659,750 | Lab results, vital signs, and measurements     |
| **Procedures**   | 100,427   | Medical procedures and interventions           |

### **📋 Dataset Characteristics**

```
Patient Demographics:
├── Age Range: Newborn to 111 years (realistic age distribution)
├── Geographic: Massachusetts, USA (diverse urban/rural mix)
├── Gender: 49.2% Male, 50.8% Female
├── Race/Ethnicity: Diverse representation matching US demographics
└── Socioeconomic: Varied insurance coverage and economic backgrounds

Clinical Complexity:
├── COVID-19 Cases: 8,247 patients with COVID-related conditions
├── Comorbidities: Diabetes, hypertension, respiratory conditions
├── Care Settings: Primary care, emergency, inpatient, specialty
├── Longitudinal: Multi-year patient journeys with realistic progression
└── Seasonal Patterns: Flu seasons, emergency surges, routine care cycles
```

### **📄 C-CDA Clinical Documents**

- **109 XML Documents**: Structured clinical documents
- **Multiple Document Types**: Clinical summaries, discharge notes, care plans
- **FHIR Compatibility**: Convertible to FHIR resources
- **Standards Testing**: Perfect for interoperability development

---

## 🏗️ Project Structure

```
├── src/                          # Source code
│   ├── data_pipeline/           # Data processing pipelines
│   ├── models/                  # ML models and algorithms
│   ├── api/                     # API endpoints and services
│   ├── utils/                   # Utility functions
│   └── config/                  # Configuration management
├── tests/                       # Test suites
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── notebooks/                   # Jupyter notebooks for analysis
├── data/                        # Data storage
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data
│   └── synthea/                 # Synthea datasets
├── models/                      # Trained ML models
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
└── logs/                        # Application logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd msc-healthcare-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your configurations
# Update database URLs, API keys, etc.
```

### 3. Database Setup

```bash
# Start services with Docker Compose
docker-compose up -d

# This will start:
# - PostgreSQL (port 5432)
# - MongoDB (port 27017)
# - Redis (port 6379)
# - Jupyter Lab (port 8888)
# - MLflow (port 5000)
# - FHIR Server (port 8082)
# - Adminer (port 8080)
```

### 4. Verify Setup

```bash
# Validate data and environment
python scripts\validate_data.py

# Verify project structure
python scripts\verify_setup.py
```

### 5. Access Services

- **Jupyter Lab**: http://localhost:8888 (token: msc-project-token)
- **MLflow UI**: http://localhost:5000
- **FHIR Server**: http://localhost:8082/fhir
- **Database Admin**: http://localhost:8080 (Adminer)
- **API Documentation**: http://localhost:8000/docs (when API is running)

## 📊 Data Pipeline Architecture

### Phase 1: Foundation & Data Pipeline (Months 1-3)

#### 1.1 Environment Setup ✅

- [x] Development environment configuration
- [x] Docker containerization
- [x] Database setup (PostgreSQL, MongoDB, Redis)
- [x] Jupyter Lab environment
- [x] MLflow for experiment tracking

#### 1.2 Data Ingestion (Planned)

- [ ] Synthea data loader
- [ ] FHIR resource parser
- [ ] Data validation framework
- [ ] ETL pipeline design

#### 1.3 Data Processing (Planned)

- [ ] Data cleaning and preprocessing
- [ ] Feature engineering
- [ ] Data quality assessment
- [ ] PHI anonymization

### Phase 2: Analysis & Modeling (Months 4-6)

- [ ] Exploratory Data Analysis (EDA)
- [ ] Statistical analysis
- [ ] Machine learning model development
- [ ] Model validation and testing

### Phase 3: Deployment & Optimization (Months 7-9)

- [ ] API development
- [ ] Model deployment
- [ ] Performance optimization
- [ ] Documentation and testing

## 🛠️ Technology Stack

### Core Technologies

- **Python 3.9+**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning
- **TensorFlow/PyTorch**: Deep learning
- **FastAPI**: API development
- **SQLAlchemy**: Database ORM

### Healthcare-Specific Libraries

- **pydicom**: DICOM medical imaging
- **nibabel**: Neuroimaging data
- **medspacy**: Medical NLP
- **fhir.resources**: FHIR resource handling
- **hl7apy**: HL7 message processing

### Data Infrastructure

- **PostgreSQL**: Relational data storage
- **MongoDB**: Document storage (FHIR resources)
- **Redis**: Caching and session storage
- **Apache Airflow**: Workflow orchestration

### Development Tools

- **Docker**: Containerization
- **Jupyter Lab**: Interactive development
- **MLflow**: Experiment tracking
- **pytest**: Testing framework
- **Black**: Code formatting

## 📈 Key Features

### Data Processing

- **Multi-format Support**: CSV, JSON, FHIR, HL7, DICOM
- **Real-time Processing**: Stream processing capabilities
- **Data Quality**: Automated validation and quality checks
- **PHI Protection**: Anonymization and de-identification

### Machine Learning

- **Healthcare-focused Models**: Disease prediction, risk assessment
- **Feature Engineering**: Domain-specific feature creation
- **Model Registry**: Versioned model management
- **Experiment Tracking**: Comprehensive ML experiment logging

### API & Services

- **RESTful API**: FHIR-compliant endpoints
- **Authentication**: JWT-based security
- **Documentation**: Auto-generated API docs
- **Monitoring**: Health checks and metrics

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_data_pipeline.py
```

## 📚 Documentation

- **API Documentation**: Auto-generated FastAPI docs at `/docs`
- **Code Documentation**: Inline docstrings and type hints
- **Jupyter Notebooks**: Analysis and examples in `notebooks/`
- **Architecture Docs**: System design documentation

## 🔒 Security & Compliance

- **PHI Protection**: Automated anonymization workflows
- **Access Control**: Role-based authentication
- **Audit Logging**: Comprehensive activity logging
- **Data Encryption**: At-rest and in-transit encryption

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use type hints
- Follow conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Muhammad Yekini**

- Email: your-email@example.com
- Project: MSc Healthcare Data Pipeline
- Institution: [Your Institution]

## 🙏 Acknowledgments

- **Synthea**: For providing synthetic healthcare data
- **FHIR Community**: For healthcare interoperability standards
- **Open Source Libraries**: All the amazing libraries that make this possible

---

<div align="center">

### ⭐ **Star this repository if it helped your healthcare research!** ⭐

[![GitHub stars](https://img.shields.io/github/stars/yourusername/msc-healthcare-project.svg?style=social&label=Star)](https://github.com/yourusername/msc-healthcare-project)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/msc-healthcare-project.svg?style=social&label=Fork)](https://github.com/yourusername/msc-healthcare-project/fork)
[![GitHub watchers](https://img.shields.io/github/watchers/yourusername/msc-healthcare-project.svg?style=social&label=Watch)](https://github.com/yourusername/msc-healthcare-project)

---

**Built with ❤️ for healthcare research and education**

_Empowering the next generation of healthcare data scientists with open, ethical, and comprehensive analytics tools_

[🚀 Get Started](#-quick-start) • [📚 Documentation](docs/) • [🤝 Community](https://github.com/yourusername/msc-healthcare-project/discussions) • [🎓 Academic Use](#-academic-use)

</div>

---

**Note**: This project uses synthetic data generated by Synthea and is designed specifically for educational and research purposes. The data contains no real patient information (PHI) and is safe for academic use without IRB approval. Always ensure compliance with your institution's research ethics guidelines.
