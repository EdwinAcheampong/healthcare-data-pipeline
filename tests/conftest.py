"""
Pytest configuration and fixtures for healthcare analytics tests.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock

# Test data fixtures
@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing."""
    return pd.DataFrame({
        'Id': ['patient_1', 'patient_2', 'patient_3'],
        'BIRTHDATE': ['1980-01-01', '1990-05-15', '2000-12-31'],
        'GENDER': ['M', 'F', 'M'],
        'RACE': ['white', 'black', 'asian'],
        'ETHNICITY': ['nonhispanic', 'hispanic', 'nonhispanic']
    })

@pytest.fixture
def sample_condition_data():
    """Sample condition data for testing."""
    return pd.DataFrame({
        'PATIENT': ['patient_1', 'patient_2', 'patient_3'],
        'ENCOUNTER': ['enc_1', 'enc_2', 'enc_3'],
        'START': ['2020-01-01', '2020-03-15', '2020-06-30'],
        'STOP': ['2020-01-15', '2020-04-01', '2020-07-15'],
        'DESCRIPTION': ['COVID-19', 'Diabetes', 'Hypertension']
    })

@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()

# Test markers
pytestmark = pytest.mark.healthcare
