# Codebase Cleanup Plan

## Overview
This document outlines the cleanup and reorganization of the healthcare data pipeline codebase before Phase 3 deployment.

## Current Issues to Address

### 1. File Organization
- Multiple documentation files in root directory
- Temporary files and logs in root
- Inconsistent file naming
- Missing proper documentation structure

### 2. Models Directory
- Multiple result files with timestamps
- Need to keep only the latest/best results
- Archive old results

### 3. Logs Directory
- Large log files that shouldn't be in version control
- Need proper log rotation and cleanup

### 4. Documentation
- Move all documentation to docs/ folder
- Create proper documentation structure
- Update README files

## Cleanup Actions

### Phase 1: Documentation Reorganization
1. Move `PHASE_2A_IMPACT_SUMMARY.md` to `docs/`
2. Move `deliverable.md` to `docs/`
3. Update main README.md with proper structure
4. Create comprehensive project documentation

### Phase 2: File Cleanup
1. Remove temporary files
2. Clean up models directory (keep only latest results)
3. Clean up logs directory
4. Remove .gitkeep files where not needed

### Phase 3: Structure Optimization
1. Create proper directory structure
2. Update .gitignore
3. Create deployment documentation
4. Prepare for Phase 3

## Target Structure
```
healthcare-data-pipeline/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── pytest.ini
├── Makefile
├── docker-compose.yml
├── docker-compose.prod.yml
├── Dockerfile.jupyter
├── .gitignore
├── .pre-commit-config.yaml
├── env.example
├── src/
│   ├── api/
│   ├── config/
│   ├── data_pipeline/
│   ├── models/
│   ├── utils/
│   └── main.py
├── docs/
│   ├── README.md
│   ├── guides/
│   ├── images/
│   ├── PHASE_2A_IMPLEMENTATION.md
│   ├── PHASE_2B_IMPLEMENTATION.md
│   ├── PHASE_2A_IMPACT_SUMMARY.md
│   ├── deliverable.md
│   └── PROJECT_STRUCTURE.md
├── scripts/
├── tests/
├── notebooks/
├── data/
├── models/
│   ├── phase_2a_test_results.json
│   └── phase_2b_results_latest.json
└── logs/
    └── .gitkeep
```

## Files to Remove/Archive
- `PHASE_2A_IMPACT_SUMMARY.md` (move to docs/)
- `deliverable.md` (move to docs/)
- Old timestamped result files in models/
- Large log files in logs/
- Temporary files and .gitkeep files where not needed
