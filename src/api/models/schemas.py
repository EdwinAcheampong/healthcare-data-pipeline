"""
Data schemas for the Healthcare Data Pipeline API.

This module contains Pydantic models for healthcare data structures
and domain-specific schemas used throughout the API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date, time
from enum import Enum


class PatientStatus(str, Enum):
    """Patient status enumeration."""
    ADMITTED = "admitted"
    DISCHARGED = "discharged"
    TRANSFERRED = "transferred"
    PENDING = "pending"
    EMERGENCY = "emergency"


class StaffRole(str, Enum):
    """Staff role enumeration."""
    DOCTOR = "doctor"
    NURSE = "nurse"
    TECHNICIAN = "technician"
    ADMINISTRATOR = "administrator"
    SPECIALIST = "specialist"
    RESIDENT = "resident"


class ShiftType(str, Enum):
    """Shift type enumeration."""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"
    ON_CALL = "on_call"
    WEEKEND = "weekend"


class DepartmentType(str, Enum):
    """Department type enumeration."""
    EMERGENCY = "emergency"
    INTENSIVE_CARE = "intensive_care"
    SURGERY = "surgery"
    PEDIATRICS = "pediatrics"
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    GENERAL = "general"


class PatientData(BaseModel):
    """Schema for patient data."""
    
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=150, description="Patient age")
    gender: str = Field(..., description="Patient gender")
    status: PatientStatus = Field(..., description="Current patient status")
    department: str = Field(..., description="Current department")
    admission_date: Optional[datetime] = Field(None, description="Admission date")
    discharge_date: Optional[datetime] = Field(None, description="Discharge date")
    severity_score: Optional[float] = Field(None, ge=0, le=10, description="Patient severity score")
    comorbidities: List[str] = Field(default_factory=list, description="List of comorbidities")
    medications: List[str] = Field(default_factory=list, description="Current medications")
    vital_signs: Optional[Dict[str, float]] = Field(None, description="Current vital signs")
    care_plan: Optional[Dict[str, Any]] = Field(None, description="Care plan details")
    
    @validator('severity_score')
    def validate_severity_score(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError('Severity score must be between 0 and 10')
        return v


class StaffData(BaseModel):
    """Schema for staff data."""
    
    staff_id: str = Field(..., description="Unique staff identifier")
    name: str = Field(..., description="Staff member name")
    role: StaffRole = Field(..., description="Staff role")
    department: str = Field(..., description="Assigned department")
    shift: ShiftType = Field(..., description="Current shift")
    experience_years: float = Field(..., ge=0, description="Years of experience")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    current_patients: int = Field(0, ge=0, description="Number of patients currently assigned")
    max_patients: int = Field(..., ge=1, description="Maximum patients this staff can handle")
    availability: bool = Field(True, description="Current availability status")
    specializations: List[str] = Field(default_factory=list, description="Specializations")
    
    @validator('current_patients')
    def validate_current_patients(cls, v, values):
        if 'max_patients' in values and v > values['max_patients']:
            raise ValueError('Current patients cannot exceed maximum capacity')
        return v


class WorkloadMetrics(BaseModel):
    """Schema for workload metrics."""
    
    department: str = Field(..., description="Department name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    total_patients: int = Field(..., ge=0, description="Total patients in department")
    total_staff: int = Field(..., ge=0, description="Total staff in department")
    patient_staff_ratio: float = Field(..., description="Patient to staff ratio")
    average_wait_time: float = Field(..., ge=0, description="Average wait time in minutes")
    bed_occupancy_rate: float = Field(..., ge=0, le=1, description="Bed occupancy rate")
    staff_utilization_rate: float = Field(..., ge=0, le=1, description="Staff utilization rate")
    emergency_admissions: int = Field(0, ge=0, description="Number of emergency admissions")
    scheduled_procedures: int = Field(0, ge=0, description="Number of scheduled procedures")
    critical_patients: int = Field(0, ge=0, description="Number of critical patients")
    efficiency_score: float = Field(..., ge=0, le=1, description="Overall efficiency score")
    
    @validator('patient_staff_ratio')
    def validate_ratio(cls, v):
        if v < 0:
            raise ValueError('Patient-staff ratio cannot be negative')
        return v


class OptimizationResult(BaseModel):
    """Schema for optimization results."""
    
    optimization_id: str = Field(..., description="Unique optimization identifier")
    department: str = Field(..., description="Target department")
    timestamp: datetime = Field(default_factory=datetime.now, description="Optimization timestamp")
    current_state: WorkloadMetrics = Field(..., description="Current workload state")
    recommended_state: WorkloadMetrics = Field(..., description="Recommended workload state")
    recommended_staff_allocation: Dict[str, int] = Field(..., description="Recommended staff allocation")
    recommended_schedule: Dict[str, Any] = Field(..., description="Recommended schedule")
    expected_improvements: Dict[str, float] = Field(..., description="Expected improvements")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in recommendations")
    constraints_applied: List[str] = Field(default_factory=list, description="Constraints applied")
    optimization_method: str = Field(..., description="Optimization method used")
    processing_time: float = Field(..., description="Processing time in seconds")


class ScheduleSlot(BaseModel):
    """Schema for schedule time slots."""
    
    start_time: time = Field(..., description="Slot start time")
    end_time: time = Field(..., description="Slot end time")
    staff_required: int = Field(..., ge=0, description="Required staff count")
    staff_assigned: List[str] = Field(default_factory=list, description="Assigned staff IDs")
    patient_count: int = Field(0, ge=0, description="Expected patient count")
    activity_type: str = Field(..., description="Type of activity")
    priority: int = Field(1, ge=1, le=5, description="Priority level (1-5)")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be after start time')
        return v


class DepartmentSchedule(BaseModel):
    """Schema for department schedule."""
    
    department: str = Field(..., description="Department name")
    schedule_date: date = Field(..., description="Schedule date")
    shifts: Dict[ShiftType, List[ScheduleSlot]] = Field(..., description="Schedule by shift")
    total_staff_hours: float = Field(..., description="Total staff hours scheduled")
    coverage_gaps: List[Dict[str, Any]] = Field(default_factory=list, description="Identified coverage gaps")


class PredictionInput(BaseModel):
    """Schema for prediction input data."""
    
    department: str = Field(..., description="Target department")
    prediction_date: date = Field(..., description="Date for prediction")
    historical_data: List[WorkloadMetrics] = Field(..., description="Historical workload data")
    external_factors: Optional[Dict[str, Any]] = Field(None, description="External factors")
    seasonal_patterns: Optional[Dict[str, float]] = Field(None, description="Seasonal patterns")
    special_events: Optional[List[str]] = Field(None, description="Special events affecting demand")
    
    @validator('historical_data')
    def validate_historical_data(cls, v):
        if len(v) < 7:
            raise ValueError('At least 7 days of historical data required')
        return v


class ModelMetadata(BaseModel):
    """Schema for model metadata."""
    
    model_id: str = Field(..., description="Unique model identifier")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    training_date: datetime = Field(..., description="Training date")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    training_data_size: int = Field(..., description="Training data size")
    validation_data_size: int = Field(..., description="Validation data size")
    model_size_mb: float = Field(..., description="Model file size in MB")
    deployment_status: str = Field(..., description="Deployment status")


class AlertRule(BaseModel):
    """Schema for alert rules."""
    
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    metric: str = Field(..., description="Metric to monitor")
    threshold: float = Field(..., description="Alert threshold")
    operator: str = Field(..., description="Comparison operator")
    severity: str = Field(..., description="Alert severity")
    department: Optional[str] = Field(None, description="Target department")
    enabled: bool = Field(True, description="Rule enabled status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class Alert(BaseModel):
    """Schema for alerts."""
    
    alert_id: str = Field(..., description="Unique alert identifier")
    rule_id: str = Field(..., description="Triggering rule ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    metric_value: float = Field(..., description="Current metric value")
    threshold_value: float = Field(..., description="Threshold value")
    department: Optional[str] = Field(None, description="Affected department")
    acknowledged: bool = Field(False, description="Alert acknowledged status")
    resolved: bool = Field(False, description="Alert resolved status")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")
