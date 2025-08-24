"""
Reinforcement Learning Environment for Healthcare Workload Management

This module implements a custom OpenAI Gym environment for healthcare workload
optimization using reinforcement learning. The environment simulates healthcare
workload patterns and allows RL agents to learn optimal resource allocation strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthcareWorkloadState:
    """Represents the state of the healthcare workload environment."""
    
    # Current workload metrics
    current_patients: int
    current_staff: int
    current_beds: int
    current_wait_time: float
    
    # Resource utilization
    staff_utilization: float
    bed_utilization: float
    equipment_utilization: float
    
    # Time-based features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    is_holiday: bool
    
    # Predicted workload (from Phase 2A models)
    predicted_patients_1h: float
    predicted_patients_4h: float
    predicted_patients_24h: float
    
    # Compliance metrics
    compliance_score: float
    safety_score: float
    quality_score: float


@dataclass
class HealthcareWorkloadAction:
    """Represents an action in the healthcare workload environment."""
    
    # Staffing decisions
    add_staff: int
    remove_staff: int
    
    # Bed management
    add_beds: int
    remove_beds: int
    
    # Equipment allocation
    add_equipment: int
    remove_equipment: int
    
    # Operational decisions
    activate_overflow: bool
    activate_emergency_protocol: bool


class HealthcareWorkloadEnvironment(gym.Env):
    """
    Healthcare Workload Management Environment for Reinforcement Learning.
    
    This environment simulates a healthcare facility where an RL agent must
    make decisions about resource allocation to optimize patient care while
    maintaining compliance and cost efficiency.
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the healthcare workload environment.
        
        Args:
            historical_data: Historical workload data from Phase 2A
            config: Configuration dictionary for environment parameters
        """
        super().__init__()
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Historical data
        self.historical_data = historical_data
        self.current_step = 0
        self.max_steps = len(historical_data) - 1
        
        # Current state
        self.current_state = None
        
        # Statistics tracking
        self.episode_stats = {
            'total_reward': 0.0,
            'total_cost': 0.0,
            'total_patients_served': 0,
            'compliance_violations': 0,
            'safety_incidents': 0
        }
        
        # Define action and observation spaces
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()
        
        logger.info("Healthcare Workload Environment initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the environment."""
        return {
            # Resource constraints
            'max_staff': 100,
            'max_beds': 200,
            'max_equipment': 50,
            'min_staff': 10,
            'min_beds': 20,
            'min_equipment': 5,
            
            # Cost parameters
            'staff_cost_per_hour': 50.0,
            'bed_cost_per_hour': 10.0,
            'equipment_cost_per_hour': 25.0,
            'overflow_cost_multiplier': 2.0,
            'emergency_protocol_cost': 1000.0,
            
            # Compliance parameters
            'min_staff_patient_ratio': 0.1,
            'max_wait_time_threshold': 4.0,  # hours
            'min_compliance_score': 0.8,
            'min_safety_score': 0.9,
            
            # Reward weights
            'reward_weights': {
                'patient_satisfaction': 1.0,
                'cost_efficiency': 0.5,
                'compliance': 2.0,
                'safety': 3.0,
                'quality': 1.5
            }
        }
    
    def _define_action_space(self) -> spaces.Box:
        """Define the action space for the environment."""
        # Actions: [add_staff, remove_staff, add_beds, remove_beds, 
        #          add_equipment, remove_equipment, activate_overflow, 
        #          activate_emergency_protocol]
        return spaces.Box(
            low=np.array([-10, -10, -20, -20, -5, -5, 0, 0]),
            high=np.array([10, 10, 20, 20, 5, 5, 1, 1]),
            dtype=np.float32
        )
    
    def _define_observation_space(self) -> spaces.Box:
        """Define the observation space for the environment."""
        # State: [current_patients, current_staff, current_beds, current_wait_time,
        #         staff_utilization, bed_utilization, equipment_utilization,
        #         hour_of_day, day_of_week, is_weekend, is_holiday,
        #         predicted_patients_1h, predicted_patients_4h, predicted_patients_24h,
        #         compliance_score, safety_score, quality_score]
        return spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([1000, 200, 500, 24, 1, 1, 1, 23, 6, 1, 1, 1000, 1000, 1000, 1, 1, 1]),
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'total_cost': 0.0,
            'total_patients_served': 0,
            'compliance_violations': 0,
            'safety_incidents': 0
        }
        
        # Initialize state from historical data
        self.current_state = self._get_initial_state()
        
        logger.debug(f"Environment reset. Initial state: {self.current_state}")
        return self._state_to_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action array from the RL agent
            
        Returns:
            observation: Current state observation
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        # Parse action
        parsed_action = self._parse_action(action)
        
        # Apply action and get new state
        new_state = self._apply_action(parsed_action)
        
        # Calculate reward
        reward = self._calculate_reward(new_state, parsed_action)
        
        # Update episode statistics
        self._update_episode_stats(new_state, parsed_action, reward)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Move to next step
        self.current_step += 1
        self.current_state = new_state
        
        # Prepare info
        info = {
            'step': self.current_step,
            'action': parsed_action,
            'reward': reward,
            'stats': self.episode_stats.copy()
        }
        
        return self._state_to_observation(), reward, done, info
    
    def _get_initial_state(self) -> HealthcareWorkloadState:
        """Get initial state from historical data."""
        if self.current_step >= len(self.historical_data):
            raise ValueError("Current step exceeds historical data length")
        
        row = self.historical_data.iloc[self.current_step]
        
        return HealthcareWorkloadState(
            current_patients=int(row.get('patient_count', 50)),
            current_staff=int(row.get('staff_count', 20)),
            current_beds=int(row.get('bed_count', 100)),
            current_wait_time=float(row.get('wait_time', 1.0)),
            staff_utilization=float(row.get('staff_utilization', 0.7)),
            bed_utilization=float(row.get('bed_utilization', 0.6)),
            equipment_utilization=float(row.get('equipment_utilization', 0.5)),
            hour_of_day=int(row.get('hour', 8)),
            day_of_week=int(row.get('day_of_week', 0)),
            is_weekend=bool(row.get('is_weekend', False)),
            is_holiday=bool(row.get('is_holiday', False)),
            predicted_patients_1h=float(row.get('predicted_patients_1h', 50)),
            predicted_patients_4h=float(row.get('predicted_patients_4h', 55)),
            predicted_patients_24h=float(row.get('predicted_patients_24h', 60)),
            compliance_score=1.0,
            safety_score=1.0,
            quality_score=1.0
        )
    
    def _parse_action(self, action: np.ndarray) -> HealthcareWorkloadAction:
        """Parse raw action array into structured action."""
        return HealthcareWorkloadAction(
            add_staff=int(action[0]),
            remove_staff=int(action[1]),
            add_beds=int(action[2]),
            remove_beds=int(action[3]),
            add_equipment=int(action[4]),
            remove_equipment=int(action[5]),
            activate_overflow=bool(action[6] > 0.5),
            activate_emergency_protocol=bool(action[7] > 0.5)
        )
    
    def _apply_action(self, action: HealthcareWorkloadAction) -> HealthcareWorkloadState:
        """Apply action to current state and return new state."""
        # Get next historical data point
        next_step = min(self.current_step + 1, len(self.historical_data) - 1)
        next_row = self.historical_data.iloc[next_step]
        
        # Calculate new resource levels
        new_staff = max(
            self.config['min_staff'],
            min(
                self.config['max_staff'],
                self.current_state.current_staff + action.add_staff - action.remove_staff
            )
        )
        
        new_beds = max(
            self.config['min_beds'],
            min(
                self.config['max_beds'],
                self.current_state.current_beds + action.add_beds - action.remove_beds
            )
        )
        
        new_equipment = max(
            self.config['min_equipment'],
            min(
                self.config['max_equipment'],
                int(self.current_state.equipment_utilization * 50) + 
                action.add_equipment - action.remove_equipment
            )
        )
        
        # Calculate new patient count (affected by actions)
        base_patients = int(next_row.get('patient_count', 50))
        
        # Adjust based on staffing and capacity
        staff_effect = (new_staff - self.current_state.current_staff) * 0.1
        bed_effect = (new_beds - self.current_state.current_beds) * 0.05
        
        new_patients = max(0, base_patients + staff_effect + bed_effect)
        
        # Calculate new wait time
        staff_patient_ratio = new_staff / max(new_patients, 1)
        bed_patient_ratio = new_beds / max(new_patients, 1)
        
        base_wait_time = float(next_row.get('wait_time', 1.0))
        new_wait_time = max(0, base_wait_time * (1 - staff_patient_ratio * 0.5) * (1 - bed_patient_ratio * 0.3))
        
        # Calculate utilization rates
        new_staff_utilization = min(1.0, new_patients / max(new_staff, 1))
        new_bed_utilization = min(1.0, new_patients / max(new_beds, 1))
        new_equipment_utilization = min(1.0, new_patients / max(new_equipment, 1))
        
        # Calculate compliance and safety scores
        compliance_score = self._calculate_compliance_score(
            new_staff, new_beds, new_patients, new_wait_time
        )
        safety_score = self._calculate_safety_score(
            new_staff, new_beds, new_patients, new_wait_time
        )
        quality_score = self._calculate_quality_score(
            new_staff, new_beds, new_patients, new_wait_time
        )
        
        return HealthcareWorkloadState(
            current_patients=int(new_patients),
            current_staff=new_staff,
            current_beds=new_beds,
            current_wait_time=new_wait_time,
            staff_utilization=new_staff_utilization,
            bed_utilization=new_bed_utilization,
            equipment_utilization=new_equipment_utilization,
            hour_of_day=int(next_row.get('hour', 8)),
            day_of_week=int(next_row.get('day_of_week', 0)),
            is_weekend=bool(next_row.get('is_weekend', False)),
            is_holiday=bool(next_row.get('is_holiday', False)),
            predicted_patients_1h=float(next_row.get('predicted_patients_1h', 50)),
            predicted_patients_4h=float(next_row.get('predicted_patients_4h', 55)),
            predicted_patients_24h=float(next_row.get('predicted_patients_24h', 60)),
            compliance_score=compliance_score,
            safety_score=safety_score,
            quality_score=quality_score
        )
    
    def _calculate_compliance_score(self, staff: int, beds: int, patients: int, wait_time: float) -> float:
        """Calculate compliance score based on current state."""
        # Staff-to-patient ratio compliance
        staff_ratio = staff / max(patients, 1)
        staff_compliance = min(1.0, staff_ratio / self.config['min_staff_patient_ratio'])
        
        # Wait time compliance
        wait_compliance = max(0.0, 1.0 - (wait_time / self.config['max_wait_time_threshold']))
        
        # Overall compliance score
        compliance_score = (staff_compliance + wait_compliance) / 2
        
        return max(0.0, min(1.0, compliance_score))
    
    def _calculate_safety_score(self, staff: int, beds: int, patients: int, wait_time: float) -> float:
        """Calculate safety score based on current state."""
        # Staff adequacy for safety
        staff_safety = min(1.0, staff / max(patients * 0.15, 1))
        
        # Wait time safety (longer waits = lower safety)
        wait_safety = max(0.0, 1.0 - (wait_time / 8.0))  # 8 hours max wait for safety
        
        # Bed adequacy for safety
        bed_safety = min(1.0, beds / max(patients * 1.2, 1))  # 20% buffer for safety
        
        # Overall safety score
        safety_score = (staff_safety + wait_safety + bed_safety) / 3
        
        return max(0.0, min(1.0, safety_score))
    
    def _calculate_quality_score(self, staff: int, beds: int, patients: int, wait_time: float) -> float:
        """Calculate quality score based on current state."""
        # Staff quality (adequate staffing for quality care)
        staff_quality = min(1.0, staff / max(patients * 0.12, 1))
        
        # Wait time quality (shorter waits = better quality)
        wait_quality = max(0.0, 1.0 - (wait_time / 6.0))  # 6 hours max wait for quality
        
        # Resource adequacy for quality
        resource_quality = min(1.0, (staff + beds) / max(patients * 1.5, 1))
        
        # Overall quality score
        quality_score = (staff_quality + wait_quality + resource_quality) / 3
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_reward(self, state: HealthcareWorkloadState, action: HealthcareWorkloadAction) -> float:
        """Calculate reward for the current state and action."""
        weights = self.config['reward_weights']
        
        # Patient satisfaction reward (based on wait time and resource adequacy)
        patient_satisfaction = (
            (1.0 - state.current_wait_time / 8.0) * 0.4 +
            min(1.0, state.current_staff / max(state.current_patients * 0.15, 1)) * 0.3 +
            min(1.0, state.current_beds / max(state.current_patients * 1.2, 1)) * 0.3
        )
        
        # Cost efficiency reward (penalize over-staffing and unnecessary resources)
        staff_cost = state.current_staff * self.config['staff_cost_per_hour']
        bed_cost = state.current_beds * self.config['bed_cost_per_hour']
        equipment_cost = int(state.equipment_utilization * 50) * self.config['equipment_cost_per_hour']
        
        total_cost = staff_cost + bed_cost + equipment_cost
        cost_efficiency = max(0.0, 1.0 - (total_cost / 10000.0))  # Normalize to 0-1
        
        # Compliance reward
        compliance_reward = state.compliance_score
        
        # Safety reward
        safety_reward = state.safety_score
        
        # Quality reward
        quality_reward = state.quality_score
        
        # Calculate weighted reward
        reward = (
            weights['patient_satisfaction'] * patient_satisfaction +
            weights['cost_efficiency'] * cost_efficiency +
            weights['compliance'] * compliance_reward +
            weights['safety'] * safety_reward +
            weights['quality'] * quality_reward
        )
        
        # Penalize extreme actions
        action_penalty = 0.0
        if abs(action.add_staff) > 5 or abs(action.remove_staff) > 5:
            action_penalty -= 0.1
        if abs(action.add_beds) > 10 or abs(action.remove_beds) > 10:
            action_penalty -= 0.1
        
        return reward + action_penalty
    
    def _update_episode_stats(self, state: HealthcareWorkloadState, action: HealthcareWorkloadAction, reward: float):
        """Update episode statistics."""
        self.episode_stats['total_reward'] += reward
        self.episode_stats['total_patients_served'] += state.current_patients
        
        # Calculate costs
        staff_cost = state.current_staff * self.config['staff_cost_per_hour']
        bed_cost = state.current_beds * self.config['bed_cost_per_hour']
        equipment_cost = int(state.equipment_utilization * 50) * self.config['equipment_cost_per_hour']
        self.episode_stats['total_cost'] += staff_cost + bed_cost + equipment_cost
        
        # Track violations
        if state.compliance_score < self.config['min_compliance_score']:
            self.episode_stats['compliance_violations'] += 1
        if state.safety_score < self.config['min_safety_score']:
            self.episode_stats['safety_incidents'] += 1
    
    def _state_to_observation(self) -> np.ndarray:
        """Convert state to observation array."""
        state = self.current_state
        return np.array([
            state.current_patients,
            state.current_staff,
            state.current_beds,
            state.current_wait_time,
            state.staff_utilization,
            state.bed_utilization,
            state.equipment_utilization,
            state.hour_of_day,
            state.day_of_week,
            float(state.is_weekend),
            float(state.is_holiday),
            state.predicted_patients_1h,
            state.predicted_patients_4h,
            state.predicted_patients_24h,
            state.compliance_score,
            state.safety_score,
            state.quality_score
        ], dtype=np.float32)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get current episode statistics."""
        return self.episode_stats.copy()
    
    def render(self, mode='human'):
        """Render the current state (for debugging)."""
        if mode == 'human':
            state = self.current_state
            print(f"\n=== Healthcare Workload Environment (Step {self.current_step}) ===")
            print(f"Patients: {state.current_patients}, Staff: {state.current_staff}, Beds: {state.current_beds}")
            print(f"Wait Time: {state.current_wait_time:.2f}h, Staff Util: {state.staff_utilization:.2f}")
            print(f"Compliance: {state.compliance_score:.2f}, Safety: {state.safety_score:.2f}, Quality: {state.quality_score:.2f}")
            print(f"Total Reward: {self.episode_stats['total_reward']:.2f}")
            print(f"Total Cost: ${self.episode_stats['total_cost']:.2f}")
            print("=" * 60)
