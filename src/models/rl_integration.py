"""
Integration Module for Healthcare Workload Management

This module integrates the Phase 2A prediction models with the Phase 2B RL control system
to create a complete end-to-end healthcare workload optimization pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
import os
from datetime import datetime, timedelta

# Import Phase 2A components
from .feature_engineering import FeatureEngineer
from .baseline_models import HealthcareBaselineModels
from .advanced_models import AdvancedHealthcareModels
from .model_evaluation import HealthcareModelEvaluator

# Import Phase 2B components
from .rl_environment import HealthcareWorkloadEnvironment, HealthcareWorkloadState
from .ppo_agent import PPOHHealthcareAgent, PPOConfig

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for the integrated system."""
    
    # Prediction model settings
    prediction_horizon_hours: int = 72
    prediction_update_frequency_minutes: int = 15
    
    # RL training settings
    rl_training_episodes: int = 1000
    rl_evaluation_episodes: int = 100
    rl_update_frequency: int = 10  # Update policy every N episodes
    
    # Integration settings
    use_prediction_for_rl: bool = True
    prediction_confidence_threshold: float = 0.8
    fallback_to_baseline: bool = True
    
    # Performance thresholds
    min_prediction_accuracy: float = 0.85  # 85% accuracy threshold
    min_rl_performance: float = 0.7  # 70% performance threshold
    
    # Output settings
    save_intermediate_results: bool = True
    generate_reports: bool = True
    log_detailed_metrics: bool = True


class HealthcareWorkloadOptimizer:
    """
    Integrated healthcare workload optimization system.
    
    This class combines prediction models from Phase 2A with RL control
    from Phase 2B to create a complete optimization pipeline.
    """
    
    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        prediction_model_path: Optional[str] = None,
        rl_model_path: Optional[str] = None
    ):
        """
        Initialize the integrated optimizer.
        
        Args:
            config: Integration configuration
            prediction_model_path: Path to saved prediction model
            rl_model_path: Path to saved RL model
        """
        self.config = config or IntegrationConfig()
        
        # Initialize Phase 2A components
        self.feature_engineer = FeatureEngineer()
        self.baseline_models = HealthcareBaselineModels()
        self.advanced_models = AdvancedHealthcareModels()
        self.model_evaluator = HealthcareModelEvaluator()
        
        # Initialize Phase 2B components
        self.rl_environment = None
        self.rl_agent = None
        
        # Load models if provided
        self.prediction_model = None
        self.best_prediction_model_name = None
        
        if prediction_model_path and os.path.exists(prediction_model_path):
            self._load_prediction_model(prediction_model_path)
        
        if rl_model_path and os.path.exists(rl_model_path):
            self._load_rl_model(rl_model_path)
        
        # Performance tracking
        self.performance_metrics = {
            'prediction_accuracy': [],
            'rl_performance': [],
            'overall_optimization_score': [],
            'compliance_violations': [],
            'cost_savings': []
        }
        
        # Results storage
        self.optimization_results = []
        
        logger.info("Healthcare Workload Optimizer initialized")
    
    def _load_prediction_model(self, model_path: str):
        """Load a saved prediction model."""
        try:
            # Load model evaluation results
            with open(model_path, 'r') as f:
                results = json.load(f)
            
            # Get best model
            if 'comparison' in results and 'summary' in results['comparison']:
                self.best_prediction_model_name = results['comparison']['summary']['overall_best']
                logger.info(f"Loaded prediction model: {self.best_prediction_model_name}")
            else:
                logger.warning("Could not determine best prediction model from results")
                
        except Exception as e:
            logger.error(f"Error loading prediction model: {e}")
    
    def _load_rl_model(self, model_path: str):
        """Load a saved RL model."""
        try:
            # This would be implemented based on the specific model format
            # For now, we'll initialize a new agent and load weights later
            logger.info(f"RL model loading from {model_path} - to be implemented")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
    
    def prepare_prediction_data(
        self, 
        encounters_df: pd.DataFrame,
        patients_df: pd.DataFrame,
        conditions_df: pd.DataFrame,
        medications_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for prediction using Phase 2A feature engineering.
        
        Args:
            encounters_df: Encounters data
            patients_df: Patients data
            conditions_df: Conditions data
            medications_df: Medications data
            
        Returns:
            feature_df: Engineered features for prediction
        """
        logger.info("Preparing prediction data with feature engineering")
        
        # Engineer features using Phase 2A
        feature_df = self.feature_engineer.engineer_features(
            encounters_df, patients_df, conditions_df, medications_df
        )
        
        # Scale features
        feature_df_scaled = self.feature_engineer.scale_features(feature_df)
        
        logger.info(f"Feature engineering completed. Shape: {feature_df_scaled.shape}")
        return feature_df_scaled
    
    def train_prediction_models(
        self, 
        encounters_df: pd.DataFrame,
        patients_df: pd.DataFrame,
        conditions_df: pd.DataFrame,
        medications_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train prediction models using Phase 2A components.
        
        Args:
            encounters_df: Encounters data
            patients_df: Patients data
            conditions_df: Conditions data
            medications_df: Medications data
            
        Returns:
            results: Training results and model performance
        """
        logger.info("Training prediction models")
        
        # Prepare features
        feature_df = self.prepare_prediction_data(
            encounters_df, patients_df, conditions_df, medications_df
        )
        
        # Train baseline models
        baseline_results = self.baseline_models.train_all_baseline_models(encounters_df)
        
        # Train advanced models
        advanced_results = self.advanced_models.train_all_advanced_models(feature_df)
        
        # Run complete evaluation
        evaluation_results = self.model_evaluator.run_complete_evaluation(
            encounters_df, patients_df, conditions_df, medications_df
        )
        
        # Store best model
        if evaluation_results['comparison']['summary']['overall_best']:
            self.best_prediction_model_name = evaluation_results['comparison']['summary']['overall_best']
            self.prediction_model = evaluation_results['models'][self.best_prediction_model_name]
        
        logger.info(f"Prediction model training completed. Best model: {self.best_prediction_model_name}")
        return evaluation_results
    
    def predict_workload(
        self, 
        current_data: pd.DataFrame,
        horizon_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict future workload using the best prediction model.
        
        Args:
            current_data: Current healthcare data
            horizon_hours: Prediction horizon in hours
            
        Returns:
            predictions: Workload predictions for the specified horizon
        """
        if self.prediction_model is None:
            logger.error("No prediction model available")
            return {}
        
        horizon_hours = horizon_hours or self.config.prediction_horizon_hours
        
        try:
            # Prepare features for prediction
            feature_df = self.feature_engineer.engineer_features(
                current_data['encounters'], 
                current_data['patients'],
                current_data['conditions'],
                current_data['medications']
            )
            
            # Make predictions
            if self.best_prediction_model_name in ['linear_regression', 'ridge_regression', 'lasso_regression']:
                predictions = self.prediction_model.predict(feature_df)
            elif self.best_prediction_model_name in ['random_forest', 'gradient_boosting']:
                predictions = self.prediction_model.predict(feature_df)
            else:
                # For ensemble models, use predict method
                predictions = self.prediction_model.predict(feature_df)
            
            # Create prediction timeline
            prediction_timeline = []
            current_time = datetime.now()
            
            for i, pred in enumerate(predictions[:horizon_hours]):
                prediction_time = current_time + timedelta(hours=i)
                prediction_timeline.append({
                    'timestamp': prediction_time,
                    'predicted_patients': max(0, int(pred)),
                    'confidence': self._calculate_prediction_confidence(pred, i)
                })
            
            return {
                'predictions': prediction_timeline,
                'model_name': self.best_prediction_model_name,
                'horizon_hours': horizon_hours,
                'confidence_threshold': self.config.prediction_confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {}
    
    def _calculate_prediction_confidence(self, prediction: float, hour_ahead: int) -> float:
        """Calculate confidence in prediction based on time horizon."""
        # Confidence decreases with time horizon
        base_confidence = 0.9
        decay_factor = 0.95 ** hour_ahead
        return max(0.1, base_confidence * decay_factor)
    
    def initialize_rl_environment(
        self, 
        historical_data: pd.DataFrame,
        prediction_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RL environment with historical data and predictions.
        
        Args:
            historical_data: Historical healthcare workload data
            prediction_data: Optional prediction data to enhance environment
        """
        logger.info("Initializing RL environment")
        
        # Enhance historical data with predictions if available
        if prediction_data and self.config.use_prediction_for_rl:
            enhanced_data = self._enhance_data_with_predictions(historical_data, prediction_data)
        else:
            enhanced_data = historical_data
        
        # Initialize environment
        self.rl_environment = HealthcareWorkloadEnvironment(enhanced_data)
        
        # Initialize RL agent
        state_dim = self.rl_environment.observation_space.shape[0]
        action_dim = self.rl_environment.action_space.shape[0]
        
        rl_config = PPOConfig()
        self.rl_agent = PPOHHealthcareAgent(state_dim, action_dim, rl_config)
        
        logger.info(f"RL environment initialized with {state_dim} states and {action_dim} actions")
    
    def _enhance_data_with_predictions(
        self, 
        historical_data: pd.DataFrame, 
        prediction_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Enhance historical data with prediction information."""
        enhanced_data = historical_data.copy()
        
        # Add prediction columns
        enhanced_data['predicted_patients_1h'] = 0.0
        enhanced_data['predicted_patients_4h'] = 0.0
        enhanced_data['predicted_patients_24h'] = 0.0
        
        # Fill prediction data where available
        for i, pred in enumerate(prediction_data['predictions']):
            if i < len(enhanced_data):
                enhanced_data.loc[i, 'predicted_patients_1h'] = pred['predicted_patients']
                if i + 4 < len(enhanced_data):
                    enhanced_data.loc[i, 'predicted_patients_4h'] = pred['predicted_patients']
                if i + 24 < len(enhanced_data):
                    enhanced_data.loc[i, 'predicted_patients_24h'] = pred['predicted_patients']
        
        return enhanced_data
    
    def train_rl_agent(self) -> Dict[str, Any]:
        """
        Train the RL agent for healthcare workload optimization.
        
        Returns:
            training_results: Training statistics and performance metrics
        """
        if self.rl_environment is None or self.rl_agent is None:
            logger.error("RL environment or agent not initialized")
            return {}
        
        logger.info(f"Starting RL training for {self.config.rl_training_episodes} episodes")
        
        training_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'compliance_violations': [],
            'safety_violations': [],
            'cost_savings': []
        }
        
        for episode in range(self.config.rl_training_episodes):
            # Reset environment
            state = self.rl_environment.reset()
            episode_reward = 0
            episode_length = 0
            compliance_violations = 0
            safety_violations = 0
            
            done = False
            while not done:
                # Get current state dict for compliance checking
                current_state_dict = {
                    'current_patients': self.rl_environment.current_state.current_patients,
                    'current_staff': self.rl_environment.current_state.current_staff,
                    'current_beds': self.rl_environment.current_state.current_beds,
                    'current_wait_time': self.rl_environment.current_state.current_wait_time
                }
                
                # Select action
                action, action_info = self.rl_agent.select_action(state, current_state_dict)
                
                # Take step in environment
                next_state, reward, done, info = self.rl_environment.step(action)
                
                # Store transition
                self.rl_agent.store_transition(
                    state, action, reward, action_info['value'], 
                    action_info['log_prob'], action_info['entropy'], done
                )
                
                # Update episode statistics
                episode_reward += reward
                episode_length += 1
                
                if not action_info['is_compliant']:
                    compliance_violations += 1
                
                if self.rl_environment.current_state.safety_score < 0.9:
                    safety_violations += 1
                
                state = next_state
            
            # Log episode statistics
            self.rl_agent.log_episode_stats(
                episode_reward, episode_length, compliance_violations, safety_violations
            )
            
            # Store training results
            training_results['episode_rewards'].append(episode_reward)
            training_results['episode_lengths'].append(episode_length)
            training_results['compliance_violations'].append(compliance_violations)
            training_results['safety_violations'].append(safety_violations)
            
            # Calculate cost savings (simplified)
            baseline_cost = episode_length * 1000  # Baseline cost per step
            actual_cost = self.rl_environment.get_episode_stats()['total_cost']
            cost_savings = (baseline_cost - actual_cost) / baseline_cost
            training_results['cost_savings'].append(cost_savings)
            
            # Update policy periodically
            if (episode + 1) % self.config.rl_update_frequency == 0:
                update_stats = self.rl_agent.update_policy()
                logger.debug(f"Policy updated at episode {episode + 1}: {update_stats}")
            
            # Log progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(training_results['episode_rewards'][-100:])
                avg_compliance = 1 - np.mean(training_results['compliance_violations'][-100:])
                logger.info(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Compliance = {avg_compliance:.2f}")
        
        logger.info("RL training completed")
        return training_results
    
    def evaluate_integrated_system(
        self, 
        test_data: pd.DataFrame,
        num_episodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the integrated prediction + RL system.
        
        Args:
            test_data: Test data for evaluation
            num_episodes: Number of evaluation episodes
            
        Returns:
            evaluation_results: Comprehensive evaluation results
        """
        num_episodes = num_episodes or self.config.rl_evaluation_episodes
        
        logger.info(f"Evaluating integrated system with {num_episodes} episodes")
        
        # Initialize environment with test data
        self.initialize_rl_environment(test_data)
        
        evaluation_results = {
            'prediction_accuracy': 0.0,
            'rl_performance': 0.0,
            'overall_optimization_score': 0.0,
            'compliance_rate': 0.0,
            'safety_rate': 0.0,
            'cost_efficiency': 0.0,
            'episode_details': []
        }
        
        total_rewards = []
        total_compliance_violations = 0
        total_safety_violations = 0
        total_cost_savings = 0
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.rl_environment.reset()
            episode_reward = 0
            episode_compliance_violations = 0
            episode_safety_violations = 0
            
            done = False
            while not done:
                # Get current state dict
                current_state_dict = {
                    'current_patients': self.rl_environment.current_state.current_patients,
                    'current_staff': self.rl_environment.current_state.current_staff,
                    'current_beds': self.rl_environment.current_state.current_beds,
                    'current_wait_time': self.rl_environment.current_state.current_wait_time
                }
                
                # Select action
                action, action_info = self.rl_agent.select_action(state, current_state_dict)
                
                # Take step
                next_state, reward, done, info = self.rl_environment.step(action)
                
                # Update episode statistics
                episode_reward += reward
                
                if not action_info['is_compliant']:
                    episode_compliance_violations += 1
                
                if self.rl_environment.current_state.safety_score < 0.9:
                    episode_safety_violations += 1
                
                state = next_state
            
            # Store episode results
            total_rewards.append(episode_reward)
            total_compliance_violations += episode_compliance_violations
            total_safety_violations += episode_safety_violations
            
            # Calculate cost savings
            baseline_cost = len(self.rl_environment.historical_data) * 1000
            actual_cost = self.rl_environment.get_episode_stats()['total_cost']
            cost_savings = (baseline_cost - actual_cost) / baseline_cost
            total_cost_savings += cost_savings
            
            # Store episode details
            evaluation_results['episode_details'].append({
                'episode': episode,
                'reward': episode_reward,
                'compliance_violations': episode_compliance_violations,
                'safety_violations': episode_safety_violations,
                'cost_savings': cost_savings
            })
        
        # Calculate overall metrics
        evaluation_results['rl_performance'] = np.mean(total_rewards)
        evaluation_results['compliance_rate'] = 1 - (total_compliance_violations / (num_episodes * 100))  # Assuming 100 steps per episode
        evaluation_results['safety_rate'] = 1 - (total_safety_violations / (num_episodes * 100))
        evaluation_results['cost_efficiency'] = total_cost_savings / num_episodes
        
        # Overall optimization score (weighted combination)
        evaluation_results['overall_optimization_score'] = (
            0.3 * evaluation_results['rl_performance'] +
            0.3 * evaluation_results['compliance_rate'] +
            0.2 * evaluation_results['safety_rate'] +
            0.2 * evaluation_results['cost_efficiency']
        )
        
        logger.info(f"Evaluation completed. Overall score: {evaluation_results['overall_optimization_score']:.3f}")
        return evaluation_results
    
    def optimize_workload(
        self, 
        current_data: pd.DataFrame,
        optimization_horizon_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform end-to-end workload optimization.
        
        Args:
            current_data: Current healthcare data
            optimization_horizon_hours: Optimization horizon in hours
            
        Returns:
            optimization_results: Complete optimization results and recommendations
        """
        logger.info("Starting end-to-end workload optimization")
        
        optimization_horizon_hours = optimization_horizon_hours or self.config.prediction_horizon_hours
        
        # Step 1: Make predictions
        predictions = self.predict_workload(current_data, optimization_horizon_hours)
        
        # Step 2: Initialize RL environment with predictions
        self.initialize_rl_environment(current_data, predictions)
        
        # Step 3: Train RL agent
        training_results = self.train_rl_agent()
        
        # Step 4: Evaluate integrated system
        evaluation_results = self.evaluate_integrated_system(current_data)
        
        # Step 5: Generate optimization recommendations
        recommendations = self._generate_recommendations(predictions, evaluation_results)
        
        # Compile results
        optimization_results = {
            'predictions': predictions,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'optimization_horizon_hours': optimization_horizon_hours
        }
        
        # Store results
        self.optimization_results.append(optimization_results)
        
        # Save results if configured
        if self.config.save_intermediate_results:
            self._save_optimization_results(optimization_results)
        
        logger.info("Workload optimization completed")
        return optimization_results
    
    def _generate_recommendations(
        self, 
        predictions: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization recommendations based on predictions and evaluation."""
        recommendations = {
            'staffing_recommendations': [],
            'bed_management_recommendations': [],
            'equipment_recommendations': [],
            'operational_recommendations': [],
            'risk_alerts': []
        }
        
        # Analyze predictions for staffing needs
        if predictions and 'predictions' in predictions:
            for pred in predictions['predictions']:
                if pred['predicted_patients'] > 100:  # High workload threshold
                    recommendations['staffing_recommendations'].append({
                        'timestamp': pred['timestamp'],
                        'action': 'Increase staffing',
                        'reason': f"Predicted {pred['predicted_patients']} patients",
                        'priority': 'High' if pred['predicted_patients'] > 150 else 'Medium'
                    })
        
        # Add recommendations based on evaluation results
        if evaluation_results['compliance_rate'] < 0.9:
            recommendations['risk_alerts'].append({
                'type': 'Compliance Risk',
                'message': f"Compliance rate {evaluation_results['compliance_rate']:.2f} below threshold 0.9",
                'priority': 'High'
            })
        
        if evaluation_results['safety_rate'] < 0.95:
            recommendations['risk_alerts'].append({
                'type': 'Safety Risk',
                'message': f"Safety rate {evaluation_results['safety_rate']:.2f} below threshold 0.95",
                'priority': 'Critical'
            })
        
        return recommendations
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """Save optimization results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.json"
        filepath = os.path.join("models", filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Optimization results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of system performance."""
        if not self.optimization_results:
            return {}
        
        latest_results = self.optimization_results[-1]
        
        return {
            'latest_optimization_score': latest_results['evaluation_results']['overall_optimization_score'],
            'prediction_accuracy': latest_results.get('predictions', {}).get('model_name', 'Unknown'),
            'compliance_rate': latest_results['evaluation_results']['compliance_rate'],
            'safety_rate': latest_results['evaluation_results']['safety_rate'],
            'cost_efficiency': latest_results['evaluation_results']['cost_efficiency'],
            'total_optimizations': len(self.optimization_results),
            'last_optimization': latest_results['timestamp']
        }
