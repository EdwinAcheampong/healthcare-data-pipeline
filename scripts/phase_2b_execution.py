#!/usr/bin/env python3
"""
Phase 2B Execution Script: RL System Development

This script executes the complete Phase 2B implementation including:
1. RL Environment Setup
2. PPO Agent Training
3. Healthcare Compliance Integration
4. End-to-End Optimization Pipeline
"""

import sys
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.rl_integration import HealthcareWorkloadOptimizer, IntegrationConfig
from models.rl_environment import HealthcareWorkloadEnvironment
from models.ppo_agent import PPOHHealthcareAgent, PPOConfig
from data_pipeline.etl import ETLPipeline
from utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def load_phase_2a_results():
    """Load Phase 2A results and best model."""
    try:
        # Load Phase 2A evaluation results
        phase_2a_results_path = "models/phase_2a_test_results.json"
        if os.path.exists(phase_2a_results_path):
            with open(phase_2a_results_path, 'r') as f:
                phase_2a_results = json.load(f)
            
            logger.info("Phase 2A results loaded successfully")
            return phase_2a_results
        else:
            logger.warning("Phase 2A results not found, will train new models")
            return None
    except Exception as e:
        logger.error(f"Error loading Phase 2A results: {e}")
        return None


def prepare_training_data():
    """Prepare training data for RL system."""
    try:
        # Initialize ETL pipeline
        etl = ETLPipeline()
        
        # Load processed data
        logger.info("Loading processed healthcare data")
        
        encounters_df = pd.read_parquet("data/processed/parquet/encounters.parquet")
        patients_df = pd.read_parquet("data/processed/parquet/patients.parquet")
        conditions_df = pd.read_parquet("data/processed/parquet/conditions.parquet")
        medications_df = pd.read_parquet("data/processed/parquet/medications.parquet")
        
        # Create synthetic workload data for RL training
        logger.info("Creating synthetic workload data for RL training")
        
        # Generate time series data with realistic patterns
        n_samples = 1000
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        # Create realistic workload patterns
        base_patients = 50
        hourly_pattern = np.sin(np.arange(n_samples) * 2 * np.pi / 24) * 20  # Daily cycle
        weekly_pattern = np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 7)) * 10  # Weekly cycle
        noise = np.random.normal(0, 5, n_samples)
        
        patient_counts = np.maximum(10, base_patients + hourly_pattern + weekly_pattern + noise)
        
        # Create synthetic workload data
        workload_data = pd.DataFrame({
            'timestamp': timestamps,
            'patient_count': patient_counts.astype(int),
            'staff_count': np.maximum(5, (patient_counts * 0.15).astype(int)),
            'bed_count': np.maximum(20, (patient_counts * 1.2).astype(int)),
            'wait_time': np.maximum(0.1, np.random.exponential(2, n_samples)),
            'staff_utilization': np.random.uniform(0.3, 0.9, n_samples),
            'bed_utilization': np.random.uniform(0.4, 0.8, n_samples),
            'equipment_utilization': np.random.uniform(0.2, 0.7, n_samples),
            'hour': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'is_weekend': timestamps.dayofweek >= 5,
            'is_holiday': False,  # Simplified for now
            'predicted_patients_1h': patient_counts + np.random.normal(0, 3, n_samples),
            'predicted_patients_4h': patient_counts + np.random.normal(0, 5, n_samples),
            'predicted_patients_24h': patient_counts + np.random.normal(0, 8, n_samples)
        })
        
        logger.info(f"Created synthetic workload data with {len(workload_data)} samples")
        
        return {
            'encounters': encounters_df,
            'patients': patients_df,
            'conditions': conditions_df,
            'medications': medications_df,
            'workload': workload_data
        }
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        raise


def test_rl_environment():
    """Test the RL environment functionality."""
    logger.info("Testing RL Environment")
    
    try:
        # Create synthetic data for testing
        n_samples = 100
        test_data = pd.DataFrame({
            'patient_count': np.random.randint(20, 100, n_samples),
            'staff_count': np.random.randint(10, 50, n_samples),
            'bed_count': np.random.randint(30, 150, n_samples),
            'wait_time': np.random.uniform(0.5, 4.0, n_samples),
            'staff_utilization': np.random.uniform(0.3, 0.9, n_samples),
            'bed_utilization': np.random.uniform(0.4, 0.8, n_samples),
            'equipment_utilization': np.random.uniform(0.2, 0.7, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([True, False], n_samples),
            'is_holiday': np.random.choice([True, False], n_samples),
            'predicted_patients_1h': np.random.randint(20, 100, n_samples),
            'predicted_patients_4h': np.random.randint(20, 100, n_samples),
            'predicted_patients_24h': np.random.randint(20, 100, n_samples)
        })
        
        # Initialize environment
        env = HealthcareWorkloadEnvironment(test_data)
        
        # Test environment reset
        initial_state = env.reset()
        logger.info(f"Environment reset successful. State shape: {initial_state.shape}")
        
        # Test environment step
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        next_state, reward, done, info = env.step(action)
        
        logger.info(f"Environment step successful:")
        logger.info(f"  - Next state shape: {next_state.shape}")
        logger.info(f"  - Reward: {reward:.3f}")
        logger.info(f"  - Done: {done}")
        logger.info(f"  - Info keys: {list(info.keys())}")
        
        # Test multiple steps
        total_reward = 0
        step_count = 0
        
        while not done and step_count < 50:
            action = np.random.uniform(-1, 1, env.action_space.shape[0])
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        logger.info(f"Environment test completed:")
        logger.info(f"  - Total steps: {step_count}")
        logger.info(f"  - Total reward: {total_reward:.3f}")
        logger.info(f"  - Episode stats: {env.get_episode_stats()}")
        
        return True
        
    except Exception as e:
        logger.error(f"RL Environment test failed: {e}")
        return False


def test_ppo_agent():
    """Test the PPO agent functionality."""
    logger.info("Testing PPO Agent")
    
    try:
        # Create test data
        n_samples = 50
        test_data = pd.DataFrame({
            'patient_count': np.random.randint(20, 100, n_samples),
            'staff_count': np.random.randint(10, 50, n_samples),
            'bed_count': np.random.randint(30, 150, n_samples),
            'wait_time': np.random.uniform(0.5, 4.0, n_samples),
            'staff_utilization': np.random.uniform(0.3, 0.9, n_samples),
            'bed_utilization': np.random.uniform(0.4, 0.8, n_samples),
            'equipment_utilization': np.random.uniform(0.2, 0.7, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([True, False], n_samples),
            'is_holiday': np.random.choice([True, False], n_samples),
            'predicted_patients_1h': np.random.randint(20, 100, n_samples),
            'predicted_patients_4h': np.random.randint(20, 100, n_samples),
            'predicted_patients_24h': np.random.randint(20, 100, n_samples)
        })
        
        # Initialize environment and agent
        env = HealthcareWorkloadEnvironment(test_data)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = PPOHHealthcareAgent(state_dim, action_dim)
        
        logger.info(f"PPO Agent initialized with {state_dim} states and {action_dim} actions")
        
        # Test action selection
        state = env.reset()
        action, action_info = agent.select_action(state)
        
        logger.info(f"Action selection successful:")
        logger.info(f"  - Action shape: {action.shape}")
        logger.info(f"  - Action info keys: {list(action_info.keys())}")
        logger.info(f"  - Is compliant: {action_info['is_compliant']}")
        
        # Test training loop (short version)
        episode_rewards = []
        
        for episode in range(5):  # Short test
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, action_info = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(
                    state, action, reward, action_info['value'],
                    action_info['log_prob'], action_info['entropy'], done
                )
                
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.3f}")
        
        # Test policy update
        if len(agent.states) > 0:
            update_stats = agent.update_policy()
            logger.info(f"Policy update successful: {update_stats}")
        
        logger.info(f"PPO Agent test completed. Average reward: {np.mean(episode_rewards):.3f}")
        return True
        
    except Exception as e:
        logger.error(f"PPO Agent test failed: {e}")
        return False


def run_phase_2b_integration():
    """Run the complete Phase 2B integration."""
    logger.info("Starting Phase 2B Integration")
    
    try:
        # Load Phase 2A results
        phase_2a_results = load_phase_2a_results()
        
        # Prepare training data
        data = prepare_training_data()
        
        # Initialize optimizer with Phase 2A results
        config = IntegrationConfig(
            rl_training_episodes=100,  # Reduced for testing
            rl_evaluation_episodes=20,
            prediction_horizon_hours=24
        )
        
        optimizer = HealthcareWorkloadOptimizer(
            config=config,
            prediction_model_path="models/phase_2a_test_results.json" if phase_2a_results else None
        )
        
        # Train prediction models if needed
        if not phase_2a_results:
            logger.info("Training prediction models (Phase 2A)")
            prediction_results = optimizer.train_prediction_models(
                data['encounters'], data['patients'], data['conditions'], data['medications']
            )
        else:
            logger.info("Using existing Phase 2A prediction models")
        
        # Run end-to-end optimization
        logger.info("Running end-to-end workload optimization")
        
        optimization_results = optimizer.optimize_workload(
            current_data=data['workload'],
            optimization_horizon_hours=24
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"models/phase_2b_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        logger.info(f"Phase 2B results saved to {results_file}")
        
        # Print summary
        performance_summary = optimizer.get_performance_summary()
        logger.info("Phase 2B Performance Summary:")
        logger.info(f"  - Overall Optimization Score: {performance_summary.get('latest_optimization_score', 'N/A'):.3f}")
        logger.info(f"  - Compliance Rate: {performance_summary.get('compliance_rate', 'N/A'):.3f}")
        logger.info(f"  - Safety Rate: {performance_summary.get('safety_rate', 'N/A'):.3f}")
        logger.info(f"  - Cost Efficiency: {performance_summary.get('cost_efficiency', 'N/A'):.3f}")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"Phase 2B integration failed: {e}")
        raise


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 2B EXECUTION: RL System Development")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Test individual components
        logger.info("Testing individual components...")
        
        env_test_passed = test_rl_environment()
        agent_test_passed = test_ppo_agent()
        
        if not env_test_passed or not agent_test_passed:
            logger.error("Component tests failed. Aborting Phase 2B execution.")
            return False
        
        logger.info("All component tests passed!")
        
        # Run complete integration
        logger.info("Running complete Phase 2B integration...")
        results = run_phase_2b_integration()
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("=" * 60)
        logger.info("PHASE 2B EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Results saved to: models/phase_2b_results_*.json")
        
        # Check if results meet Phase 2B criteria
        if results and 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            overall_score = eval_results.get('overall_optimization_score', 0)
            compliance_rate = eval_results.get('compliance_rate', 0)
            safety_rate = eval_results.get('safety_rate', 0)
            
            logger.info("Phase 2B Success Criteria Check:")
            logger.info(f"  - Overall Optimization Score: {overall_score:.3f} (Target: >0.7)")
            logger.info(f"  - Compliance Rate: {compliance_rate:.3f} (Target: >0.9)")
            logger.info(f"  - Safety Rate: {safety_rate:.3f} (Target: >0.95)")
            
            if overall_score > 0.7 and compliance_rate > 0.9 and safety_rate > 0.95:
                logger.info("✅ Phase 2B SUCCESS CRITERIA MET!")
            else:
                logger.warning("⚠️  Phase 2B success criteria not fully met. Consider additional training.")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 2B execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
