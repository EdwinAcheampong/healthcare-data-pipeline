#!/usr/bin/env python3
"""
Phase 2B Test Script

This script tests all Phase 2B components including:
1. RL Environment
2. PPO Agent
3. Healthcare Compliance
4. Integration System
"""

import sys
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
import unittest
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.rl_environment import HealthcareWorkloadEnvironment, HealthcareWorkloadState, HealthcareWorkloadAction
from models.ppo_agent import PPOHHealthcareAgent, PPOConfig, HealthcareComplianceChecker
from models.rl_integration import HealthcareWorkloadOptimizer, IntegrationConfig
from utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class TestRLEnvironment(unittest.TestCase):
    """Test cases for RL Environment."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data
        n_samples = 100
        self.test_data = pd.DataFrame({
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
        
        self.env = HealthcareWorkloadEnvironment(self.test_data)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env)
        self.assertEqual(len(self.env.historical_data), 100)
        self.assertIsNotNone(self.env.action_space)
        self.assertIsNotNone(self.env.observation_space)
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape[0], 17)  # 17 state dimensions
        self.assertIsNotNone(self.env.current_state)
    
    def test_environment_step(self):
        """Test environment step functionality."""
        state = self.env.reset()
        action = np.random.uniform(-1, 1, self.env.action_space.shape[0])
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(next_state.shape[0], 17)
    
    def test_environment_episode(self):
        """Test complete episode execution."""
        state = self.env.reset()
        total_reward = 0
        step_count = 0
        
        while not self.env.current_step >= self.env.max_steps and step_count < 50:
            action = np.random.uniform(-1, 1, self.env.action_space.shape[0])
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            step_count += 1
        
        self.assertGreater(step_count, 0)
        self.assertIsInstance(total_reward, float)
    
    def test_state_consistency(self):
        """Test state consistency throughout episode."""
        state = self.env.reset()
        initial_patients = self.env.current_state.current_patients
        
        action = np.array([1, 0, 2, 0, 0, 0, 0, 0])  # Add staff and beds
        next_state, reward, done, info = self.env.step(action)
        
        # Check that state changes are reasonable
        new_patients = self.env.current_state.current_patients
        self.assertGreaterEqual(new_patients, 0)  # Patients should be non-negative


class TestPPOAgent(unittest.TestCase):
    """Test cases for PPO Agent."""
    
    def setUp(self):
        """Set up test data and agent."""
        # Create test environment
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
        
        self.env = HealthcareWorkloadEnvironment(test_data)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.agent = PPOHHealthcareAgent(self.state_dim, self.action_dim)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertIsNotNone(self.agent.actor_critic)
        self.assertIsNotNone(self.agent.optimizer)
    
    def test_action_selection(self):
        """Test action selection functionality."""
        state = self.env.reset()
        action, action_info = self.agent.select_action(state)
        
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape[0], self.action_dim)
        self.assertIsInstance(action_info, dict)
        self.assertIn('log_prob', action_info)
        self.assertIn('entropy', action_info)
        self.assertIn('value', action_info)
    
    def test_action_selection_with_compliance(self):
        """Test action selection with compliance checking."""
        state = self.env.reset()
        current_state_dict = {
            'current_patients': self.env.current_state.current_patients,
            'current_staff': self.env.current_state.current_staff,
            'current_beds': self.env.current_state.current_beds,
            'current_wait_time': self.env.current_state.current_wait_time
        }
        
        action, action_info = self.agent.select_action(state, current_state_dict)
        
        self.assertIsInstance(action, np.ndarray)
        self.assertIn('is_compliant', action_info)
        self.assertIn('compliance_info', action_info)
    
    def test_transition_storage(self):
        """Test transition storage functionality."""
        state = self.env.reset()
        action, action_info = self.agent.select_action(state)
        
        # Store transition
        self.agent.store_transition(
            state, action, 1.0, action_info['value'],
            action_info['log_prob'], action_info['entropy'], False
        )
        
        self.assertEqual(len(self.agent.states), 1)
        self.assertEqual(len(self.agent.actions), 1)
        self.assertEqual(len(self.agent.rewards), 1)
    
    def test_policy_update(self):
        """Test policy update functionality."""
        # Store some transitions
        state = self.env.reset()
        for _ in range(10):
            action, action_info = self.agent.select_action(state)
            self.agent.store_transition(
                state, action, 1.0, action_info['value'],
                action_info['log_prob'], action_info['entropy'], False
            )
        
        # Update policy
        update_stats = self.agent.update_policy()
        
        self.assertIsInstance(update_stats, dict)
        self.assertIn('value_loss', update_stats)
        self.assertIn('policy_loss', update_stats)


class TestComplianceChecker(unittest.TestCase):
    """Test cases for Healthcare Compliance Checker."""
    
    def setUp(self):
        """Set up compliance checker."""
        self.config = PPOConfig()
        self.compliance_checker = HealthcareComplianceChecker(self.config)
    
    def test_compliance_checker_initialization(self):
        """Test compliance checker initialization."""
        self.assertIsNotNone(self.compliance_checker)
        self.assertEqual(self.compliance_checker.config, self.config)
    
    def test_action_compliance_checking(self):
        """Test action compliance checking."""
        import torch
        
        # Test compliant action
        compliant_action = torch.tensor([2.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        current_state = {
            'current_patients': 50,
            'current_staff': 20,
            'current_beds': 100
        }
        
        is_compliant, compliance_info = self.compliance_checker.check_action_compliance(
            compliant_action, current_state
        )
        
        self.assertIsInstance(is_compliant, bool)
        self.assertIsInstance(compliance_info, dict)
        self.assertIn('violations', compliance_info)
    
    def test_action_constraint_application(self):
        """Test action constraint application."""
        import torch
        
        # Test action that exceeds limits
        extreme_action = torch.tensor([20.0, 0.0, 50.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        current_state = {
            'current_patients': 50,
            'current_staff': 20,
            'current_beds': 100
        }
        
        constrained_action = self.compliance_checker.apply_compliance_constraints(
            extreme_action, current_state
        )
        
        self.assertIsInstance(constrained_action, torch.Tensor)
        self.assertEqual(constrained_action.shape, extreme_action.shape)


class TestIntegration(unittest.TestCase):
    """Test cases for Integration System."""
    
    def setUp(self):
        """Set up integration test."""
        self.config = IntegrationConfig(
            rl_training_episodes=10,  # Small number for testing
            rl_evaluation_episodes=5,
            prediction_horizon_hours=24
        )
        self.optimizer = HealthcareWorkloadOptimizer(config=self.config)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.config, self.config)
        self.assertIsNotNone(self.optimizer.feature_engineer)
        self.assertIsNotNone(self.optimizer.baseline_models)
        self.assertIsNotNone(self.optimizer.advanced_models)
    
    def test_prediction_data_preparation(self):
        """Test prediction data preparation."""
        # Create test data
        encounters_df = pd.DataFrame({
            'patient': [1, 2, 3],
            'start': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'stop': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'encounterclass': ['emergency', 'outpatient', 'emergency']
        })
        
        patients_df = pd.DataFrame({
            'id': [1, 2, 3],
            'birthdate': ['1990-01-01', '1985-01-01', '1995-01-01'],
            'gender': ['M', 'F', 'M']
        })
        
        conditions_df = pd.DataFrame({
            'patient': [1, 2, 3],
            'code': ['A', 'B', 'C'],
            'description': ['Condition A', 'Condition B', 'Condition C']
        })
        
        medications_df = pd.DataFrame({
            'patient': [1, 2, 3],
            'code': ['M1', 'M2', 'M3'],
            'description': ['Med A', 'Med B', 'Med C']
        })
        
        feature_df = self.optimizer.prepare_prediction_data(
            encounters_df, patients_df, conditions_df, medications_df
        )
        
        self.assertIsInstance(feature_df, pd.DataFrame)
        self.assertGreater(len(feature_df), 0)
    
    def test_rl_environment_initialization(self):
        """Test RL environment initialization."""
        # Create test workload data
        workload_data = pd.DataFrame({
            'patient_count': np.random.randint(20, 100, 50),
            'staff_count': np.random.randint(10, 50, 50),
            'bed_count': np.random.randint(30, 150, 50),
            'wait_time': np.random.uniform(0.5, 4.0, 50),
            'staff_utilization': np.random.uniform(0.3, 0.9, 50),
            'bed_utilization': np.random.uniform(0.4, 0.8, 50),
            'equipment_utilization': np.random.uniform(0.2, 0.7, 50),
            'hour': np.random.randint(0, 24, 50),
            'day_of_week': np.random.randint(0, 7, 50),
            'is_weekend': np.random.choice([True, False], 50),
            'is_holiday': np.random.choice([True, False], 50),
            'predicted_patients_1h': np.random.randint(20, 100, 50),
            'predicted_patients_4h': np.random.randint(20, 100, 50),
            'predicted_patients_24h': np.random.randint(20, 100, 50)
        })
        
        self.optimizer.initialize_rl_environment(workload_data)
        
        self.assertIsNotNone(self.optimizer.rl_environment)
        self.assertIsNotNone(self.optimizer.rl_agent)


def run_comprehensive_tests():
    """Run comprehensive tests for Phase 2B."""
    logger.info("=" * 60)
    logger.info("PHASE 2B COMPREHENSIVE TESTING")
    logger.info("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestRLEnvironment))
    test_suite.addTest(unittest.makeSuite(TestPPOAgent))
    test_suite.addTest(unittest.makeSuite(TestComplianceChecker))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate test report
    test_report = {
        'test_summary': {
            'start_time': datetime.now().isoformat(),
            'tests_run': result.testsRun,
            'tests_failed': len(result.failures),
            'tests_errored': len(result.errors),
            'tests_passed': result.testsRun - len(result.failures) - len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
        },
        'test_details': {
            'failures': [str(failure) for failure in result.failures],
            'errors': [str(error) for error in result.errors]
        }
    }
    
    # Save test report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"models/phase_2b_test_results_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(test_report, f, indent=2, default=str)
    
    logger.info(f"Test report saved to {report_file}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests Run: {result.testsRun}")
    logger.info(f"Tests Passed: {test_report['test_summary']['tests_passed']}")
    logger.info(f"Tests Failed: {test_report['test_summary']['tests_failed']}")
    logger.info(f"Tests Errored: {test_report['test_summary']['tests_errored']}")
    logger.info(f"Success Rate: {test_report['test_summary']['success_rate']:.2%}")
    
    if test_report['test_summary']['success_rate'] >= 0.9:
        logger.info("✅ Phase 2B Testing PASSED!")
        return True
    else:
        logger.error("❌ Phase 2B Testing FAILED!")
        return False


def test_individual_components():
    """Test individual components separately."""
    logger.info("Testing individual components...")
    
    # Test RL Environment
    logger.info("1. Testing RL Environment...")
    try:
        env_test = TestRLEnvironment()
        env_test.setUp()
        env_test.test_environment_initialization()
        env_test.test_environment_reset()
        env_test.test_environment_step()
        logger.info("✅ RL Environment tests passed")
    except Exception as e:
        logger.error(f"❌ RL Environment tests failed: {e}")
        return False
    
    # Test PPO Agent
    logger.info("2. Testing PPO Agent...")
    try:
        agent_test = TestPPOAgent()
        agent_test.setUp()
        agent_test.test_agent_initialization()
        agent_test.test_action_selection()
        logger.info("✅ PPO Agent tests passed")
    except Exception as e:
        logger.error(f"❌ PPO Agent tests failed: {e}")
        return False
    
    # Test Compliance Checker
    logger.info("3. Testing Compliance Checker...")
    try:
        compliance_test = TestComplianceChecker()
        compliance_test.setUp()
        compliance_test.test_compliance_checker_initialization()
        logger.info("✅ Compliance Checker tests passed")
    except Exception as e:
        logger.error(f"❌ Compliance Checker tests failed: {e}")
        return False
    
    # Test Integration
    logger.info("4. Testing Integration...")
    try:
        integration_test = TestIntegration()
        integration_test.setUp()
        integration_test.test_optimizer_initialization()
        logger.info("✅ Integration tests passed")
    except Exception as e:
        logger.error(f"❌ Integration tests failed: {e}")
        return False
    
    logger.info("✅ All individual component tests passed!")
    return True


def main():
    """Main test execution function."""
    logger.info("Starting Phase 2B Testing")
    
    try:
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        
        # Test individual components first
        individual_tests_passed = test_individual_components()
        
        if not individual_tests_passed:
            logger.error("Individual component tests failed. Aborting comprehensive tests.")
            return False
        
        # Run comprehensive tests
        comprehensive_tests_passed = run_comprehensive_tests()
        
        if comprehensive_tests_passed:
            logger.info("🎉 All Phase 2B tests completed successfully!")
        else:
            logger.error("💥 Phase 2B tests failed!")
        
        return comprehensive_tests_passed
        
    except Exception as e:
        logger.error(f"Phase 2B testing failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
