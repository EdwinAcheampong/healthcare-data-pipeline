#!/usr/bin/env python3
"""
Simplified RL System Execution Script - Healthcare Data Pipeline
This version runs with fewer epochs for faster testing.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import torch

# Add src to path and import necessary modules
sys.path.append(str(Path(__file__).parent.parent))
from src.models.rl_environment import HealthcareWorkloadEnvironment
from src.models.ppo_agent import PPOHHealthcareAgent, PPOConfig
from src.models.rl_integration import HealthcareWorkloadOptimizer, IntegrationConfig
from src.utils.logging import setup_logging

# Setup logging
logger = setup_logging()

def main():
    """
    Main function to orchestrate the RL agent training and evaluation with simplified settings.
    """
    logger.info("Starting the simplified RL System Execution pipeline.")

    try:
        # --- 1. Load Data ---
        logger.info("Loading feature-engineered data for RL environment...")
        # This assumes the ML script ran and created these features.
        X_train = pd.read_parquet("data/processed/X_train.parquet")
        y_train = pd.read_parquet("data/processed/y_train.parquet")
        train_data = pd.concat([X_train, y_train], axis=1)
        logger.info(f"Loaded training data with shape: {train_data.shape}")

        # --- 2. Initialize HealthcareWorkloadOptimizer with simplified config ---
        logger.info("Initializing HealthcareWorkloadOptimizer with simplified settings...")
        simple_config = IntegrationConfig()
        simple_config.rl_training_episodes = 50  # Reduced from 1000 to 50
        simple_config.rl_evaluation_episodes = 10  # Reduced from 100 to 10
        simple_config.rl_update_frequency = 5  # Update more frequently
        
        optimizer = HealthcareWorkloadOptimizer(simple_config)
        logger.info("HealthcareWorkloadOptimizer initialized successfully with simplified settings.")

        # --- 3. Initialize RL Environment and Agent ---
        logger.info("Initializing RL Environment and Agent...")
        optimizer.initialize_rl_environment(train_data)
        logger.info("RL Environment and Agent initialized successfully.")

        # --- 4. Train the RL Agent (simplified) ---
        logger.info(f"Starting simplified RL agent training with {simple_config.rl_training_episodes} episodes...")
        training_results = optimizer.train_rl_agent()
        logger.info(f"RL agent training completed. Episodes: {len(training_results.get('episode_rewards', []))}")

        # --- 5. Save the Trained Agent ---
        logger.info("Saving the trained RL agent...")
        agent_path = Path("models/ppo_agent.pth")
        agent_path.parent.mkdir(exist_ok=True)
        
        if optimizer.rl_agent and optimizer.rl_agent.actor_critic:
            torch.save(optimizer.rl_agent.actor_critic.state_dict(), agent_path)
            logger.info(f"Agent saved to {agent_path}")
        else:
            logger.warning("RL agent not properly initialized, skipping save.")

        # --- 6. Quick Evaluation ---
        logger.info(f"Running quick evaluation with {simple_config.rl_evaluation_episodes} episodes...")
        evaluation_results = optimizer.evaluate_integrated_system(train_data, simple_config.rl_evaluation_episodes)
        logger.info(f"Evaluation completed. Overall score: {evaluation_results.get('overall_optimization_score', 0):.3f}")

        logger.info("Simplified RL System Execution pipeline finished successfully.")

    except FileNotFoundError as e:
        logger.error(f"Required data files not found: {e}")
        logger.error("Please run the ETL and ML pipelines first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the RL pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
