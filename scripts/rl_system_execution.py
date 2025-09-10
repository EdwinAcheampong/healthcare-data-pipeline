#!/usr/bin/env python3
"""
RL System Execution Script - Healthcare Data Pipeline
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
    Main function to orchestrate the RL agent training and evaluation.
    """
    logger.info("Starting the full RL System Execution pipeline.")

    try:
        # --- 1. Load Data ---
        logger.info("Loading feature-engineered data for RL environment...")
        # This assumes the ML script ran and created these features.
        # A more robust system might pass data between steps.
        X_train = pd.read_parquet("data/processed/X_train.parquet")
        y_train = pd.read_parquet("data/processed/y_train.parquet")
        train_data = pd.concat([X_train, y_train], axis=1)
        logger.info(f"Loaded training data with shape: {train_data.shape}")

        # --- 2. Initialize HealthcareWorkloadOptimizer ---
        logger.info("Initializing HealthcareWorkloadOptimizer...")
        optimizer = HealthcareWorkloadOptimizer(IntegrationConfig())
        logger.info("HealthcareWorkloadOptimizer initialized successfully.")

        # --- 3. Initialize RL Environment and Agent ---
        logger.info("Initializing RL Environment and Agent...")
        optimizer.initialize_rl_environment(train_data)
        logger.info("RL Environment and Agent initialized successfully.")

        # --- 4. Train the RL Agent ---
        logger.info("Starting RL agent training...")
        optimizer.train_rl_agent()
        logger.info(f"RL agent training completed.")

        # --- 5. Save the Trained Agent ---
        logger.info("Saving the trained RL agent...")
        agent_path = Path("models/ppo_agent.pth")
        agent_path.parent.mkdir(exist_ok=True)
        torch.save(optimizer.rl_agent.actor_critic.state_dict(), agent_path)
        logger.info(f"Agent saved to {agent_path}")

        logger.info("RL System Execution pipeline finished successfully.")

    except FileNotFoundError:
        logger.error("Processed data not found. Please run the ETL and ML pipelines first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the RL pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
