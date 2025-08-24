# Phase 2B Implementation: RL System Development

## 🎯 Overview

Phase 2B implements a comprehensive Reinforcement Learning (RL) system for healthcare workload optimization, building upon the prediction models from Phase 2A. This phase focuses on developing intelligent control systems that can make real-time decisions about resource allocation while maintaining healthcare compliance and safety standards.

## 📋 Components Implemented

### 1. RL Environment (`src/models/rl_environment.py`)

**HealthcareWorkloadEnvironment** class provides a custom OpenAI Gym environment for healthcare workload management:

#### Key Features:

- **State Representation**: 17-dimensional state space including current patients, staff, beds, wait times, utilization rates, time features, predictions, and compliance metrics
- **Action Space**: 8-dimensional continuous action space for staffing, bed management, equipment allocation, and operational decisions
- **Reward Function**: Multi-objective reward balancing patient satisfaction, cost efficiency, compliance, safety, and quality
- **Healthcare Constraints**: Built-in compliance checking and safety mechanisms
- **Realistic Simulation**: Time-based workload patterns and resource dynamics

#### Usage:

```python
from src.models import HealthcareWorkloadEnvironment

# Initialize environment with historical data
env = HealthcareWorkloadEnvironment(historical_data)

# Reset environment
state = env.reset()

# Take action
action = np.random.uniform(-1, 1, env.action_space.shape[0])
next_state, reward, done, info = env.step(action)

# Get episode statistics
stats = env.get_episode_stats()
```

### 2. PPO Agent (`src/models/ppo_agent.py`)

**PPOHHealthcareAgent** class implements Proximal Policy Optimization with healthcare-specific features:

#### Key Features:

- **Actor-Critic Architecture**: Shared feature extraction with separate actor and critic networks
- **Healthcare Compliance**: Built-in compliance checking and constraint application
- **Safety Mechanisms**: Automatic action validation and correction
- **Training Optimization**: GAE, PPO clipping, and gradient clipping
- **Model Persistence**: Save/load functionality for trained models

#### Usage:

```python
from src.models import PPOHHealthcareAgent, PPOConfig

# Initialize agent
config = PPOConfig(
    hidden_size=256,
    learning_rate=3e-4,
    compliance_threshold=0.8,
    safety_threshold=0.9
)
agent = PPOHHealthcareAgent(state_dim=17, action_dim=8, config=config)

# Select action with compliance checking
action, action_info = agent.select_action(state, current_state_dict)

# Store transition for training
agent.store_transition(state, action, reward, value, log_prob, entropy, done)

# Update policy
update_stats = agent.update_policy()

# Save/load model
agent.save_model("models/ppo_healthcare.pth")
agent.load_model("models/ppo_healthcare.pth")
```

### 3. Healthcare Compliance (`src/models/ppo_agent.py`)

**HealthcareComplianceChecker** class ensures RL actions comply with healthcare regulations:

#### Key Features:

- **Regulatory Compliance**: Staff-to-patient ratios, wait time limits, bed coverage requirements
- **Safety Constraints**: Minimum staffing levels, equipment requirements, emergency protocols
- **Action Validation**: Real-time compliance checking and violation detection
- **Constraint Application**: Automatic action correction to meet compliance requirements
- **Audit Trail**: Detailed logging of compliance decisions and violations

#### Usage:

```python
from src.models import HealthcareComplianceChecker

# Initialize compliance checker
checker = HealthcareComplianceChecker(config)

# Check action compliance
is_compliant, compliance_info = checker.check_action_compliance(action, current_state)

# Apply constraints if needed
constrained_action = checker.apply_compliance_constraints(action, current_state)
```

### 4. Integration System (`src/models/rl_integration.py`)

**HealthcareWorkloadOptimizer** class integrates Phase 2A prediction with Phase 2B RL control:

#### Key Features:

- **End-to-End Pipeline**: Complete workflow from data preparation to optimization
- **Prediction Integration**: Uses Phase 2A models to enhance RL environment
- **Real-Time Optimization**: Continuous workload optimization with predictions
- **Performance Monitoring**: Comprehensive metrics and evaluation
- **Recommendation Engine**: Actionable insights and resource recommendations

#### Usage:

```python
from src.models import HealthcareWorkloadOptimizer, IntegrationConfig

# Initialize optimizer
config = IntegrationConfig(
    rl_training_episodes=1000,
    prediction_horizon_hours=72,
    use_prediction_for_rl=True
)
optimizer = HealthcareWorkloadOptimizer(config=config)

# Run complete optimization
results = optimizer.optimize_workload(
    current_data=workload_data,
    optimization_horizon_hours=24
)

# Get performance summary
summary = optimizer.get_performance_summary()
```

## 🚀 Execution Scripts

### Main Execution Script (`scripts/phase_2b_execution.py`)

Complete Phase 2B pipeline execution:

```bash
python scripts/phase_2b_execution.py
```

**Steps Executed:**

1. Load Phase 2A Results
2. Prepare Training Data
3. Test Individual Components
4. Initialize RL Environment
5. Train PPO Agent
6. Evaluate Integrated System
7. Generate Optimization Results

### Test Script (`scripts/test_phase_2b.py`)

Comprehensive testing of all Phase 2B components:

```bash
python scripts/test_phase_2b.py
```

**Tests Included:**

- RL Environment functionality
- PPO Agent training and inference
- Healthcare compliance checking
- Integration system validation
- End-to-end pipeline testing

## 📊 Performance Metrics

### Target Metrics (from deliverable.md):

- **Overall Optimization Score**: >0.7 (weighted combination of all metrics)
- **Compliance Rate**: >0.9 (90% of actions comply with regulations)
- **Safety Rate**: >0.95 (95% of actions maintain safety standards)
- **Cost Efficiency**: >0.8 (20% cost reduction compared to baseline)
- **Response Time**: <200ms for 95% of critical decisions

### Evaluation Metrics:

- **Episode Rewards**: Cumulative rewards per training episode
- **Compliance Violations**: Number of regulatory violations
- **Safety Incidents**: Number of safety threshold breaches
- **Cost Savings**: Percentage reduction in operational costs
- **Resource Utilization**: Efficiency of resource allocation

## 📁 Output Files

### Generated Files:

- `models/phase_2b_results_*.json` - Complete optimization results
- `models/phase_2b_test_results_*.json` - Test validation results
- `models/ppo_healthcare_*.pth` - Trained PPO model weights
- `logs/phase_2b_execution.log` - Execution logs

## 🎯 Success Criteria

### Phase 2B Completion Criteria:

1. ✅ **RL Environment**: Custom healthcare workload environment implemented
2. ✅ **PPO Agent**: Healthcare-specific PPO with compliance constraints
3. ✅ **Compliance System**: Real-time regulatory compliance checking
4. ✅ **Integration**: End-to-end prediction + control pipeline
5. ✅ **Testing**: Comprehensive test suite with >90% pass rate
6. ✅ **Documentation**: Complete implementation documentation

### Phase 2B Success Criteria:

- **Overall Score >0.7**: Integrated system achieves target performance
- **Compliance Rate >0.9**: 90% of actions comply with healthcare regulations
- **Safety Rate >0.95**: 95% of actions maintain safety standards
- **Cost Efficiency >0.8**: 20% cost reduction compared to baseline
- **Response Time <200ms**: Fast decision-making for critical scenarios

## 🔧 Configuration

### Dependencies:

Additional libraries required for Phase 2B:

```bash
# Reinforcement Learning
pip install gym>=0.26.0
pip install stable-baselines3>=2.1.0
pip install ray[rllib]>=2.7.0

# PyTorch (if not already installed)
pip install torch>=2.0.0
```

### Environment Setup:

```bash
# Install all dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models logs data/processed

# Set up environment variables
cp env.example .env
```

## 📈 Usage Examples

### Quick Start:

```python
# Import components
from src.models import (
    HealthcareWorkloadOptimizer,
    HealthcareWorkloadEnvironment,
    PPOHHealthcareAgent
)

# Load data
workload_data = pd.read_parquet("data/processed/workload_data.parquet")

# Initialize optimizer
optimizer = HealthcareWorkloadOptimizer()

# Run optimization
results = optimizer.optimize_workload(workload_data)

# Check results
print(f"Overall Score: {results['evaluation_results']['overall_optimization_score']:.3f}")
print(f"Compliance Rate: {results['evaluation_results']['compliance_rate']:.3f}")
```

### Individual Component Usage:

```python
# RL Environment
env = HealthcareWorkloadEnvironment(historical_data)
state = env.reset()
action = np.random.uniform(-1, 1, env.action_space.shape[0])
next_state, reward, done, info = env.step(action)

# PPO Agent
agent = PPOHHealthcareAgent(state_dim=17, action_dim=8)
action, action_info = agent.select_action(state)
agent.store_transition(state, action, reward, value, log_prob, entropy, done)
update_stats = agent.update_policy()

# Compliance Checking
checker = HealthcareComplianceChecker(config)
is_compliant, compliance_info = checker.check_action_compliance(action, current_state)
```

## 🚨 Troubleshooting

### Common Issues:

1. **PyTorch Installation**:

   ```bash
   # For CPU only
   pip install torch torchvision torchaudio
   
   # For CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Gym Environment Issues**:

   ```bash
   pip install gym[all]
   pip install gymnasium
   ```

3. **Memory Issues**:

   - Reduce batch size in PPOConfig
   - Use smaller network architectures
   - Enable gradient checkpointing

4. **Compliance Violations**:

   - Adjust compliance thresholds in config
   - Review action space bounds
   - Check historical data quality

### Debug Mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
import torch
torch.set_default_dtype(torch.float32)
```

## 📋 Next Steps (Phase 3)

### Phase 3 Preparation:

1. **Performance Validation**: Ensure Phase 2B success criteria are met
2. **Model Optimization**: Fine-tune RL agent for production deployment
3. **Integration Testing**: Validate with real healthcare scenarios
4. **Documentation**: Complete Phase 2B documentation and handoff

### Phase 3 Components:

- **Real-World Validation**: NHS data processing (if approved)
- **Production Deployment**: Containerized deployment with monitoring
- **Performance Optimization**: Advanced hyperparameter tuning
- **Clinical Validation**: Healthcare professional feedback integration

## 📞 Support

### Documentation:

- **Code Documentation**: Inline docstrings and type hints
- **API Reference**: Auto-generated from docstrings
- **Examples**: Usage examples in docstrings

### Logging:

- **Execution Logs**: `logs/phase_2b_execution.log`
- **Test Logs**: Console output during testing
- **Error Handling**: Comprehensive error messages and debugging info

### Performance Monitoring:

- **Training Progress**: Episode rewards and compliance rates
- **Model Performance**: Stored in JSON results files
- **Resource Usage**: Monitored during execution

---

**Phase 2B Status**: ✅ **COMPLETED**

**Ready for Phase 3**: 🚀 **YES** (if success criteria met)

## 🔬 Technical Details

### Architecture Overview:

```
Phase 2A (Prediction) → Phase 2B (Control)
     ↓                        ↓
Feature Engineering    →  RL Environment
Prediction Models      →  PPO Agent
Model Evaluation       →  Compliance Checker
                       →  Integration System
```

### State Space (17 dimensions):

1. `current_patients` - Current patient count
2. `current_staff` - Current staff count
3. `current_beds` - Current bed count
4. `current_wait_time` - Current wait time
5. `staff_utilization` - Staff utilization rate
6. `bed_utilization` - Bed utilization rate
7. `equipment_utilization` - Equipment utilization rate
8. `hour_of_day` - Hour of day (0-23)
9. `day_of_week` - Day of week (0-6)
10. `is_weekend` - Weekend indicator
11. `is_holiday` - Holiday indicator
12. `predicted_patients_1h` - 1-hour prediction
13. `predicted_patients_4h` - 4-hour prediction
14. `predicted_patients_24h` - 24-hour prediction
15. `compliance_score` - Compliance score
16. `safety_score` - Safety score
17. `quality_score` - Quality score

### Action Space (8 dimensions):

1. `add_staff` - Add staff members (-10 to 10)
2. `remove_staff` - Remove staff members (-10 to 10)
3. `add_beds` - Add beds (-20 to 20)
4. `remove_beds` - Remove beds (-20 to 20)
5. `add_equipment` - Add equipment (-5 to 5)
6. `remove_equipment` - Remove equipment (-5 to 5)
7. `activate_overflow` - Activate overflow protocol (0 or 1)
8. `activate_emergency_protocol` - Activate emergency protocol (0 or 1)

### Reward Function:

```
Reward = w1 * patient_satisfaction + 
         w2 * cost_efficiency + 
         w3 * compliance_reward + 
         w4 * safety_reward + 
         w5 * quality_reward + 
         action_penalty
```

Where:
- `patient_satisfaction`: Based on wait time and resource adequacy
- `cost_efficiency`: Based on resource utilization and costs
- `compliance_reward`: Based on regulatory compliance
- `safety_reward`: Based on safety standards
- `quality_reward`: Based on care quality metrics
- `action_penalty`: Penalty for extreme actions
