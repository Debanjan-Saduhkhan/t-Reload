# T-Reload Research Paper Implementation

This repository contains a complete implementation of the **t-reload** method described in the research paper. The implementation follows best practices for reproducibility and includes comprehensive documentation, testing, and evaluation tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Paper Information](#paper-information)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [Citation](#citation)

## 🎯 Overview

The t-reload method is a novel approach to model training and optimization that enables efficient parameter reloading during the training process. This implementation provides:

- **Complete Model Architecture**: Full implementation of the described method
- **Training Pipeline**: End-to-end training with t-reload functionality
- **Evaluation Tools**: Comprehensive metrics and visualization
- **Data Processing**: Flexible data loading and preprocessing
- **Configuration Management**: Easy experiment configuration
- **Reproducibility**: Deterministic training and evaluation

## 📚 Paper Information

this implementation covers:

- **Title**: t-RELOAD: A REinforcement Learning-based REcommendation for Outcome-driven Application
- **Authors**: Debanjan Sadhukhan, Sachin Kumar, Swarit Sankule, and Tridib Mukherjee
- **Venue**: Proceedings of The Third International Conference on Artificial Intelligence and Machine Learning Systems (AIMLSystems 2023)
- **Year**: 2023
- **DOI**: https://doi.org/10.1145/nnnnnnn.nnnnnnn

### Paper Abstract

Games of skill provide an excellent source of entertainment to realize self-esteem, relaxation and social gratification. Engagement in online skill gaming platforms is however heavily dependent on the outcomes and experience (e.g., wins/losses). This work addresses the question "how can we leverage off-policy data for recommendation to solve cold-start problem while ensuring reward-driven optimality from platform-perspective in outcome-based applications?"

### Key Contributions

The paper introduces **t-RELOAD**: A REinforcement Learning-based REcommendation framework for Outcome-driven Application consisting of 3-layer-based architecture:
1. **Off-policy data-collection** (through already deployed solution)
2. **Offline training** (using relevancy)
3. **Online exploration with turbo-reward** (t-reward, using engagement)

### Implementation Components

This implementation includes:
- **Transformer-based Model**: Main recommendation model with t-reload functionality
- **DQN (Deep Q-Network)**: Reinforcement learning component for optimal recommendation policy
- **Recommendation Environment**: Simulated environment for training and evaluation
- **Experience Replay**: Buffer for storing and replaying training experiences

### Exact Algorithms from the Paper

This implementation follows the **exact algorithms** described in the t-RELOAD paper:

#### Algorithm 1: Offline Training (using relevancy)

**Input**: Batches of data (s, a_actual, s', r)  
**Output**: Trained policy network θ

1. Weights are initialized randomly for both policy(θ) and target network(θ')
2. Collect batches of data
3. for batch = 1, 2, ..., N do
4.   for each new training sample (s, a_actual, s', r) do
5.     Select an action either by exploration or exploitation
6.     Set a_rec = action from explore/exploitation
7.     Estimate reward r(s, a_rec, a_actual) following Eq. 2
8.     Estimate target_y = [r(s, a_rec, a_actual) + γQ*(s', argmax_a' Q*(s', a'; θ); θ') + ε]
9.   end for
10.  Train policy network using Eq. 4
11.  After some iteration, copy weights from policy network to target network
12. end for

**Key Features**:
- Uses relevancy-based rewards (Equation 2 from paper)
- Implements Double-DQN with Noise Clipping for better performance
- Follows hyperparameters from Table 2: learning rate 0.00005, gamma 0.7, epsilon_min 0.01

#### Algorithm 2: Online Training (using turbo-reward for engagement)

**Input**: Pre-trained offline model, engagement metrics  
**Output**: Updated policy network for engagement optimization

1. Set ε as learning rate
2. Weights are copied using Algo 1 for both policy and target network
3. for each batch = 1, 2, ..., N do
4.   Generate actions either by exploration/exploitation
5.   Estimate platform-centric reward r (e.g., engagement)
6.   Update the weights of policy network using r and ε
7.   After some iteration, copy weights from policy network to target network
8. end for

**Key Features**:
- Uses turbo-reward function based on platform objectives (engagement)
- Combines relevancy rewards with engagement rewards
- Gradually modifies weights using incremental reinforcement learning
- Optimizes for long-term engagement (number-of-games, inter-session-duration)

### Datasets and Evaluation

- **Dataset Size**: (1855131, 124) - total number of user interactions
- **Evaluation Metrics**: Engagement metrics, recommendation effectiveness
- **Baselines**: XGBoost-based recommendation system

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12.0 or higher
- CUDA (optional, for GPU acceleration)

### Install Dependencies

#### Option 1: Using Virtual Environment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd paper_impl

# Create virtual environment
python -m venv t-reload-env

# Activate virtual environment
# On macOS/Linux:
source t-reload-env/bin/activate
# On Windows:
# t-reload-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import paper_impl; print('Package imported successfully')"
```

#### Option 2: Using Conda

```bash
# Clone the repository
git clone <repository-url>
cd paper_impl

# Create conda environment
conda create -n t-reload python=3.9 -y
conda activate t-reload

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import paper_impl; print('Package imported successfully')"
```

## 🏃 Quick Start

### Prerequisites Check

Before starting, ensure you have:
- Python 3.8 or higher
- At least 4GB RAM
- GPU (optional, for faster training)

### 1. Parse the Research Paper

```bash
cd paper_impl

# Activate your environment first
source t-reload-env/bin/activate  # or conda activate t-reload

# Parse the PDF to extract key information
python parse_paper.py
```

This will extract key information from the PDF and save it to `analysis/parsed_paper.json`.

**Expected Output:**
```
Parsed paper information:
{
  "title": "t-RELOAD: A REinforcement Learning-based REcommendation...",
  "authors": [...],
  "year": "2023",
  ...
}
```

### 2. Create Sample Data

```bash
# Create sample training, validation, and test data
python -c "from data import create_sample_data; create_sample_data('data')"
```

**Expected Output:**
```
Sample data created successfully
Sample data created in data
```

### 3. Create Default Configuration

```bash
# Generate default configuration file
python -c "from config import get_default_config; config = get_default_config(); config.save('configs/default_config.json'); print('Default configuration created')"
```

**Expected Output:**
```
Default configuration created
Configuration saved to configs/default_config.json
```

### 4. Train the Model

```bash
# Start training with sample data
python train.py \
    --config configs/default_config.json \
    --experiment-name "my_first_experiment" \
    --output-dir "results/my_experiment" \
    --seed 42 \
    --create-sample-data
```

**Expected Output:**
```
[INFO] Creating sample data...
[INFO] Sample data created in data
[INFO] Created model with X parameters
[INFO] Initialized trainer on device: cpu
[INFO] Epoch 0, Batch 0/1, Loss: X.XXXX
...
[INFO] Training completed successfully!
[INFO] Best validation loss: X.XXXX
[INFO] Output saved to: results/my_experiment
```

### 5. Evaluate the Model

```bash
# Evaluate the trained model
python evaluate.py \
    --config configs/default_config.json \
    --checkpoint results/my_experiment/checkpoints/best_model.pt \
    --output-dir "results/my_experiment/evaluation"
```

**Expected Output:**
```
[INFO] Starting evaluation...
[INFO] Evaluation completed!
[INFO] Accuracy: X.XXXX
[INFO] Loss: X.XXXX

==================================================
CLASSIFICATION REPORT
==================================================
...
```

### 6. Monitor Training (Optional)

```bash
# Start TensorBoard to monitor training progress
tensorboard --logdir results/my_experiment/logs

# Open http://localhost:6006 in your browser
```

### 7. Train DQN Component (Reinforcement Learning)

```bash
# Train the DQN reinforcement learning component
python train_dqn.py \
    --num-episodes 1000 \
    --num-items 50 \
    --max-steps 100 \
    --eval-interval 100 \
    --save-interval 200 \
    --output-dir "results/dqn_experiment"
```

**Expected Output:**
```
[INFO] Creating DQN agent and environment...
[INFO] Initialized recommendation environment with 50 items and 1000 users
[INFO] Initialized DQN agent with 50 actions
[INFO] Starting DQN training for 1000 episodes...
[INFO] Episode 1/1000: Reward=45.234, Steps=12, Epsilon=1.000
[INFO] Episode 2/1000: Reward=32.156, Steps=8, Epsilon=0.995
...
[INFO] Final Evaluation: Mean Reward=67.891 ± 12.345
[INFO] DQN training completed successfully!
```

### 8. Run Exact Algorithms from Paper

```bash
# Execute Algorithm 1 (Offline Training) and Algorithm 2 (Online Training)
python train_treload_algorithms.py --offline-iterations 100 --epochs-per-iteration 5 --online-iterations 50

# Quick test run
python train_treload_algorithms.py --offline-iterations 5 --epochs-per-iteration 2 --online-iterations 3 --batch-size 20

# Full paper reproduction (500 iterations, 5 epochs each)
python train_treload_algorithms.py --offline-iterations 500 --epochs-per-iteration 5 --online-iterations 100
```

**Algorithm Parameters** (from paper Table 2):
- Learning rate: 0.00005
- Gamma (discount factor): 0.7
- Epsilon min: 0.01
- Epsilon decay: 0.05
- Target network update: every 10 iterations
- State dimension: 124
- Actions: 3 (0, 1, 2 for agent-1)

**Expected Output:**
```
[INFO] Creating t-RELOAD trainers following Algorithm 1 and 2...
[INFO] PHASE 1: ALGORITHM 1 - OFFLINE TRAINING
[INFO] Starting Algorithm 1: Offline Training
[INFO] Number of iterations: 500
[INFO] Epochs per iteration: 5
[INFO] Iteration 1/500
[INFO]   Epoch 1: Loss = 0.140380
[INFO]   Epoch 2: Loss = 0.126783
...
[INFO] PHASE 2: ALGORITHM 2 - ONLINE TRAINING
[INFO] Starting Algorithm 2: Online Training
[INFO] Number of iterations: 100
[INFO] Online Iteration 1/100
[INFO] Online Iteration 1 completed. Loss: 0.085711
...
[INFO] T-RELOAD TRAINING COMPLETED SUCCESSFULLY!
```

### 9. Use DQN for Recommendations

```python
from paper_impl import DQNAgent, TReloadEnvironment, create_dqn_agent
from paper_impl.config import get_default_config

# Create agent and environment
config = get_default_config()
agent, env = create_dqn_agent(config.model, num_items=50)

# Load trained model
agent.load_model("results/dqn_experiment/dqn_final_model.pt")

# Make recommendations
state = env.reset(user_id=123)
action = agent.select_action(state, training=False)
print(f"Recommended item: {action}")

# Get Q-values for all actions
q_values = agent.get_q_values(state)
print(f"Q-values: {q_values}")
```

## 📁 Project Structure

```
paper_impl/
├── __init__.py              # Package initialization
├── parse_paper.py           # PDF parsing utilities
├── utils.py                 # Common utilities and helpers
├── model.py                 # Main model implementation
├── train.py                 # Training script
├── train_dqn.py             # DQN training script
├── train_treload_algorithms.py  # Exact algorithms from paper
├── evaluate.py              # Evaluation script
├── config.py                # Configuration management
├── dqn.py                   # DQN implementation
├── algorithms.py            # Algorithm 1 & 2 implementations
├── data/                    # Data loading and preprocessing
│   ├── __init__.py
│   └── dataset.py
├── requirements.txt          # Dependencies
├── README.md                # This file
└── reproduce.sh             # End-to-end reproduction script
```

## 💻 Usage

### Basic Usage

```python
from paper_impl import TReloadModel, ExperimentConfig

# Load configuration
config = ExperimentConfig.load("configs/default_config.json")

# Create model
model = TReloadModel(config.model)

# Use the model
outputs = model(input_ids, attention_mask, labels)
```

### Advanced Usage

```python
from paper_impl import TReloadTrainer, TReloadDataLoader

# Create trainer
trainer = TReloadTrainer(model, config.training.to_dict())

# Create data loaders
train_loader, val_loader, test_loader = TReloadDataLoader.create_dataloaders(
    train_path="data/train.json",
    val_path="data/val.json",
    test_path="data/test.json",
    config=config.training.to_dict()
)

# Training loop
for epoch in range(config.training.num_epochs):
    # Train
    train_metrics = trainer.train_epoch(train_loader, epoch)
    
    # Evaluate
    val_metrics = trainer.evaluate(val_loader, epoch)
    
    # Reload parameters if needed (t-reload functionality)
    if should_reload:
        trainer.reload_and_continue("checkpoints/previous_model.pt")
```

## ⚙️ Configuration

The project uses a hierarchical configuration system:

### Model Configuration

```python
from paper_impl.config import ModelConfig

model_config = ModelConfig(
    model_type="t-reload",
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    dropout=0.1,
    activation="gelu"
)
```

### Training Configuration

```python
from paper_impl.config import TrainingConfig

training_config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    warmup_steps=1000,
    weight_decay=0.01
)
```

### Complete Configuration

```python
from paper_impl.config import ExperimentConfig

config = ExperimentConfig(
    experiment_name="t-reload_experiment",
    seed=42,
    device="cuda",
    model=model_config,
    training=training_config
)

# Save configuration
config.save("configs/experiment.json")
```

## 🎓 Training

### Training Script

```bash
python train.py \
    --config configs/default_config.json \
    --experiment-name "my_experiment" \
    --output-dir "results/my_experiment" \
    --seed 42 \
    --device cuda
```

### Training Features

- **Automatic Checkpointing**: Saves best model and regular checkpoints
- **TensorBoard Logging**: Real-time training visualization
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Based on validation loss
- **T-Reload Functionality**: Parameter reloading during training

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir results/my_experiment/logs

# View logs
tail -f results/my_experiment/logs/train.log
```

## 📊 Evaluation

### Evaluation Script

```bash
python evaluate.py \
    --config configs/default_config.json \
    --checkpoint results/checkpoints/best_model.pt \
    --output-dir "evaluation_results"
```

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Loss**: Cross-entropy loss
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual representation of predictions
- **Training Curves**: Loss and accuracy over time

### Results Output

Evaluation results are saved in multiple formats:
- `metrics.json`: Structured metrics data
- `classification_report.txt`: Human-readable report
- `confusion_matrix.png`: Visual confusion matrix
- `training_curves.png`: Training and validation curves

## 🔬 Reproducibility

### Deterministic Training

```python
from paper_impl.utils import set_seed

# Set all random seeds
set_seed(42)
```

### Environment Setup

```bash
# Create conda environment
conda create -n t-reload python=3.9
conda activate t-reload

# Install exact versions
pip install -r requirements.txt

# Verify CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Reproduction Script

```bash
# Run complete reproduction
bash reproduce.sh

# This script will:
# 1. Parse the research paper
# 2. Create sample data
# 3. Train the model
# 4. Evaluate results
# 5. Generate final report
```

### One-Command Reproduction

For the fastest way to reproduce the entire experiment:

```bash
# Make script executable (if needed)
chmod +x reproduce.sh

# Run complete reproduction
./reproduce.sh
```

**Expected Output:**
```
🚀 Starting T-Reload Research Paper Reproduction
================================================
[INFO] Creating project directories...
[INFO] Step 1: Parsing research paper...
[SUCCESS] Paper parsed successfully
[INFO] Step 2: Creating sample data...
[SUCCESS] Sample data created successfully
[INFO] Step 3: Creating default configuration...
[SUCCESS] Default configuration saved
[INFO] Step 4: Installing dependencies...
[SUCCESS] Dependencies installed
[INFO] Step 5: Verifying PyTorch installation...
[INFO] Step 6: Starting model training...
[SUCCESS] Training completed successfully
[INFO] Step 7: Evaluating trained model...
[SUCCESS] Evaluation completed successfully
[INFO] Step 8: Generating final report...
[SUCCESS] Final report generated
🎉 Reproduction completed successfully!
================================================
```

The script creates a comprehensive report at `results/reproduction/REPRODUCTION_REPORT.md`.

## 📋 Step-by-Step Execution Guide

### Complete Workflow

Follow these steps to run the complete implementation:

#### Step 1: Environment Setup
```bash
# Navigate to project directory
cd paper_impl

# Create and activate virtual environment
python -m venv t-reload-env
source t-reload-env/bin/activate  # macOS/Linux
# t-reload-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

#### Step 2: Paper Analysis
```bash
# Parse the research paper PDF
python parse_paper.py

# Check extracted information
cat analysis/parsed_paper.json | python -m json.tool
```

#### Step 3: Data Preparation
```bash
# Create sample datasets
python -c "from data import create_sample_data; create_sample_data('data')"

# Verify data creation
ls -la data/
# Should show: train.json, val.json, test.json
```

#### Step 4: Configuration
```bash
# Create default configuration
python -c "from config import get_default_config; config = get_default_config(); config.save('configs/default_config.json')"

# View configuration
cat configs/default_config.json | python -m json.tool
```

#### Step 5: Training
```bash
# Start training
python train.py \
    --config configs/default_config.json \
    --experiment-name "t-reload_experiment" \
    --output-dir "results/experiment_1" \
    --seed 42 \
    --create-sample-data

# Monitor progress
tail -f results/experiment_1/logs/train.log
```

#### Step 6: Evaluation
```bash
# Evaluate the trained model
python evaluate.py \
    --config configs/default_config.json \
    --checkpoint results/experiment_1/checkpoints/best_model.pt \
    --output-dir "results/experiment_1/evaluation"

# View results
ls -la results/experiment_1/evaluation/
```

#### Step 7: Visualization
```bash
# Start TensorBoard
tensorboard --logdir results/experiment_1/logs

# Open browser: http://localhost:6006
```

### Troubleshooting Common Issues

#### Issue: Import Errors
```bash
# Solution: Ensure environment is activated
source t-reload-env/bin/activate
python -c "import paper_impl; print('OK')"
```

#### Issue: CUDA/GPU Not Available
```bash
# Check PyTorch installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If CUDA not available, PyTorch will use CPU
# Training will be slower but functional
```

#### Issue: Memory Errors
```bash
# Reduce batch size in config
python -c "
from config import get_default_config
config = get_default_config()
config.training.batch_size = 16
config.save('configs/small_batch_config.json')
print('Small batch config created')
"
```

#### Issue: File Not Found Errors
```bash
# Ensure all directories exist
mkdir -p analysis data configs results logs models
```

### Performance Optimization

#### For Faster Training
```bash
# Use GPU if available
python train.py --device cuda

# Increase batch size (if memory allows)
python -c "
from config import get_default_config
config = get_default_config()
config.training.batch_size = 64
config.save('configs/fast_config.json')
"
```

#### For Memory Efficiency
```bash
# Reduce model size
python -c "
from config import get_default_config
config = get_default_config()
config.model.hidden_size = 384
config.model.num_layers = 6
config.save('configs/lightweight_config.json')
"
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass
6. **Submit** a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt[dev]

# Run tests
pytest

# Code formatting
black paper_impl/
flake8 paper_impl/
mypy paper_impl/
```

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{paper_title,
  title={Paper Title},
  author={Author Names},
  journal={Journal/Conference Name},
  year={Year},
  url={Repository URL}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper authors for the research
- PyTorch team for the deep learning framework
- Open source community for various utilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](issues) page
2. Review the documentation
3. Create a new issue with detailed information

---

**Note**: This implementation is based on the research paper and may require adjustments based on specific experimental setups or requirements.
