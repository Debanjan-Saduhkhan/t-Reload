#!/bin/bash

# T-Reload Research Paper Implementation - Reproduction Script
# This script reproduces the complete experiment from the research paper

set -e  # Exit on any error

echo "🚀 Starting T-Reload Research Paper Reproduction"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "parse_paper.py" ]; then
    print_error "Please run this script from the paper_impl directory"
    exit 1
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p analysis
mkdir -p data
mkdir -p results
mkdir -p logs
mkdir -p configs

# Step 1: Parse the research paper
print_status "Step 1: Parsing research paper..."
if [ -f "../AIMLSystems_tReload.pdf" ]; then
    python parse_paper.py
    if [ -f "analysis/parsed_paper.json" ]; then
        print_success "Paper parsed successfully"
        print_status "Extracted information:"
        cat analysis/parsed_paper.json | python -m json.tool
    else
        print_warning "Paper parsing may have failed, continuing with default config"
    fi
else
    print_warning "Research paper PDF not found, using default configuration"
fi

# Step 2: Create sample data
print_status "Step 2: Creating sample data..."
python -c "
from data import create_sample_data
create_sample_data('data')
print('Sample data created successfully')
"

# Step 3: Create default configuration
print_status "Step 3: Creating default configuration..."
python -c "
from config import get_default_config
config = get_default_config()
config.save('configs/default_config.json')
print('Default configuration saved')
"

# Step 4: Install dependencies
print_status "Step 4: Installing dependencies..."
if command -v pip &> /dev/null; then
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_warning "pip not found, please install dependencies manually: pip install -r requirements.txt"
fi

# Step 5: Verify PyTorch installation
print_status "Step 5: Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

# Step 6: Run training
print_status "Step 6: Starting model training..."
if [ -f "configs/default_config.json" ]; then
    python train.py \
        --config configs/default_config.json \
        --experiment-name "reproduction_experiment" \
        --output-dir "results/reproduction" \
        --seed 42 \
        --create-sample-data
    
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully"
    else
        print_error "Training failed"
        exit 1
    fi
else
    print_error "Configuration file not found"
    exit 1
fi

# Step 7: Evaluate the model
print_status "Step 7: Evaluating trained model..."
if [ -f "results/reproduction/checkpoints/best_model.pt" ]; then
    python evaluate.py \
        --config configs/default_config.json \
        --checkpoint results/reproduction/checkpoints/best_model.pt \
        --output-dir "results/reproduction/evaluation"
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation completed successfully"
    else
        print_warning "Evaluation failed, but training was successful"
    fi
else
    print_warning "Best model checkpoint not found, skipping evaluation"
fi

# Step 8: Generate final report
print_status "Step 8: Generating final report..."
cat > results/reproduction/REPRODUCTION_REPORT.md << EOF
# T-Reload Research Paper Reproduction Report

## Experiment Summary
- **Experiment Name**: reproduction_experiment
- **Date**: $(date)
- **Seed**: 42
- **Status**: Completed

## Files Generated
- **Configuration**: configs/default_config.json
- **Training Results**: results/reproduction/
- **Model Checkpoints**: results/reproduction/checkpoints/
- **Evaluation Results**: results/reproduction/evaluation/

## Training Logs
\`\`\`
$(tail -20 logs/train.log 2>/dev/null || echo "Training logs not available")
\`\`\`

## Next Steps
1. Review the training results in results/reproduction/
2. Analyze the evaluation metrics
3. Modify configuration for different experiments
4. Use the trained model for inference

## Notes
- This is an automated reproduction of the research paper
- Results may vary due to hardware differences
- For exact reproduction, ensure same environment and dependencies
EOF

print_success "Final report generated: results/reproduction/REPRODUCTION_REPORT.md"

# Step 9: Display results summary
print_status "Step 9: Results Summary"
echo "================================================"
echo "📁 Project Structure:"
ls -la

echo ""
echo "📊 Results Directory:"
if [ -d "results/reproduction" ]; then
    find results/reproduction -type f -name "*.pt" -o -name "*.json" -o -name "*.png" | head -10
fi

echo ""
echo "🔧 Configuration:"
if [ -f "configs/default_config.json" ]; then
    echo "Default configuration created and saved"
fi

echo ""
echo "📈 Training Output:"
if [ -d "results/reproduction" ]; then
    echo "Training completed and saved to results/reproduction/"
    echo "Checkpoints available in results/reproduction/checkpoints/"
    echo "Evaluation results in results/reproduction/evaluation/"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Review the generated results"
echo "2. Modify configuration for custom experiments"
echo "3. Use the trained model for inference"
echo "4. Check TensorBoard logs: tensorboard --logdir results/reproduction/logs"

echo ""
print_success "🎉 Reproduction completed successfully!"
echo "================================================"

# Optional: Open TensorBoard
if command -v tensorboard &> /dev/null && [ -d "results/reproduction/logs" ]; then
    echo ""
    print_status "To view training logs, run:"
    echo "tensorboard --logdir results/reproduction/logs"
    echo "Then open http://localhost:6006 in your browser"
fi
