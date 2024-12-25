# MINIM RLHF Training

A system for optimizing medical image generation using RLHF (Reinforcement Learning from Human Feedback) with MINIM and expert scoring.

## Overview

The system consists of three main components:
1. **Synthetic System**: Generates medical images using MINIM
2. **Selector**: Quality assessment model (0-1 score range)
3. **Policy**: RLHF optimization model (0-1 score range)

## Installation
```bash
git clone [repository-url]
cd MINIM_RLHF
pip install -r requirements.txt
```

## Data Format

### Expert Ratings
`data/medical_ratings.csv`:
```csv
image_path,rating,modality,expert_level,prompt
data/oct/image1.png,3,OCT,3,OCT: healthy retinal layers
data/ct/image2.dcm,2,CT,2,Chest CT: contrast enhanced scan showing aortic calcification
```

- rating: 1-3 scale (custom rating range, normalized to 0-1 during training)
- modality: OCT, CT, X-Ray ...
- expert_level: 1-3 (used for weighted training)
- prompt: prompt used to generate the image

## Project Structure

```
MINIM_RLHF/
├── checkpoints/
│   └── [timestamp]/
│       └── epoch_[n]/
│           ├── policy_checkpoint.pt
│           ├── selector_checkpoint.pt
│           └── synthetic_checkpoint.pt
├── config/
│   └── config.yaml       # Configuration file
├── src/
│   ├── rl/
│   │   └── policy.py     # Policy model implementation
│   ├── selector/
│   │   └── receive_selector.py  # Selector model implementation
│   ├── synthetic/
│   │   └── synthetic_system.py  # Image generation system
│   ├── raters/
│   │   └── rating_system.py     # Rating processing system
│   └── utils/
│       ├── device_utils.py      # Device management utilities
│       ├── medical_processor.py # Medical image processing
│       └── training_utils.py    # Training utilities
```

## Configuration

The system can be configured through `config.yaml`. Key configuration options include:

```yaml
# Training configuration
training:
  stage: 1  # 1 or 2
  max_epochs: 100
  batch_size: 32
  load_latest_checkpoint: false

# Data configuration
data:
  rating_range: [1, 2, 3]  # Custom rating range
  augmentation:
    enable: true
```

## Training Process

The training process consists of two stages:

### Stage 1
- Train both Policy and Selector models using all available data
- Policy learns to align prompts with images
- Selector learns to assess image quality

### Stage 2
- Selector filters out low-quality images
- Policy and Selector models are updated using filtered high-quality images
- Optional synthetic system updates using high-quality samples

## Usage

1. Prepare your medical image dataset and ratings file
2. Configure the system through `config.yaml`
3. Run the training:

```bash
python train.py
```

## Monitoring

The system uses Weights & Biases for experiment tracking. Key metrics include:
- Policy and Selector losses
- Pass rates
- Learning rates
- Synthetic system performance (if enabled)

## Model Checkpoints

Checkpoints are automatically saved based on the configured frequency:
```yaml
training:
  checkpoints:
    save_dir: "checkpoints"
    save_frequency: 10
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- transformers
- einops
- wandb
- Pillow
- numpy