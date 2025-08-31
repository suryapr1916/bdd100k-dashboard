# BDD100k Object Detection Assignment

A comprehensive computer vision pipeline for object detection on the BDD100k dataset, implementing data analysis, model training, and evaluation with interactive visualizations.

## Overview

This project implements a complete dataset-focused object detection pipeline. The system processes the BDD100k dataset for autonomous driving scenarios, focusing on 10 object detection classes including vehicles, pedestrians, traffic elements, and road infrastructure.

## Dataset

- **BDD100k Dataset**: Berkeley DeepDrive dataset with 100k driving images
- **Classes**: 12 object categories (bike, bus, car, drivable area, lane, motor, person, rider, traffic light, traffic sign, train, truck)
- **Format**: Original JSON annotations converted to YOLO format for training

## Project Structure

```
├── data/                           # Dataset and model storage
│   ├── assignment_data_bdd/        # BDD100k dataset files
│   ├── metadata/                   # Processed metadata and mappings
│   ├── models/                     # Pre-trained and trained models
│   └── validation_data_visualization/ # Evaluation result images
├── data_analysis/                  # Dataset analysis tools
│   ├── dashboard.py               # Interactive data exploration dashboard
│   ├── reader.py                  # Dataset reader utilities
│   └── config.py                  # Analysis configuration
├── model_training/                 # Model training pipeline
│   ├── yolo_train.py              # YOLOv9 training script
│   ├── val.py                     # Model validation and evaluation
│   ├── dataloader.py              # Custom data loading utilities
│   └── conversion.py              # Data format conversion
├── evaluation_visualization/       # Model evaluation dashboard
│   ├── dashboard.py               # Interactive evaluation visualization
│   └── reader.py                  # Evaluation data reader
├── config.py                      # Global configuration
├── setup.py                       # Data preprocessing setup
├── Dockerfile                     # Container configuration
└── requirements.txt               # Python dependencies
```
### Prerequisites

**System Requirements:**
- Docker Engine 20.10+
- For GPU acceleration: NVIDIA Container Toolkit
- Minimum 8GB RAM, 16GB recommended
- 8GB+ free disk space for dataset and models

**GPU Support Setup:**
You must install NVIDIA Container Toolkit to use your gpu for training and inference.

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Setup
```bash
# Clone repository and navigate to project directory
cd bdd100k-dashboard

# Build the Docker image (this may take 10-15 minutes)
docker build -t bdd-detection .

# Verify build success
docker images | grep bdd-detection
```

### Run Different Tasks

**1. Data Analysis Dashboard**
```bash
# CPU only
docker run -p 8504:8501 -e ARG_NUM=1 bdd-detection

# With GPU support (recommended for faster processing)
docker run --gpus all -p 8504:8501 -e ARG_NUM=1 bdd-detection
```
Access dashboard at: http://localhost:8504

**2. Model Training**
```bash
# CPU training (very slow, not recommended)
docker run -e ARG_NUM=2 bdd-detection

# GPU training (recommended)
docker run --gpus all -e ARG_NUM=2 bdd-detection

# With volume mounting to persist training results
docker run --gpus all -v $(pwd)/data:/app/data -e ARG_NUM=2 bdd-detection
```

**3. Model Evaluation**
```bash
# CPU evaluation
docker run -e ARG_NUM=3 bdd-detection

# GPU evaluation (faster)
docker run --gpus all -e ARG_NUM=3 bdd-detection

# With volume mounting to persist evaluation results
docker run --gpus all -v $(pwd)/data:/app/data -e ARG_NUM=3 bdd-detection
```

**4. Evaluation Visualization Dashboard**
```bash
# CPU only
docker run -p 8504:8501 -e ARG_NUM=4 bdd-detection

# With GPU support
docker run --gpus all -p 8504:8501 -e ARG_NUM=4 bdd-detection
```
Access dashboard at: http://localhost:8504

### Setup (without Docker)

Create an environment, activate and run this script.

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize data processing
python setup.py

# Run data analysis dashboard
streamlit run data_analysis/dashboard.py

# Train model
python model_training/yolo_train.py

# Evaluate model
python model_training/val.py
python model_training/conversion.py

# Run evaluation dashboard
streamlit run evaluation_visualization/dashboard.py
```

## Key Components

### Data Processing
- **setup.py**: Converts BDD100k JSON to YOLO format, generates metadata
- **reader.py**: Efficient dataset filtering and loading utilities
- **config.py**: Centralized path and configuration management

### Model Architecture
- **YOLOv9e**: Advanced single-stage object detector
- **Transfer Learning**: Fine-tuned on BDD100k with frozen backbone layers
- **Multi-class Detection**: Handles 12 object categories simultaneously

### Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across multiple IoU thresholds
- **Class-wise Performance**: Individual metrics per object category
- **Confusion Analysis**: Detailed breakdown of prediction accuracy

## Results

Model evaluation results are saved to:
- `data/validation_metrics.txt`: Quantitative performance metrics
- `data/metadata/evaluation_data.json`: Detailed prediction analysis
- `data/validation_data_visualization/`: Visual prediction samples
