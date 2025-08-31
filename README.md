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

## Features

### 1. Data Analysis (10 points)
- **Interactive Dashboard**: Streamlit-based exploration of dataset statistics
- **Distribution Analysis**: Class distribution, weather conditions, scene types, time of day
- **Attribute Filtering**: Filter by occlusion, truncation, traffic light colors
- **Visualization**: RGB distribution analysis and object dimension scatter plots
- **Anomaly Detection**: Identify patterns and outliers in different object classes

### 2. Model Training (5+5 points)
- **YOLOv9e Architecture**: State-of-the-art object detection model
- **Transfer Learning**: Fine-tuning from pre-trained weights
- **Custom Data Loader**: Efficient BDD100k format handling
- **Training Pipeline**: Complete training workflow with validation

### 3. Evaluation & Visualization (10 points)
- **Quantitative Metrics**: mAP (Mean Average Precision), IoU (Intersection over Union)
- **Qualitative Analysis**: Visual comparison of predictions vs ground truth
- **Error Analysis**: Classification of correct predictions, missed detections, false positives
- **Interactive Dashboard**: Browse evaluation results with filtering capabilities

## Docker Usage

The project is fully containerized for easy deployment and testing.

### Prerequisites

**System Requirements:**
- Docker Engine 20.10+
- For GPU acceleration: NVIDIA Container Toolkit
- Minimum 8GB RAM, 16GB recommended
- 20GB+ free disk space for dataset and models

**GPU Support Setup:**
```bash
# Install NVIDIA Container Toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**For other distributions:**
```bash
# CentOS/RHEL/Fedora
curl -s -L https://nvidia.github.io/nvidia-docker/centos8/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build Container
```bash
# Clone repository and navigate to project directory
cd /path/to/nus-data

# Build the Docker image (this may take 10-15 minutes)
docker build -t bdd-detection .

# Verify build success
docker images | grep bdd-detection
```

### Run Different Tasks

**1. Data Analysis Dashboard**
```bash
# CPU only
docker run -p 8501:8501 -e ARG_NUM=1 bdd-detection

# With GPU support (recommended for faster processing)
docker run --gpus all -p 8501:8501 -e ARG_NUM=1 bdd-detection
```
Access dashboard at: http://localhost:8501

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
docker run -p 8501:8501 -e ARG_NUM=4 bdd-detection

# With GPU support
docker run --gpus all -p 8501:8501 -e ARG_NUM=4 bdd-detection
```
Access dashboard at: http://localhost:8501

### Volume Mounting Options
```bash
# Mount specific directories for data persistence
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -p 8501:8501 \
  -e ARG_NUM=1 \
  bdd-detection

# Mount entire project (development mode)
docker run --gpus all \
  -v $(pwd):/app \
  -p 8501:8501 \
  -e ARG_NUM=1 \
  bdd-detection
```

### Troubleshooting
```bash
# Check container logs
docker logs <container_id>

# Interactive shell access for debugging
docker run -it --gpus all bdd-detection /bin/bash

# Verify GPU access inside container
docker run --gpus all bdd-detection nvidia-smi

# Check available GPU memory
docker run --gpus all bdd-detection python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Remove container after use (cleanup)
docker run --rm --gpus all -e ARG_NUM=1 bdd-detection
```

## Local Development

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (optional, CPU training supported)

### Setup
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

## Technical Implementation

### Data Format Conversion
The pipeline converts BDD100k's native JSON format to YOLO-compatible text annotations, enabling efficient training with the Ultralytics framework.

### Custom Data Loader
Implements optimized data loading with proper batching, augmentation, and normalization for the BDD100k dataset structure.

### Evaluation Pipeline
Comprehensive evaluation including:
- Prediction-target matching based on IoU and class agreement
- Visualization of correct predictions, missed detections, and false positives
- Statistical analysis of model performance across different conditions

## Performance Analysis

The system provides detailed analysis of model performance including:
- Weather condition impact on detection accuracy
- Scene type performance variations
- Object size and aspect ratio correlation with detection success
- Time of day effects on model performance

## Dependencies

Core libraries:
- **ultralytics**: YOLOv9 implementation
- **torch/torchvision**: Deep learning framework
- **streamlit**: Interactive dashboard framework
- **plotly**: Advanced visualization
- **opencv-python**: Image processing
- **pandas/numpy**: Data manipulation

## Results

Model evaluation results are saved to:
- `data/validation_metrics.txt`: Quantitative performance metrics
- `data/metadata/evaluation_data.json`: Detailed prediction analysis
- `data/validation_data_visualization/`: Visual prediction samples
