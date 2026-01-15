# Kuchipudi Mudra Classification 🙏

Deep learning-based classification system for Indian classical dance hand gestures (mudras) from Kuchipudi tradition.

## 🎯 Project Overview

This project implements multiple approaches for mudra recognition:
- **VGG16 + SVM**: Transfer learning with traditional classifier
- **Fine-tuned ConvNeXt**: Modern CNN architecture
- **Real-time Detection**: MediaPipe + trained models for live inference

## 📊 Dataset

- **Classes**: 28 Kuchipudi mudra gestures
- **Split**: 70% train / 10% validation / 20% test
- **Source**: Kuchipudi Mudra Dataset

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mudra-classification.git
cd mudra-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training Pipeline
```bash
# 1. Split dataset
python scripts/split_data.py

# 2. Extract VGG16 features
python src/feature_extraction.py

# 3. Train SVM classifier
python src/train.py

# 4. Evaluate model
python src/evaluate.py
python src/confusion_matrix.py

# 5. Visualize features (optional)
python src/feature_visualization.py
```

### Real-time Inference
```bash
# Using webcam
python deployment/realtime_predict.py

# Using MediaPipe hand detection
python deployment/mediapipe_predict.py

# Single image prediction
python deployment/predict_mudra.py --image tests/test_image.jpg
```

## 📁 Project Structure
```
mudra-classification/
├── data/                  # Dataset and metadata
│   ├── Kuchipudi-Mudra-Dataset-master.zip
│   └── class_names.txt
├── scripts/              # Data preprocessing
│   ├── split_data.py
│   └── generate_class_names.py
├── src/                  # Core ML pipeline
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   ├── confusion_matrix.py
│   └── feature_visualization.py
├── models/               # Trained models
│   ├── svm_mudra_model.pkl
│   ├── vgg16_feature_extractor.h5
│   └── label_map.pkl
├── features/            # Extracted features (not in git)
├── deployment/          # Inference scripts
│   ├── predict_mudra.py
│   ├── realtime_predict.py
│   └── mediapipe_predict.py
├── tests/              # Test images
├── results/            # Evaluation outputs
└── config.py          # Configuration
```

## 🏗️ Model Architectures

### VGG16 + SVM
- Feature extractor: VGG16 (ImageNet pretrained, frozen at block5_pool)
- Classifier: SVM with RBF kernel (C=10)
- Input: 128×128 RGB images

### ConvNeXt
- Fine-tuned ConvNeXt model
- End-to-end training

## 📈 Results

- **Test Accuracy**: [Add after training]
- **Validation Accuracy**: [Add after training]

View confusion matrix: `results/confusion_matrix.png`

## 🛠️ Technologies

- **Deep Learning**: TensorFlow/Keras, PyTorch
- **ML**: scikit-learn
- **Computer Vision**: OpenCV, MediaPipe
- **Visualization**: Matplotlib, Seaborn
