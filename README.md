# Deep Learning Projects - Image Processing

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive collection of deep learning models for **anomaly detection in large intestine medical images** using state-of-the-art convolutional neural networks. This project implements and compares three different CNN architectures trained on multiple medical imaging datasets to achieve accurate classification of intestinal abnormalities.

## 🎯 Project Description

This project focuses on developing automated systems for detecting anomalies in large intestine medical images using deep learning techniques. The models are designed to assist medical professionals in identifying various types of intestinal abnormalities through computer vision and machine learning.

**Key Features:**
- **Multi-Architecture Comparison**: Implementation of ResNet50, ResNet101, and VGG16 models
- **Multiple Dataset Support**: Training on Kvasir and Nerthus medical imaging datasets
- **Transfer Learning**: Leveraging pre-trained ImageNet weights for improved performance
- **Data Augmentation**: Advanced preprocessing techniques for robust model training
- **Comprehensive Evaluation**: Detailed performance metrics and classification reports

## 🚀 Features & Benefits

### Model Architectures
- **ResNet50**: Efficient residual network with 50 layers
- **ResNet101**: Deeper residual network with 101 layers for enhanced feature extraction
- **VGG16**: Classic convolutional architecture with 16 layers

### Dataset Support
- **Kvasir Dataset**: Multi-class intestinal abnormality classification (8-14 classes)
- **Nerthus Dataset**: Specialized medical imaging dataset (4 classes)

### Technical Advantages
- **Transfer Learning**: Pre-trained ImageNet weights for faster convergence
- **Data Augmentation**: Rotation, horizontal flipping, and nearest neighbor filling
- **Dropout Regularization**: Prevents overfitting with 0.5 dropout rate
- **Adam Optimizer**: Efficient gradient-based optimization
- **Early Stopping**: Prevents overtraining with automatic stopping

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- Keras 2.x
- Google Colab (recommended for GPU acceleration)

### Dependencies
```bash
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install opencv-python
pip install pandas
```

### Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Deep-Learning-Projects-Image-Processing.git
   cd Deep-Learning-Projects-Image-Processing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Google Colab users:**
   - Upload the notebooks to Google Colab
   - Mount Google Drive for dataset access
   - Enable GPU runtime for faster training

## 💻 Usage

### Training Models

#### 1. ResNet50 Training
```python
# Load and preprocess data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    fill_mode='nearest',
    horizontal_flip=True
)

# Build ResNet50 model
resnet_50_model = Sequential()
pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg',
    weights='imagenet'
)

# Add custom classification layers
resnet_50_model.add(pretrained_model)
resnet_50_model.add(Flatten())
resnet_50_model.add(Dense(512, activation='relu'))
resnet_50_model.add(Dropout(0.5))
resnet_50_model.add(Dense(num_classes, activation='softmax'))

# Compile and train
resnet_50_model.compile(
    Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
```

#### 2. ResNet101 Training
```python
# Similar structure with ResNet101 backbone
pretrained_model = tf.keras.applications.ResNet101(
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg',
    weights='imagenet'
)
```

#### 3. VGG16 Training
```python
# VGG16 implementation
pretrained_model = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    pooling='maxpool'
)
```

### Model Evaluation
```python
# Generate predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Classification report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

### Inference
```python
# Load trained model
model = tf.keras.models.load_model('path/to/trained/model.h5')

# Preprocess new image
img = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
```

## 📁 Project Structure

```
Deep-Learning-Projects-Image-Processing/
├── README.md
└── Anomaly-Detection-In-Large-Intestine/
    ├── README.md
    ├── ResNet50_Kvasir_HASHIR.ipynb
    ├── ResNet50_KvasirDataset_V2_HASHIR.ipynb
    ├── ResNet50_KvasirDataset_V3_HASHIR.ipynb
    ├── ResNet50_Nurthus_HASHIR.ipynb
    ├── ResNet101_Kvasir_Dataset_HASHIR.ipynb
    ├── ResNet101_Kvasir_Dataset_V2_HASHIR.ipynb
    ├── ResNet101_Kvasir_Dataset_v3_HASHIR.ipynb
    ├── ResNet101_Nurthus_HASHIR.ipynb
    ├── VGG_16_Kvasir_Dataset_HASHIR.ipynb
    ├── VGG_16_Kvasir_Dataset_V2_HASHIR.ipynb
    ├── VGG_16_Kvasir_Dataset_V3_HASHIR.ipynb
    └── VGG_16_Nurthus_HASHIR.ipynb
```

### Notebook Organization
- **ResNet50**: Implementation with different dataset versions
- **ResNet101**: Deeper network architecture experiments
- **VGG16**: Classic CNN architecture implementation
- **Dataset Variants**: V1, V2, V3 versions of Kvasir dataset
- **Nerthus**: Specialized medical imaging dataset

## 📊 Dataset Details

### Kvasir Dataset
- **Classes**: 8-14 different intestinal abnormality types
- **Image Size**: 224x224 pixels
- **Format**: RGB images
- **Split**: Train/Validation/Test sets
- **Augmentation**: Rotation (±20°), horizontal flip, nearest neighbor filling

### Nerthus Dataset
- **Classes**: 4 different abnormality categories
- **Image Size**: 224x224 pixels
- **Format**: RGB images
- **Medical Focus**: Specialized intestinal imaging

### Dataset Access
- **Google Drive Link**: [Dataset Download](https://drive.google.com/drive/folders/1yUQorGyRr9gJ13I_x9JDdlOnxa0cSjb_?usp=drive_link)
- **Structure**: Organized in train/val/test directories
- **Preprocessing**: Automatic data augmentation and normalization

## 📈 Results & Evaluation Metrics

### Performance Summary

| Model | Dataset | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| ResNet50 | Kvasir V2 | **94%** | 0.95 | 0.94 | 0.94 |
| ResNet101 | Kvasir V3 | **91%** | 0.86 | 0.91 | 0.88 |
| VGG16 | Kvasir V2 | **78%** | 0.93 | 0.78 | 0.79 |
| ResNet50 | Kvasir V3 | **84%** | 0.80 | 0.84 | 0.81 |
| ResNet101 | Kvasir | **81%** | 0.82 | 0.81 | 0.81 |

### Key Findings
- **ResNet50** achieved the highest accuracy (94%) on Kvasir V2 dataset
- **ResNet101** showed consistent performance across different dataset versions
- **VGG16** demonstrated good precision but lower recall rates
- All models showed strong performance on multi-class classification tasks

### Classification Reports
Detailed per-class performance metrics are available in each notebook, including:
- Precision, Recall, and F1-score for each class
- Support (number of samples per class)
- Macro and weighted averages
- Confusion matrices with visualizations

## 🔧 Technical Specifications

### Model Configurations
- **Input Size**: 224x224x3 (RGB images)
- **Batch Size**: 32
- **Epochs**: 5-10 (with early stopping)
- **Learning Rate**: 0.0001 (Adam optimizer)
- **Loss Function**: Categorical crossentropy
- **Regularization**: Dropout (0.5), Batch normalization

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 8GB+ VRAM (NVIDIA Tesla/RTX series)
- **Cloud**: Google Colab Pro (recommended for training)

## 🚀 Future Improvements

### Planned Enhancements
- [ ] **Data Augmentation**: Implement advanced augmentation techniques (mixup, cutmix)
- [ ] **Model Ensemble**: Combine multiple models for improved accuracy
- [ ] **Attention Mechanisms**: Add attention layers for better feature focus
- [ ] **Real-time Inference**: Develop web application for live predictions
- [ ] **Model Optimization**: Quantization and pruning for deployment
- [ ] **Cross-validation**: Implement k-fold validation for robust evaluation
- [ ] **Hyperparameter Tuning**: Automated hyperparameter optimization
- [ ] **Additional Datasets**: Support for more medical imaging datasets

### Research Directions
- [ ] **Explainable AI**: Implement Grad-CAM for model interpretability
- [ ] **Few-shot Learning**: Adapt models for limited data scenarios
- [ ] **Multi-modal Fusion**: Combine image and clinical data
- [ ] **Federated Learning**: Distributed training across medical institutions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hashir Ahmad**
- **Email**: [HashirAhmad330@gmail.com]
- **LinkedIn**: [https://www.linkedin.com/in/hashirahmed07/]
- **GitHub**: [https://github.com/CodeByHashir]

## 🙏 Acknowledgments

- **Kvasir Dataset**: Medical imaging dataset for gastrointestinal disease classification
- **Nerthus Dataset**: Specialized medical imaging dataset
- **TensorFlow/Keras**: Deep learning framework
- **Google Colab**: Cloud computing platform for model training
- **Medical Imaging Community**: For providing valuable datasets and research insights

## 📞 Contact

For questions, suggestions, or collaborations, please feel free to reach out:

- **Issues**: [GitHub Issues](https://github.com/yourusername/Deep-Learning-Projects-Image-Processing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Deep-Learning-Projects-Image-Processing/discussions)
- **Email**: [HashirAhmad330@gmail.com]

---

**Note**: This project is for research and educational purposes. For medical applications, please consult with healthcare professionals and ensure proper validation and regulatory compliance.