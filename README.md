# 🌾 Paddy Disease Classification and Detection using Deep Learning (ResNet50) 

This project presents a complete end-to-end deep learning pipeline for **classifying paddy leaf diseases** and **detecting affected regions** through Grad-CAM.  
It combines dataset preprocessing, exploratory data analysis (EDA), model building, transfer learning, fine-tuning, performance evaluation, visualization, and real-time prediction.



##  Dataset

The dataset used for this project is publicly available on Kaggle:

🔗 **https://www.kaggle.com/datasets/imbikramsaha/paddy-doctor**

It contains multiple paddy leaf disease classes organized into separate folders.  
Each folder represents one disease category.



##  Project Overview

This project aims to support agricultural disease diagnosis by:

- Automatically identifying **paddy leaf diseases** from images.  
- Detecting and highlighting the infected region using **Grad-CAM** heatmaps.  
- Providing confidence-based predictions to help with decision support.  
- Analyzing dataset quality with detailed EDA to guide preprocessing and augmentation.  
- Training a high-performance classifier using **ResNet50 transfer learning** and careful fine-tuning.

The codebase is modular and reproducible so you can extend it for further research or production deployment.



##  Features

- Multi-class **paddy disease classification** with confidence scores.  
- Disease **region detection** using Grad-CAM heatmaps for interpretability.  
- Full preprocessing and data augmentation pipeline to improve generalization.  
- Extensive **Exploratory Data Analysis (EDA)** with plots saved to disk.  
- Transfer learning with **ResNet50** (ImageNet weights) and staged fine-tuning.  
- Evaluation pipeline: confusion matrix, classification report, ROC curves.  
- Prediction utilities: single-image and batch inference, image display, Grad-CAM overlay.  
- Outputs saved for reproducible reports (PNGs, CSVs, model checkpoint).


##  Technologies Used

- **Python 3**  
- **TensorFlow / Keras** (ResNet50, ImageDataGenerator)  
- **NumPy, Pandas**  
- **Matplotlib, Seaborn**  
- **OpenCV, Pillow (PIL)**  
- **Scikit-learn**


##  Project Workflow 

### 1. Data Loading
- Read images from class-wise folders into a DataFrame with columns: `filepath`, `label`.  
- Validate that all file paths are accessible and count images per class.

### 2. Train–Validation Split
- Use an **80/20 stratified split** to preserve class proportions between training and validation sets.

### 3. Image Preprocessing
- Resize images to **224×224**.  
- Normalize pixel values to `[0, 1]`.  
- Convert to tensors suitable for model input.

### 4. Data Augmentation
- Apply random transforms during training: rotation, flips, zoom, shifts, shear, brightness.  
- Preview augmentations to ensure label fidelity.

### 5. Exploratory Data Analysis (EDA)
- Visualize and save:
  - Class distribution (bar chart, pie, donut).  
  - One sample per class grid.  
  - Image size and aspect ratio distributions.  
  - Brightness and RGB mean plots.  
  - Train vs validation splits and augmentation previews.  
- Use EDA findings to set augmentation and preprocessing choices.

### 6. Model Architecture (ResNet50)
- Use `tf.keras.applications.ResNet50(include_top=False, weights='imagenet')` as backbone.  
- Add custom top:
  - GlobalAveragePooling2D → BatchNorm → Dense(512, ReLU) → Dropout → Dense(NUM_CLASSES, softmax).

### 7. Training Strategy
- **Phase 1 — Train top layers only**:
  - Freeze backbone, train new head layers (Adam, lr=1e-3).  
- **Phase 2 — Fine-tuning**:
  - Unfreeze upper backbone layers, train with lower lr (1e-5).  
- Callbacks used: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.  
- Save `class_indices.json` for consistent label mapping at inference.


##  Visualizations Used

### Dataset Visualizations
- Pie chart & donut chart for class proportions.  
- Bar chart for absolute class counts (colored).  
- One-sample-per-class image grid.  
- Image width/height histograms, aspect ratio, image area boxplots.  
- Brightness (HSV V-channel) histogram.  
- Mean RGB per-class bar chart.  
- Augmentation preview and train/val count comparisons.

### Training Visualizations
- Training & validation accuracy curve.  
- Training & validation loss curve.  
- Combined 3×2 frame containing:
- Accuracy line, Loss line, Accuracy scatter, Loss scatter, Accuracy boxplot, Loss boxplot.

### Evaluation Visualizations
- Confusion matrix heatmap.  
- Per-class accuracy bar plot.  
- ROC curves (one-vs-rest).  
- Grad-CAM heatmaps and misclassified examples gallery.  
- Batch predictions CSV for further analysis.


##  Grad-CAM Heatmaps

Grad-CAM is used to make the model’s decisions interpretable. The project’s Grad-CAM utilities:

- **Highlight infected leaf regions** by overlaying class-discriminative activation maps on the original image.  
- **Visualize the model’s decision areas** so you can inspect which parts of the leaf contributed most to a particular prediction.  
- **Explain model predictions** by providing a visual explanation that can be reviewed by domain experts.  
- **Analyze correct and misclassified images** — comparing Grad-CAM overlays for both cases helps find labeling issues or model failure modes.  
- **Automatic storage**: generated heatmaps and overlays are saved to a dedicated folder (e.g., `gradcam_examples/`) for reproducibility.


##  Results

- **High validation accuracy** observed after fine-tuning the model on the dataset.  
- **Strong generalization** to validation images following staged training and augmentation.  
- **Clear localized detection** via Grad-CAM: heatmaps often focus on diseased spots.  
- **Reliable multi-class predictions** with top-k candidate outputs and confidence scores.  
- Project is **extensible** for mobile or IoT deployment after further model compression or export to TFLite.


##  Future Enhancements

Planned improvements to take the project from prototype to production:

- **Implement segmentation (U-Net)** for precise disease area measurement and percent lesion estimation.  
- **Add mobile deployment (TensorFlow Lite)** to allow field diagnostics on Android/iOS devices.  
- **Build a web app** (Flask or Streamlit) for easy image upload, prediction and visualization.  
- **Add noise-robust training** and advanced augmentation pipelines (e.g., MixUp, CutMix).  
- **Experiment with EfficientNet or Vision Transformers** for possibly better accuracy/efficiency trade-offs.  
- **Train on multiple datasets** and perform domain adaptation to increase robustness across environments.

##  Conclusion

This project delivers a powerful, end-to-end deep learning system for both classification and detection of paddy leaf diseases.
With an advanced ResNet50 backbone, detailed data analysis, rich visualizations, and Grad-CAM interpretability, this solution is suitable for:

--Academic research — reproducible experiments and interpretable visualizations.

--Agricultural automation — rapid on-field disease screening (after mobile export).

--Machine learning portfolios — a strong project demonstrating EDA, transfer learning, and explainability.

--Real-time crop health monitoring — as a foundation for web/mobile deployment.

