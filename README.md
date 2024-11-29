# Protein/Peptide Thermal Stability Prediction

## Project Overview

This project aims to predict the melting temperature (Tm) of proteins, a key indicator of their thermal stability, using machine learning (ML) and deep learning (DL) models. Accurate prediction of protein Tm is critical in various fields, including drug design, protein engineering, and understanding disease mechanisms.

## Objectives
- **Develop 3D Convolutional Neural Networks (CNNs)** to predict protein Tm from 3D structural data.
- **Implement Random Forest** to predict Tm using sequence-derived features and physicochemical properties of proteins.

## Data Collection and Preprocessing

The primary data source is the **Protein Data Bank (PDB)**, which provides detailed 3D structural data for proteins. Key preprocessing steps include:
- **Reading PDB files** for structural information.
- **Centering protein coordinates** to standardize position.
- **Creating a 3D grid representation** of the protein structures.
- **Extracting atomic and physicochemical features** for model input.

## Model Development

### 1. **3D Convolutional Neural Networks (CNNs)**
- **Input Layer:** 3D grid representation of proteins.
- **Convolutional Layers:** Conv3D layers with ReLU activation to capture spatial features.
- **Pooling Layers:** MaxPooling3D for dimensionality reduction.
- **Dense Layers:** Fully connected layers for integrating learned features.
- **Output Layer:** Predicting continuous Tm values.

### 2. **Graph Neural Networks (GNNs)**
- **Graph Convolutional Layers:** Aggregating features from neighboring atoms.
- **Message Passing:** Capturing local and global graph patterns for better structural understanding.

### 3. **Random Forest**
- An ensemble method used to predict Tm based on sequence-derived features, including amino acid properties and sequence length.

## Results and Evaluation

- The **3D CNN** model achieved strong predictive performance, with a **Mean Absolute Error (MAE)** of **13.37** and an **R²** of **0.35** on the test set. The model successfully captured complex spatial features influencing protein stability.
- **Random Forest** and other traditional ML models were explored, with **Random Forest** showing robust performance in predicting Tm based on sequence-derived features.

## Challenges
- **Computational Complexity:** 3D CNNs require substantial computational resources, especially when working with large protein datasets.
- **Data Quality:** Variability in the quality of PDB structures affected the consistency of model performance.
- **Model Generalization:** Overfitting and achieving generalization to unseen data remain ongoing challenges.

## Future Work
- **Feature Extraction Enhancement:** Integrate **secondary structure information** from tools like **GMX do_dssp** to improve feature representation.
- **Advanced GNN Architectures:** Explore more complex GNN models and hybrid approaches that combine CNN and GNN techniques for enhanced performance.
- **Model Generalization:** Expand the dataset, apply **regularization techniques**, and explore strategies to improve model robustness and avoid overfitting.

---


### Acknowledgments
- **Protein Data Bank (PDB)** for providing the structural data.
- **Scikit-learn** for machine learning tools.
- **TensorFlow** and **PyTorch** for deep learning model development.
