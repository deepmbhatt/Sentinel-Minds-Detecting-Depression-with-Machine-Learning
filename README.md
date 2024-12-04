# Sentinel Minds: Detecting Depression with Machine Learning  
![NumPy](https://img.shields.io/badge/NumPy-1.24.0-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.5.3-green)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11.2-yellowgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-blueviolet)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![Keras](https://img.shields.io/badge/Keras-2.6.0-red)
---

## 🌟 Introduction  
Mental health is a cornerstone of human well-being, yet it often goes unnoticed and untreated. **Sentinel Minds** is a machine learning-powered project designed to predict depression based on multiple lifestyle and demographic factors. By leveraging data analysis and predictive modeling, this project aims to contribute to early detection and awareness of mental health conditions, paving the way for timely intervention.

---

## 📋 Dataset Information  

The dataset contains comprehensive mental health and lifestyle data:  
- **Demographics**: ID, Name, Gender, Age, City  
- **Education & Profession**: Profession, Degree, Working Professional/Student, CGPA  
- **Stress Factors**: Academic/Work Pressure, Financial Stress  
- **Satisfaction Levels**: Study Satisfaction, Job Satisfaction  
- **Health & Lifestyle**: Sleep Duration, Dietary Habits, Family History of Mental Illness  
- **Outcome Variable**: Depression indicator (Yes/No)  

The data was preprocessed to handle missing values, ensure consistency, and prepare it for modeling.

---

## 🚀 Project Workflow  

1. **Data Preprocessing**: Cleaning and organizing the dataset for analysis.  
2. **Exploratory Data Analysis**: Visualizing patterns and relationships in the data using heatmaps, boxplots, and correlation graphs.  
3. **Feature Engineering**: Selecting relevant features that influence depression prediction.  
4. **Model Training**: Applying various machine learning models to classify individuals.  
5. **Model Evaluation**: Comparing performance metrics like accuracy to select the best model.  

---

## 🔬 Models Used  

1. **Decision Tree**: A simple tree-like structure for decision-making, achieved an accuracy of 90.23%.  
2. **K-Nearest Neighbors (KNN)**: A distance-based algorithm that classifies based on neighbors, with 91.51% accuracy.  
3. **Logistic Regression**: A statistical model for binary classification, achieving 93.73% accuracy.  
4. **MLP Classifier**: A deep learning-based model, with the highest accuracy of 93.89%.  
5. **Naive Bayes**: A probabilistic model using Bayes' theorem, achieved 89.83% accuracy.  
6. **Random Forest**: An ensemble model combining multiple decision trees, with 93.72% accuracy.  

---

## 📊 Model Performance  

| **Serial Number** | **Model Name**            | **Accuracy** |
|--------------------|---------------------------|--------------|
| 1                  | Decision Tree            | 0.902368     |
| 2                  | K-Nearest Neighbors      | 0.915132     |
| 3                  | Logistic Regression      | 0.937353     |
| 4                  | MLP Classifier           | 0.938918     |
| 5                  | Naive Bayes              | 0.898386     |
| 6                  | Random Forest            | 0.937247     |

---

## 📚 Libraries and Tools Used  

This project was powered by the following libraries:  
- **[NumPy](https://numpy.org/)**: Efficient numerical operations.  
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis.  
- **[Seaborn](https://seaborn.pydata.org/)**: Visualization of complex datasets.  
- **[Matplotlib](https://matplotlib.org/)**: Plotting detailed graphs and charts.  
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning model implementation.  
- **[TensorFlow](https://www.tensorflow.org/)**: Building neural networks.  
- **[Keras](https://keras.io/)**: Simplified API for neural networks.  

---

## 🎯 Objective  

To empower researchers and mental health professionals with tools for early detection of depression, enabling timely interventions to improve mental health outcomes.  

---

## 🔧 Getting Started  

To use this project locally, follow these steps:  

### 📂 Clone the Repository  

Run the following command in your terminal to clone this repository:  

```bash
git clone https://github.com/deepmbhatt/Sentinel-Minds-Detecting-Depression-with-Machine-Learning.git
