# Predicting Musculoskeletal Diseases Using Biomarkers

## Description  
This project was undertaken as a capstone for the course **BG4104: Machine Learning & Optimisation for Bioengineers**, part of the **Machine Learning and Data Analytics** specialization at Nanyang Technological University, Singapore. The goal was to predict musculoskeletal diseases (MSD) in elderly individuals using biomarkers such as handgrip strength, age, sex, and BMI. By leveraging supervised machine learning models, this project aims to enhance early detection of MSD. Three machine learning algorithms—FeedForward Neural Networks (FFNN), Support Vector Machines (SVM), and K-Nearest Neighbors (KNN)—were implemented and evaluated, with KNN emerging as the most promising in specific cases.  

---

## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Machine Learning Models](#machine-learning-models)  
4. [Results](#results)  
5. [Future Improvements](#future-improvements)  
6. [Technologies Used](#technologies-used)  
7. [Acknowledgments](#acknowledgments)  
8. [References](#references)  

---

## Project Overview  
Musculoskeletal diseases are a global concern, particularly among the elderly, as they impact mobility and quality of life. This project investigates the use of machine learning to enhance early detection. The dataset was sourced from a study on elderly Malaysians and focuses on biomarkers linked to MSD risk.  
Key steps:  
- Preprocessing the dataset (handling class imbalance with SMOTE).  
- Hyperparameter tuning with GridSearchCV.  
- Evaluation using metrics such as F1-score, recall, and precision.  

---

## Dataset  
The dataset, titled **"HGS and NCDs Elderly Malaysia"**, was sourced from a study conducted by Zulkefley Mohammad and Shamsul Azhar Shah. It contains data from 1204 elderly participants aged 60–91, focusing on handgrip strength and non-communicable diseases. The dataset is publicly available at [this link](https://data.mendeley.com/datasets/hsc4k7vtfp/1/files/637771d5-40d3-4663-bbec-238335db7491) and is published with the DOI [10.17632/hsc4k7vtfp.1](https://doi.org/10.17632/hsc4k7vtfp.1).  

The selected features for this project include:  
- **Age**  
- **Sex** (0: Male, 1: Female)  
- **BMI**  
- **Handgrip Strength**  
- **MSD Status** (0: No, 1: Yes)  

A significant challenge was class imbalance, with only 15.6% of cases being positive for MSD. SMOTE was applied to address this issue, improving sensitivity to minority class predictions.  

---

## Machine Learning Models  
### 1. FeedForward Neural Network (FFNN):  
- Architectures tested: Single and double hidden layers.  
- Optimized hyperparameters: Hidden layer size, activation function, learning rate, and alpha (regularization).  
- **Best parameters:**  
  - Hidden layers: (180, 90)  
  - Activation: ReLU  
  - Learning rate: 0.01  
  - Alpha: 0.005  

### 2. Support Vector Machines (SVM):  
- Kernels tested: Linear, RBF, Sigmoid.  
- Optimized hyperparameters: C (regularization), gamma (flexibility), and coef0 (impact of data points).  
- **Best parameters:**  
  - Linear kernel: C = 0.1  
  - RBF kernel: C = 10, Gamma = 2  
  - Sigmoid kernel: C = 0.1, Gamma = 0.5, Coef0 = 0  

### 3. K-Nearest Neighbors (KNN):  
- Optimized hyperparameters: Number of neighbors (k), weighting, and distance metric.  
- **Best parameters:**  
  - Neighbors: 5  
  - Weighting: Distance  
  - Distance metric: Euclidean  

---

## Results  
### Performance Comparison (80-20 Split):  

| **Model** | **Metric**   | **F1-Score** | **Recall** | **Precision** | **Best Parameters**                              |  
|-----------|--------------|--------------|------------|---------------|-------------------------------------------------|  
| **FFNN**  |   | 0.89         | 0.91       | 0.87          | Hidden layers: (180, 90), ReLU, LR: 0.01, α: 0.005 |  
| **SVM**   | Linear       | 0.85         | 0.84       | 0.86          | C = 0.1                                          |  
|           | RBF          | 0.83         | 0.82       | 0.84          | C = 10, Gamma = 2                               |  
|           | Sigmoid      | 0.80         | 0.79       | 0.81          | C = 0.1, Gamma = 0.5, Coef0 = 0                 |  
| **KNN**   |  | 0.91         | 0.93       | 0.89          | Neighbors: 5, Weighting: Distance, Metric: Euclidean |  

- **FFNN** performed well with high flexibility due to its hyperparameter tuning but required significant computational resources.  
- **SVM** showed consistent performance but lagged in complex relationships, with the sigmoid kernel being the weakest.  
- **KNN** outperformed others due to the dataset’s structured and low-dimensional nature. It was computationally efficient and delivered robust generalization.  

---

## Future Improvements  
- **Feature Engineering:** Introduce derived metrics like relative handgrip strength (grip strength/BMI) or include lifestyle factors (e.g., smoking, alcohol consumption).  
- **Ensemble Learning:** Combine models using bagging (e.g., RandomForestClassifier) or boosting (e.g., AdaBoost, XGBoost) to improve robustness.  
- **Data Expansion:** Collect more diverse data to improve generalization and explore higher-dimensional feature spaces.  

---

## Technologies Used  
- **Programming Language:** Python  
- **Libraries:** Scikit-learn, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn  

---

## Acknowledgments  
This project was developed as part of the BG4104 course at Nanyang Technological University, Singapore. Special thanks to Professors **Ong Chi Wei** and **Park Seung Min** for their guidance and support throughout this project.  

---

## References  
- Zulkefley Mohammad & Shamsul Azhar Shah. *HGS and NCDs Elderly Malaysia Dataset.* Mendeley Data. DOI: [10.17632/hsc4k7vtfp.1](https://doi.org/10.17632/hsc4k7vtfp.1). Published: 18 November 2020.  
- Vaishya et al. (2024). *Hand grip strength as a proposed new vital sign of health.* Journal of Health, Population and Nutrition.  
