# Analyzing-Emergency-Medical-Services-Response-Time
 Analyzing Emergency Medical Services Response Time
Emergency Medical Services (EMS) Response Time Analysis

Project Overview

This project aims to analyze EMS incident dispatch data to uncover patterns in response times, identify inefficiencies, and propose improvements. The analysis incorporates data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling using a custom neural network regressor.

Files in This Repository

ANALYZING_EMS_RESPONSE_TIME_PPT.pptx

A presentation summarizing the project's objectives, dataset insights, model pipeline, and key findings.

pipeline.py

Contains the code for the machine learning pipeline. This includes preprocessing steps, dimensionality reduction (PCA), and the integration of a PyTorch-based custom neural network regressor.

train.py

Implements the training and evaluation of the model pipeline. It uses train-test splitting, computes evaluation metrics (RMSE and R2), and integrates the pipeline from pipeline.py.

PracticalDS_EDA.ipynb

A Jupyter Notebook used for exploratory data analysis (EDA). It provides visualizations and statistical summaries of the EMS dataset, such as temporal patterns, severity distributions, and geographic analysis.

And many other files

Key Components

Data Preprocessing

Parsed datetime features: hour_of_day, day_of_week, and month.

Categorized features as:

Numerical: Dispatch and travel times.

Ordinal: Severity levels and location-based codes.

Nominal: Call types, boroughs, and event indicators.

Handled missing values and scaled numerical data.

Exploratory Data Analysis (EDA)

Correlation heatmaps to identify relationships between features.

Distribution plots for response and travel times.

Boxplots and scatter plots for outlier detection and feature analysis.

Model Pipeline

Preprocessor:

Nominal features: OneHotEncoder

Ordinal features: OrdinalEncoder

Numerical features: StandardScaler

Dimensionality Reduction:

PCA with 10 components

Regressor:

PyTorch-based feedforward neural network:

Input → Fully Connected Layers (16, 32 nodes) → Output

Activation: ReLU

Final Activation: SoftPlus

Optimizer: Adam

Loss Function: Mean Squared Error

Model Evaluation

Metrics:

Root Mean Square Error (RMSE): 3.22 seconds

R2 Score: 0.999

How to Run the Code

Prerequisites

Python 3.8+

Required libraries:

numpy, pandas, scikit-learn, torch

Steps

Clone the repository:

git clone https://github.com/your-repo/ems-response-analysis.git
cd ems-response-analysis

Install dependencies:

pip install -r requirements.txt

Place your dataset in the root folder as ems_data.csv (or modify the code to use your dataset path).

Run the model training script:

python train.py

View the evaluation metrics printed in the console.

Future Scope

Real-Time Predictions: Integrate the model into live EMS systems.

Advanced Features: Include traffic and weather data to improve predictions.

Explainability: Add tools like SHAP or LIME to interpret model outputs.

Model Optimization: Experiment with ensemble learning techniques and hyperparameter tuning.

Contributors

Roshan Niranjan

Kalpavruksha Tejomay K

Vindya Sree Kanchi

Kamalaesh Katari