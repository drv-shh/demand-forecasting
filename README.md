# Retail Demand Forecasting

This project builds a basic machine learning pipeline to forecast retail product demand using historical sales data.

## Project Workflow
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Demand Prediction
7. API for Predictions


## Tech Stack
- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- FastAPI


## Project Structure
load_data.py        – Load and preprocess dataset  
eda.py              – Exploratory data analysis  
feature_eng.py      – Feature engineering  
model_train.py      – Train ML model  
predict.py          – Generate predictions  
api.py              – Serve predictions via API  


## Dataset
This project uses a retail sales dataset for demand forecasting.
Source: Kaggle  

Link: https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset/data

After downloading, place the dataset in the project root directory before running the pipeline.

## Goal
To buid an end-to-end machine learning pipeline for retail demand prediction.