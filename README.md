# RecSysChallenge2024

## Overview
The RecSysChallenge2024 is an esteemed competition focused on developing advanced news recommendation systems. Organized by a collaboration of leading institutions and based on data provided by Ekstra Bladet, the challenge addresses both technical and normative aspects of news recommendations. Participants work with the extensive Ekstra Bladet News Recommendation Dataset (EB-NeRD), containing millions of user interactions and article features, to predict which article a user will click from a list of articles that was seen during a specific impression.
The goal is to create models that accurately reflect user preferences while considering the broader impacts on news consumption.

URL of the RecSysChallenge2024: https://www.recsyschallenge.com/2024/

## Table of Contents

- [Solution](#solution)
  - [Data Analysis](#data-analysis)
  - [Model](#model)
  - [Data Sets](#data-sets)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Results](#results)
  - [Example Results](#example-results)

## Solution
### Data Analysis
The project involves comprehensive data analysis, including:

- Extracting and processing time-based features such as the hour of the day and the day of the week.
- Calculating user behavior metrics like average read time, scroll percentage, and total clicks.
- Encoding categorical features and handling missing values through imputation.
- Combining user behaviors with article attributes such as category, type, sentiment, and age.

### Model
- **XGBoost**: A powerful gradient boosting framework known for its efficiency and accuracy in classification tasks. XGBoost helps in capturing complex patterns in the data by building an ensemble of decision trees.
- **ADASYN (Adaptive Synthetic Sampling Approach for Imbalanced Learning)**: A technique to address data imbalance by generating synthetic samples for the minority class. This ensures that the model learns effectively from both classes, leading to better performance on imbalanced datasets.

### Data Sets
**URL for the Data Sets** Provided by the challenge organizers : https://recsys.eb.dk/

- **Behavior:** The behavior logs for the 7-day data split period (behaviors.parquet)
- **History:** The users' click histories (history.parquet)
- **Articles:** Detailed information of news articles (articles.parquet)

## Technology Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib

## Getting Started
### Prerequisites

- Python 3.12

### Installation
Follow these steps to set up the project locally:

1. Clone the repository to your local machine:
    ```sh
    git clone https://github.com/natalienakkk/RecSysChallenge2024.git
    ```

2. Navigate to the project directory:
    ```sh
    cd RecSysChallenge2024
    ```

3. Install the necessary packages:
    ```sh
    pip install pandas
    pip install scikit-learn
    pip install xgboost
    pip install imbalanced-learn
    pip install matplotlib
    ```

## Results
The recommendation system was evaluated based on several metrics, including:

- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- ROC AUC Score
- PR AUC Score
- Visualizations such as feature importance plots were also generated to interpret the model's decisions.

### Example Results

#### 1.1. Training Performance on demo dataset
| Metric                  | Value                |
|-------------------------|----------------------|
| **Accuracy**            | 0.77155700542283     |
| **ROC AUC Score**       | 0.8679585983141518   |
| **PR AUC Score**        | 0.8938249050915605   |

#### Classification Report
| Class | Precision | Recall | F1-Score | Support  |
|-------|-----------|--------|----------|----------|
| **0** | 0.72      | 0.86   | 0.79     | 253310   |
| **1** | 0.84      | 0.68   | 0.75     | 264132   |
| **Accuracy**     |           |        | 0.77     | 517442   |
| **Macro avg**    | 0.78      | 0.77   | 0.77     | 517442   |
| **Weighted avg** | 0.78      | 0.77   | 0.77     | 517442   |

#### 1.2. Validation Performance on demo dataset
| Metric                  | Value               |
|-------------------------|---------------------|
| **Accuracy**            | 0.8502927045242117  |
| **ROC AUC Score**       | 0.6435850438899733  |
| **PR AUC Score**        | 0.12880136110875698 |

#### Classification Report
| Class | Precision | Recall | F1-Score | Support  |
|-------|-----------|--------|----------|----------|
| **0** | 0.93      | 0.80   | 0.86     | 279471   |
| **1** | 0.14      | 0.35   | 0.20     | 25444    |
| **Accuracy**     |           |        | 0.85     | 304915   |
| **Macro avg**    | 0.54      | 0.58   | 0.53     | 304915   |
| **Weighted avg** | 0.87      | 0.76   | 0.81     | 304915   |
