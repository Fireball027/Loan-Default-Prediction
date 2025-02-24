## Overview

The **Loan Default Prediction Project** applies **Deep Learning with Keras** to predict whether a borrower is likely to default on a loan. Using **LendingClub's public loan dataset**, this project explores data preprocessing, feature engineering, and neural network modeling to achieve high accuracy in financial risk assessment.

---

## Key Features

- **Data Preprocessing**: Cleans, encodes, and scales loan data for training.
- **Exploratory Data Analysis (EDA)**: Identifies patterns and insights in borrower data.
- **Neural Network Model**: Implements a deep learning model using **Keras**.
- **Model Evaluation**: Uses accuracy, precision-recall, and loss functions for performance assessment.
- **Visualization**: Displays loan status distribution and model training metrics.

---

## Project Files

### 1. `lending_club_loan_two.csv`
This dataset contains historical loan data, including:
- **loan_amnt**: The loan amount requested by the borrower.
- **term**: Loan repayment term (36 or 60 months).
- **int_rate**: Interest rate charged on the loan.
- **installment**: Fixed monthly loan payment.
- **grade & sub_grade**: Creditworthiness categories assigned to the loan.
- **loan_status**: Whether the loan was fully paid or defaulted (target variable).

### 2. `lending_club_info.csv`
This file provides descriptions of the dataset features for better understanding.

### 3. `Keras_Project.py`
This script processes the dataset, builds a deep learning model, and evaluates performance.

#### Key Components:

- **Data Preprocessing**:
  - Converts categorical features into numerical representations.
  - Normalizes numerical features for better model performance.

- **Neural Network Architecture**:
  - Uses multiple dense layers with **ReLU** activation.
  - Implements **Dropout** for regularization.
  - Uses **Binary Crossentropy** as the loss function for classification.

- **Model Training & Evaluation**:
  - Splits data into training and testing sets.
  - Compiles the model with **Adam optimizer**.
  - Evaluates performance using **accuracy and loss metrics**.

#### Example Code:
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
data = pd.read_csv('lending_club_loan_two.csv')

# Preprocessing
encoder = LabelEncoder()
data['loan_status'] = encoder.fit_transform(data['loan_status'])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('loan_status', axis=1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_scaled, data['loan_status'], test_size=0.3, random_state=42)

# Define Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
```

---

## How to Run the Project

### Step 1: Install Dependencies
Ensure you have Python installed, then install required libraries:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```

### Step 2: Run the Script
Execute the main script:
```bash
python Keras_Project.py
```

### Step 3: View Insights
- Model training accuracy and loss.
- Visualizations of loan status distribution.
- Feature importance analysis.

---

## Future Enhancements

- **Hyperparameter Tuning**: Optimize network architecture for better accuracy.
- **Feature Engineering**: Introduce additional financial risk indicators.
- **Advanced Neural Networks**: Experiment with **LSTMs or Transformer models** for sequential financial data.
- **Deployment**: Convert model into an interactive web-based loan prediction tool.

---

## Conclusion

The **Loan Default Prediction Project** leverages **deep learning** to assess loan risks efficiently. By training a **neural network model** on LendingClub loan data, this project helps identify patterns in loan defaults, enabling better risk management for financial institutions.

---

**Happy Predicting! 🚀**

