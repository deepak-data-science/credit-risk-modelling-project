# CU Finance: Credit Risk Modeling

A machine learning-powered web application for credit risk assessment and scoring, built with Streamlit.

## 📊 Overview

This project focuses on building a Credit Risk Classification Model as part of partial fulfilment for the award of the degree of MASTER OF SCIENCE IN DATA SCIENCE. The goal was to predict whether a customer is likely to default based on multidimensional financial and credit-history attributes. The project provided experience in both machine learning techniques and financial risk modeling concepts commonly used in the credit industry.

This application helps financial institutions assess credit risk by analyzing various customer financial parameters and generating credit scores with risk ratings. It uses a trained machine learning model to predict the probability of default and provides actionable credit scores ranging from 300 to 900.

## Dataset :

The dataset was provided by Atliq Technologies for training purpose.

## 🔧 Technical & Analytical Skills Applied

- **Exploratory Data Analysis (EDA)**: Used box plots, histograms, KDE plots, and correlation analysis to understand data distribution, behavior, and key relationships.

- **Feature Engineering**:
  - Handled missing data
  - Created derived features such as average DPD, delinquency ratio, and credit utilization per income
  - Performed multicollinearity checks using VIF
  - Conducted WOE & IV analysis for feature strength assessment

- **Model Development**:
  - Built and compared multiple classifiers: Logistic Regression, Random Forest, XGBoost
  - Performed hyperparameter tuning with RandomizedSearchCV
  - Applied SMOTETomek for class imbalance correction
  - Further optimized the best model using Optuna (Bayesian optimization)

- **Model Evaluation**:
  - Assessed performance using ROC Curve, F1-score, KS Statistic, and Rank Ordering
  - Used SHAP summary and force plots for explainability and model interpretability

- **Deployment**:
  - Built an interactive MVP using Streamlit
  - Version-controlled and published on GitHub for future improvements


## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard for inputting customer data
- **Real-time Risk Assessment**: Instant calculation of default probability and credit scores
- **Credit Rating System**: Automatic categorization into Poor, Average, Good, or Excellent ratings
- **Comprehensive Input Analysis**: Considers multiple financial factors including:
  - Age and income details
  - Loan parameters (amount, tenure, purpose, type)
  - Credit history (delinquency ratio, DPD, utilization)
  - Residence type and account information

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: Scikit-learn
- **Model Storage**: Joblib
- **Experiment Tracking**: MLflow
- **Data Processing**: Pandas, NumPy

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. **Clone the repository**:
    ```bash
   git clone <repository-url>
   https://github.com/deepak-data-science/credit-risk-modelling-project.git
    ```
   2. **Create a virtual environment** (recommended):
    ```bash
   python -m venv venv
   ```
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   3. **Install dependencies**:
    ```bash
   pip install -r requirements.txt
    ```

## 🎯 Usage

1. **Run the application**:
    ```bash
   streamlit run main.py
    ```
   2. **Access the web interface**:
   Open your browser and navigate to `http://localhost:8501`

3. **Input customer data**:
   - Enter age, income, and loan details
   - Provide credit history information
   - Select residence type, loan purpose, and loan type

4. **Generate assessment**:
   Click "Calculate Risk" to get:
   - Default probability percentage
   - Credit score (300-900 range)
   - Risk rating category
---
## 📁 File Structure

```
ml-project-credit-risk-model/
│
├── main.py                           # Streamlit web application
├── prediction_helper.py              # ML prediction logic and utilities
├── credit_risk_model.ipynb           # Jupyter notebook (model training/development)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── artifacts/
│   └── model_data.joblib            # Trained model and preprocessing components
│
├── dataset/
│   ├── bureau_data.csv              # Bureau/credit bureau data
│   ├── customers.csv                # Customer information
│   └── loans.csv                    # Loan application data
│
├── mlruns/                          # MLflow experiment tracking
│   ├── 0/
│   │   ├── meta.yaml
│   │   └── tags/
│   │       └── mlflow.experimentKind
│   └── models/                      # (currently empty)
│
└── __pycache__/  
```
---

### Input Features
- Loan amount and tenure
- Average days past due (DPD)
- Delinquency ratio
- Credit utilization ratio
- Number of open accounts
- Residence type
- Loan purpose and type

### Output Metrics
- **Default Probability**: Likelihood of loan default (0-100%)
- **Credit Score**: Numerical score from 300 to 900
- **Risk Rating**:
  - Poor: 300-499
  - Average: 500-649
  - Good: 650-749
  - Excellent: 750-900

## 🤖 Machine Learning Pipeline

1. **Data Preprocessing**: Feature scaling and encoding
2. **Model Training**: Logistic regression with hyperparameter tuning
3. **Model Evaluation**: Performance metrics and validation
4. **Model Deployment**: Joblib serialization for production use


## 🔒 Dependencies

Key libraries used:
- `streamlit`: Web application framework
- `scikit-learn`: Machine learning algorithms
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `joblib`: Model serialization
- `xgboost`: Gradient boosting (for model training)
- `optuna`: Hyperparameter optimization
- `imbalanced-learn`: Handling imbalanced datasets

## 📈 Model Performance

### 📊 Model Performance Summary

The classification model demonstrates strong overall performance with a focus on identifying minority-class events accurately.

✔ **Best Macro F1 Score: 0.7875**

#### Classification Report
- **Accuracy: 93%**
- **Majority Class (0):**
  - Precision: 0.99
  - Recall: 0.93
  - F1-score: 0.96
- **Minority Class (1):**
  - Precision: 0.56
  - Recall: 0.95
  - F1-score: 0.71

The model shows excellent recall for the minority class, making it suitable for risk-sensitive applications.

---

#### 📈 ROC–AUC Score: 0.98

The model achieves a high AUC, indicating strong separability between the two classes.

---

#### 📊 KS (Kolmogorov–Smirnov) Statistic

- **Maximum KS: ~86%**
- Indicates strong discriminatory power between events and non-events.
- High KS in the top deciles confirms effective ranking of high-risk predictions.

---

#### 🔟 Decile Analysis

- **Decile 0 (highest-risk bucket)** captures ~84% of all events, showing excellent concentration.
- Event probability sharply drops across deciles, confirming strong model calibration.
- Lower deciles contain predominantly non-events, validating separation.

### Technical Details

The credit risk model has been trained and validated on historical loan data with the following characteristics:
- Uses logistic regression for binary classification
- Features preprocessing with MinMax scaling
- Handles categorical variables through one-hot encoding

## 🚨 Important Notes

- The model uses dummy values for certain features during prediction to maintain feature consistency
- All monetary inputs should be in appropriate units (e.g., annual income)
- The application is designed for demonstration and educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Deployment Link :
https://credit-risk-modelling-project-cu.streamlit.app

## Web Interface :
<img width="684" height="509" alt="image" src="https://github.com/user-attachments/assets/a8d95c35-ec49-4d6d-b157-a89f7fee3443" />


## 📄 License

This project is intended for educational purposes.

## 🙏 Acknowledgments

- Datasets are provided by Atliq Technology for training purpose only.
- Uses industry-standard practices for credit risk modeling.
- Inspired by real-world financial technology applications.

---
