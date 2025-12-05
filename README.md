# **Credit Risk Modeling Application**

A production-style machine learning project that evaluates customer creditworthiness by predicting the probability of loan default and generating an interpretable credit score. The project combines strong statistical modeling practices with a clean Streamlit interface to support data-driven lending decisions.

---

## **📌 Project Overview**

This application predicts whether a borrower is likely to default based on multidimensional financial behavior, credit history and loan characteristics. It was developed as part of the requirements for the **Master of Science in Data Science** program.

The model produces:

* **Default probability**
* **Credit score (300–900)**
* **Risk rating**: Poor, Average, Good or Excellent

The system is designed to help financial institutions and digital lenders make more informed credit decisions.

---

## **📂 Dataset**

The dataset was provided by **Atliq Technologies** for academic learning.
It includes:

* Customer demographic information
* Loan application details
* Bureau history such as delinquency and utilization
* Account-level information

---

## **🧠 Skills & Techniques Used**

### **Exploratory Data Analysis**

* Distribution study through histograms, KDE plots and boxplots
* Correlation and multicollinearity checks
* Outlier and missing-value analysis

### **Feature Engineering**

* Derived metrics such as average DPD, delinquency ratio, utilization ratio
* Encoding categorical variables
* WOE & IV analysis for feature strength
* VIF analysis for reducing multicollinearity

### **Model Development**

* Logistic Regression, Random Forest, XGBoost
* Hyperparameter tuning with **RandomizedSearchCV** and **Optuna**
* Handling class imbalance using **SMOTETomek**
* Model explainability using **SHAP**

### **Model Evaluation**

* ROC AUC: **0.98**
* Macro F1 Score: **0.7875**
* Strong recall for minority class
* KS Statistic: **~86%**
* Decile-wise event capture analysis

### **Deployment**

* Interactive **Streamlit** web application
* Model serialized using **joblib**

---

## **🚀 Features**

* Clean and intuitive Streamlit UI
* Real-time default probability prediction
* Automatic credit score generation (300–900)
* Risk scoring categories
* Multi-parameter input support (loan, credit, income, bureau info)
* Lightweight and easy to run locally

---

## **🛠 Technology Stack**

| Layer               | Tools                                   |
| ------------------- | --------------------------------------- |
| Frontend            | Streamlit                               |
| Backend             | Python                                  |
| ML Frameworks       | scikit-learn, XGBoost, imbalanced-learn |
| Optimization        | Optuna                            |                       |
| Model Serialization | Joblib                                  |

---

## **📁 Project Structure**

```
project/
│
├── main.py                     # Streamlit app
├── prediction_helper.py        # Prediction utilities and preprocessing
├── credit_risk_model.ipynb     # Model development notebook
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── artifacts/
│   └── model_data.joblib       # Trained model and pipeline
│
├── dataset/
   ├── customers.csv
   ├── loans.csv
   └── bureau_data.csv

```

---

## **🔧 Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/deepak-data-science/credit-risk-modelling-project.git
```

### **2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
source venv/bin/activate
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **▶️ Running the Application**

Run the Streamlit app:

```bash
streamlit run main.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## **📈 Model Output**

The model generates:

### **1. Default Probability**

Likelihood (0 to 100 percent) of loan default.

### **2. Credit Score (300–900)**

Automatically derived based on predicted risk.

### **3. Risk Rating**

| Score   | Rating    |
| ------- | --------- |
| 300–499 | Poor      |
| 500–649 | Average   |
| 650–749 | Good      |
| 750–900 | Excellent |

---

## **📊 Performance Summary**

* **Accuracy:** 93 percent
* **Minority Class Recall:** 95 percent
* **ROC AUC:** 0.98
* **KS Statistic:** ~86 percent
* **Top Decile Capture:** ~84 percent of defaulters in first decile

These metrics confirm strong model separation and risk-ranking capabilities.

---

## **🌐 Live Deployment**

Streamlit Cloud:
[https://credit-risk-modelling-project-cu.streamlit.app](https://credit-risk-modelling-project-cu.streamlit.app)

Web Interface:

<img width="684" height="509" alt="image" src="https://github.com/user-attachments/assets/a8d95c35-ec49-4d6d-b157-a89f7fee3443" />

---



## **🤝 Contributing**

Contributions are welcome. To contribute:

1. Fork the repo
2. Create a new branch
3. Make changes
4. Submit a pull request

---

## **📄 License**

This project is intended for academic and educational use.

---
