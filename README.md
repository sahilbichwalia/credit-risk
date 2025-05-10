# 🔍 Credit Risk Analysis Project

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![ScikitLearn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-78%25-success.svg)]()
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b.svg)](https://credit-risk-check.streamlit.app)

## 📊 Project Overview

This project implements a machine learning solution for credit risk assessment using credit bureau data. The model analyzes various credit-related features to predict approval status and prioritize applicants into different classes (P1-P4). The implemented XGBoost model achieves 78% accuracy in classifying credit applications.

## 🗂️ Dataset Description

The project utilizes two primary data tables:

### Internal Credit Data (Case Study 1)
Contains information about account histories, including:
- Account counts (total, active, closed)
- Recent account activity (opened/closed in last 6/12 months)
- Account types (Auto, Credit Card, Consumer, Gold loan, Home loan, etc.)
- Account age metrics

### CIBIL External Data (Case Study 2)
Contains credit bureau information, including:
- Delinquency history and patterns
- Payment behavior (standard, sub-standard, doubtful, loss)
- Credit inquiries by product and time period
- Utilization metrics
- Demographic information
- Credit scores

## ⚙️ Features & Technologies

- **Model**: XGBoost classifier with optimized hyperparameters
- **Feature Selection**: Chi-square testing and p-value analysis
- **Data Processing**: Feature scaling using SciPy
- **Performance**: 78% prediction accuracy
- **Implementation**: Python with scikit-learn integration

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-analysis.git
cd credit-risk-analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

## 📈 Model Details

The credit risk assessment model:

1. Processes internal and external credit data
2. Performs feature selection using statistical testing
3. Scales features for optimal model performance
4. Trains an XGBoost classifier
5. Prioritizes applicants into different classes (P1, P2, P3, P4) based on predicted risk levels

### Key Features Used

- Account history patterns
- Delinquency behaviors
- Inquiry patterns
- Utilization ratios
- Demographic factors

## 🧪 Model Evaluation

| Metric    | Overall | Class P1 | Class P2 | Class P3 | Class P4 |
|-----------|---------|----------|----------|----------|----------|
| Accuracy  | 78%     | -        | -        | -        | -        |
| Precision | -       | 85%      | 82%      | 48%      | 74%      |
| Recall    | -       | 76%      | 93%      | 27%      | 74%      |
| F1 Score  | -       | 80%      | 87%      | 34%      | 74%      |

## 📚 Documentation

Detailed documentation about the project implementation, model architecture, and feature importance analysis is available in the `/docs` directory.

## 📝 Acknowledgements

- Credit bureau data partners
- XGBoost library developers
- Streamlit team for the web application framework

## 🔗 Live Demo

Check out the live application at [https://credit-risk-check.streamlit.app](https://credit-risk-check.streamlit.app)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

Your Name - [teams.@example.com](mailto:teams@example.com)
