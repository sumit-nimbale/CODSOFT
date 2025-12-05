**📌 Project: Credit Card Fraud Detection**

This project identifies fraudulent credit card transactions using classification algorithms.


**📂 Project Structure**
Task 2 - Credit Card Fraud Detection/
│── dataset/    
│── models/    
│── notebook/    
│── results/    
└── README.md


**🎯 Objective**
- Detect fraudulent transactions
- Handle highly imbalanced datasets
- Use ML models like Logistic Regression, Random Forest, XGBoost
- Evaluate performance using appropriate metrics


**🧵 Workflow Summary**
- Load dataset
- Handle imbalance (oversampling/undersampling/SMOTE)
- Feature scaling
- Train multiple ML models
- Evaluate fraud detection performance
- Save results and best model


**📊 Evaluation Metrics**
Because the data is imbalanced, we focus on:
- Precision
- Recall
- F1 Score
- ROC–AUC Score


**📦 Output Files**

**Inside results/:**
roc_auc_curve.png
confusion_matrix.png
metrics_report.txt

**Inside models/:**
fraud_detection_model.pkl


**📝 Notebook**
Location:
notebook/credit_card_fraud_detection.ipynb
