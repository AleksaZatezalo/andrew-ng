"""
Description: Basic Classifier in python to classify tumors.
Author: Aleska Zatezalo
Date: May 19, 2026
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load
data = load_breast_cancer()
X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train
model = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Feature importance
importance = pd.Series(np.abs(model.coef_[0]), index=data.feature_names)
print(importance.sort_values(ascending=False).head(10))