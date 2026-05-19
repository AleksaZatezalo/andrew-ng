"""
Description: Basic Classifier in python to classify tumors.
Author: Aleska Zatezalo
Date: May 19, 2026
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd

# Import data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(X.shape)
print(y.value_counts())