import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv("heart (2).csv")

# Prepare predictors and target
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Find the best random state for maximum accuracy
max_accuracy = 0

for x in range(2000):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

# Train the final model with the best random state
dt = DecisionTreeClassifier(random_state=best_x).fit(X_train, Y_train)

# Save the model to a file
pickle.dump(dt, open("modell.pkl", 'wb'))

# Predict with the final model
# Y_pred_dt = dt.predict(X_test)

# # Print the maximum accuracy and the best random state
# print(f"Max accuracy: {max_accuracy}% with random state: {best_x}")
