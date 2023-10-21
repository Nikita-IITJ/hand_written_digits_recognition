import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load the MNIST dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search for SVM
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)

# Predict using the best SVM model
prod_predictions = grid_search_svm.predict(X_test)
print(prod_predictions)

# Define parameter grid for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Grid search for Decision Tree
grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
grid_search_dt.fit(X_train, y_train)

# Predict using the best Decision Tree model
cand_predictions = grid_search_dt.predict(X_test)
print(cand_predictions)

# Production model's accuracy
prod_accuracy = accuracy_score(y_test, prod_predictions)
print(f"Production model's accuracy: {prod_accuracy:.2f}")

# Candidate model's accuracy
cand_accuracy = accuracy_score(y_test, cand_predictions)
print(f"Candidate model's accuracy: {cand_accuracy:.2f}")

# Confusion matrix between predictions of production and candidate models
conf_matrix = confusion_matrix(prod_predictions, cand_predictions)
print("Confusion matrix between predictions of production and candidate models:")
print(conf_matrix)

# 2x2 confusion matrix
correct_prod = (prod_predictions == y_test)
correct_cand = (cand_predictions == y_test)
conf_matrix_2x2 = confusion_matrix(correct_prod, correct_cand)
print("2x2 Confusion matrix:")
print(conf_matrix_2x2)

# Bonus: Macro-average F1 metrics
f1_prod = f1_score(y_test, prod_predictions, average='macro')
f1_cand = f1_score(y_test, cand_predictions, average='macro')
print(f"Production model's macro-average F1 score: {f1_prod:.2f}")
print(f"Candidate model's macro-average F1 score: {f1_cand:.2f}")


