from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.svm import SVC


# Load the MNIST dataset
digits = datasets.load_digits()

# Instantiate the Normalizer
normalizer = Normalizer()

# Apply the Normalizer
X_normalized = normalizer.fit_transform(digits.data)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, digits.target, test_size=0.3, random_state=42)

solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
rollno = "m22aie244"  

for solver in solvers:
    model = LogisticRegression(solver=solver, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = scores.mean()
    std_score = scores.std()

    # Print performance
    print(f"Performance for solver {solver}: Mean - {mean_score}, Std - {std_score}")

    # Save the model
    joblib.dump(model, f"{rollno}_lr_{solver}.joblib")



rollno = "m22aie244"
crit = "entropy"
depth = 10

model = DecisionTreeClassifier(criterion=crit, max_depth=depth)
model.fit(X_train, y_train)

# Evaluate the model using cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
mean_score = scores.mean()
std_score = scores.std()

# Print performance
print(f"Performance for criterion {crit} and max depth {depth}: Mean - {mean_score}, Std - {std_score}")

# Save the model
model_name = f"{rollno}_tree_{crit}_{depth if depth is not None else 'None'}.joblib"
joblib.dump(model, model_name)

# Using the 'rbf' kernel for SVM
kernel = 'rbf'
rollno = "m22aie244"

# Create and fit the SVM model
model = SVC(kernel=kernel, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model using cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
mean_score = scores.mean()
std_score = scores.std()

# Print performance
print(f"Performance for kernel {kernel}: Mean - {mean_score}, Std - {std_score}")

# Save the model
joblib.dump(model, f"{rollno}_svm_{kernel}.joblib")


