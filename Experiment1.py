"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import necessary libraries
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import preprocess_data,split_data,train_model,split_train_dev_test,predict_and_eval
#import pdb
#from sklearn.svm import SVC

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# 1. Get the dataset
digits = datasets.load_digits()

# 2.1. Print the number of total samples in the dataset (train + test + dev)
total_samples = len(digits.images)
print(f"Total number of samples in the dataset: {total_samples}")

# 2.2. Print the size (height and width) of the images in the dataset
image_shape = digits.images[0].shape
print(f"Size (height and width) of the images in dataset: {image_shape[0]} x {image_shape[1]}")

# 2. Qualitative sanity check of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.


# 3. Data splitting -- to create train, dev, and test sets
data = digits.images
# import ipdb; ipdb.set_trace()
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(data, digits.target, test_size=0.3, dev_size=0.25)

# 4. Data preprocessing
X_train = preprocess_data(X_train)
X_dev = preprocess_data(X_dev)
X_test = preprocess_data(X_test)


# 5. Model training
model=train_model(X_train,y_train,{'gamma':0.001},model_type="svm")

# 6. Getting model predictions on test set
# Predict the value of the digit on the test subset
predicted_test = model.predict(X_test)
predicted_dev=model.predict(X_dev)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# 7. Qualitative sanity check of the predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

#8. Evaluation
predict_and_eval(model, predicted_dev, y_dev)

predict_and_eval(model, predicted_test, y_test)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)