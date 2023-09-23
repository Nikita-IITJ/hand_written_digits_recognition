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