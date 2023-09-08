from utils import *
from itertools import product

# 1. Get the dataset
digits = datasets.load_digits()

# 2. Data splitting -- to create train, dev, and test sets

def list_of_all_test_dev_size_combination(*lists):
    return list(product(*lists))


test_size = [0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]
# kernel_types = ['linear', 'rbf']

list_of_all_test_dev_size_combinations = list_of_all_test_dev_size_combination(test_size, dev_size)
print("Number of test dev size combinations: {}".format(len(list_of_all_test_dev_size_combinations)))

#4 Hyper-parameter Tuning

############################## The below fuction can handle any number of lists ###################################

def list_of_all_param_combination(*lists):
    return list(product(*lists))


gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
# kernel_types = ['linear', 'rbf']

list_of_all_param_combinations = list_of_all_param_combination(gamma_ranges, C_ranges)

print("Number of parameter combinations: {}".format(len(list_of_all_param_combinations)))

for i, test_dev_size_combination in enumerate(list_of_all_test_dev_size_combinations):
    test_split, dev_split = test_dev_size_combination


    data = digits.images
    X_train, X_dev, X_test, Y_train, y_dev, y_test = split_train_dev_test(data, digits.target, test_size=test_split, dev_size=dev_split)

    # 3. Data preprocessing
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)
    X_test = preprocess_data(X_test)

    model = train_model(X_train, Y_train, model_params={"gamma":0.001})
    train_acc = predict_and_eval(model, X_train, Y_train)
    dev_acc = predict_and_eval(model, X_dev, y_dev)
    test_acc = predict_and_eval(model, X_test, y_test)

    print(f"test_size: {test_split} dev_size: {dev_split} train_size: {1-test_split-dev_split} train_acc: {train_acc} dev_acc: {dev_acc} test_acc: {test_acc} ")

    best_model, optimal_gamma, optimal_C, test_acc= tune_hyper_params(X_train, X_dev, X_test, Y_train, y_dev, y_test, list_of_all_param_combinations)






