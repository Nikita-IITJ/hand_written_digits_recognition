# Import necessary libraries
from sklearn import svm,datasets,metrics
from sklearn.model_selection import train_test_split


#read digits
# def read_digits():
#     digits = datasets.load_digits()
#     x = digits.images
#     y = digits.targetś
#     return x,y

#preprocess the data
def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x,y,test_size,random_state=1):
    X_train, X_test, Y_train, y_test = train_test_split(
     x,y, test_size=0.5, shuffle=False,random_state=random_state
    )
    return X_train,X_test,Y_train,y_test
# train the model of choice with the model params
def train_model(x,y,model_params,model_type="svm"):
    # Create a classifier: a support vector classifier
    if model_type=="svm":
        clf = svm.SVC
    model=clf(**model_params)
    #pdb.set_trace()
    # train the model
    model.fit(x,y)
    return model

def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
    # Split data into train and test subsets
    X_train, X_test, Y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Calculate the remaining size for the development set
    remaining_size = 1.0 - test_size
    dev_relative_size = dev_size / remaining_size
    
    # Split the train data into train and development subsets
    X_train_final, X_dev, Y_train_final, y_dev = train_test_split(
        X_train, Y_train, test_size=dev_relative_size, random_state=random_state
    )
    
    return X_train_final, X_dev, X_test, Y_train_final, y_dev, y_test

def predict_and_eval(model, X, y_true):
    predicted = model.predict(X)    
    return metrics.accuracy_score(y_true, predicted)


def tune_hyper_params(X_train, X_dev, X_test, Y_train, y_dev, y_test, list_of_all_param_combination):
    best_acc_so_far = -1
    best_model, optimal_gamma, optimal_C = None, None, None

    for gamma, C in list_of_all_param_combination:
        
        # Train the model with the current gamma and C
        model = train_model(X_train, Y_train, model_params={'gamma': gamma, 'C': C}, model_type="svm")
        
        # Get predictions on the development set
        y_predict = model.predict(X_dev)
        
        # Calculate accuracy on the development set
        current_accuracy = metrics.accuracy_score(y_dev, y_predict)
        
        # Select the hyperparameters that yield the best performance on the development set
        if current_accuracy > best_acc_so_far:
            best_acc_so_far = current_accuracy
            optimal_gamma = gamma
            optimal_C = C
            best_model = model
    
    # Print optimal hyperparameters
    print("Optimal parameters gamma:", optimal_gamma, "C:", optimal_C)
    
    # Get predictions on the test set using the best model
    y_test_predict = best_model.predict(X_test)
    
    # Evaluate and print test accuracy
    test_acc = metrics.accuracy_score(y_test, y_test_predict)
    print("Test accuracy:", test_acc)
    
    return best_model, optimal_gamma, optimal_C, test_acc



