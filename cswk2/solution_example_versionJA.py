"""
This is an example of how to submit the coursework. This file MUST contain:

  - fit_preprocess(data_path): A function that takes a single parameter, the 
  path to the CSV file containing the data. It returns an object (e.g., a 
  dictionary) with parameters used to preprocess the data.
  - load_and_preprocess(data_path, preprocess_params): A function with two 
  parameters: the path to the CSV file containing the data and the 
  preprocessing parameters object returned by fit_preprocess. It returns two 
  objects: the preprocessed data and the labels, which can be in the form of 
  numpy arrays, pandas DataFrames, or Python lists.
  - fit_model(X): A function that takes the preprocessed data as a single 
  parameter and returns a trained model, which will be used by predict. 
  This model is any object that predict requires to perform the detection task.
  - predict(X, model): A function that returns the anomaly detection prediction 
  for the input data, X,  based on the provided model.

Do not change the name of "fit_preprocess", "load_and_preprocess", "fit_model"
and "predict" functions.

Separating the code into several functions is based on the idea of running
the training of the model only once, and make predictions several times with
the already trained model.

You can include any other code that is useful for the coursework. For instance,
here `main()` is included for us to check that predict and model run properly,
and to show some results of the performance of the model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def fit_preprocess(data_path):
    """
    Return a single object representing the parameters used to preprocess the
    data.
    In this example, we return a dictionary with the mean and standard
    deviation for standardization.

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.

    Returns
    -------
    preprocess_params : dict
        Dictionary containing the parameters used to preprocess the data.
    """
    # Load from file
    data = pd.read_csv(data_path)
    # Obtain preprocess parameters
    X = data.to_numpy()     # (JA) Here it is only using one measured variable as an anomaly detector, huge loss of information (There are 52 features)
    mean = X.mean()                     # (JA) Extracting the mean and s.d. to help standardise data, preventing biased results, some potential to do some data exploration here to decide what features to keep, to help determine parameters
    std = X.std()                       # (JA) Could also just keep all features and use a very efficient variable reduction method
    preprocess_params = {'mean': mean, 'std': std}
    return preprocess_params


def load_and_preprocess(data_path, preprocess_params):
    """
    This function loads and preprocesses data from a csv file
    In this example, we read the first feature as the variable of interest,
    then standardize the data.

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.
    preprocess_params : dict
        Dictionary containing the parameters used to preprocess the data.

    Returns
    -------
    X : 2D numpy array
        Preprocessed data. Make sure X is of shape: (n_instances, n_features).
    y : 1D numpy array
        True normal/anomaly labels of the data.
    """
    # Load from file
    data = pd.read_csv(data_path)
    # Separate label and choose features, then reshape to 2D array
    indx = np.r_[1:6, 11, 13:15, 16, 25, 31, 35:37, 39:43, 47:49, 51]
    X = data.iloc[:, indx].to_numpy()     #.iloc[:,1:6:11]
    y = data['label'].to_numpy()
    # Preprocess data
    mean = preprocess_params['mean']        # (JA) once the data is extracted here we actually pre-process which is simply standardising the data
    std = preprocess_params['std']
    X = (X - mean) / std                    # standardisation applied here
    return X, y                            # (JA) Returns the pprocessed features (X) and the points we are trying to predict (y), when we train our model
                                           # additional things we can add to the model is diensitonality reduction, to process the data (try PCA (up to slide 100 is the actual dimension reduction bit)


def fit_model(X):
    """
    Returns the fitted model. In this simple example, the anomaly detection
    model is the threshold that will flag anomalies, but it can be any object
    as long as the predict() function knows how to handle it when passed as an
    argument.

    Parameters
    ----------
    X : 2D numpy array.
        Already preprocessed data used to fit the model.

    Returns
    -------
    model : object
        Fitted model.
    """
    # Step 1: Compute mean and covariance matrix
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    
    # Regularize covariance matrix to prevent singularity
    cov += np.eye(cov.shape[0]) * 1e-6
    
    # Step 2: Compute Mahalanobis distance for each point
    diff = X - mean
    inv_cov = np.linalg.inv(cov)
    mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    
    
    # Step 4: Determine the threshold at 68% significance
    threshold = np.percentile(mahalanobis_dist, 68)
    
    return threshold

    

def predict(X, model):
    """
    Accepts a 2D numpy array and an anomaly detection model, and returns a 1D
    array of predictions.

    Parameters
    ----------
    X : 2D numpy array.
        Preprocessed data.
    model : object
        Anomaly detection model.

    Returns
    -------
    y_pred : 1D numpy array.
        Array of normal/anomaly predictions.
    """
    # Unpack the model
    threshold = model
    
    # Step 1: Compute mean and covariance matrix for X
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Ensure covariance matrix is 2D
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])  # Convert scalar to 2D array
    elif np.ndim(cov) == 1:
        cov = np.diag(cov)  # Convert 1D variance to diagonal covariance matrix
    
    # Regularize covariance matrix to prevent singularity
    cov += np.eye(cov.shape[0]) * 1e-6
    
    # Step 2: Compute Mahalanobis distance for each point
    diff = X - mean
    inv_cov = np.linalg.inv(cov)
    mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    
    # Step 3: Classify based on the threshold
    y_pred = (mahalanobis_dist > threshold).astype(int)
    
    return y_pred


def main():
    """
    Main function to test the code when running this script standalone. Not
    mandatory for the deliverable but useful for the grader to check the code
    workflow and for the student to check it works as expected.
    """
    train_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv'
    test_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_test.csv'

    # Set up preprocessing
    preprocess_params = fit_preprocess(train_path)

    # Load and preprocess data
    X_train, y_train = load_and_preprocess(train_path, preprocess_params)
    X_test, y_test = load_and_preprocess(test_path, preprocess_params)

    # Get the detection model
    threshold = fit_model(X_train)

    # Check performance on data
    # (just for us to see it works properly, the grader might use other data)
    y_pred_train = predict(X_train, threshold)
    print('Training accuracy: ', accuracy_score(y_train, y_pred_train))
    print('f1_score: ', f1_score(y_train, y_pred_train))
    print('precision score: ', precision_score(y_train, y_pred_train))
    print('recall score: ', recall_score(y_train, y_pred_train))
    y_pred_test = predict(X_test, threshold)
    print('Test accuracy: ', accuracy_score(y_test, y_pred_test))
    print('f1_score: ', f1_score(y_test, y_pred_test))
    print('precision score: ', precision_score(y_test, y_pred_test))
    print('recall score: ', recall_score(y_test, y_pred_test))
    


if __name__ == '__main__':
    main()
    