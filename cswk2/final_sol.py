import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def fit_preprocess(data_path):
    """
    Return a single object representing the parameters used to preprocess the
    data.

    Here we fit a standard scalar to our data that can be used for standardisation

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.

    Returns
    -------
    preprocess_params : dict
        Dictionary containing the parameters used to preprocess the data.
    """
    data = pd.read_csv(data_path)
    X = data.drop(columns=['label']).values
    scaler = StandardScaler()
    scaler.fit(X)
    return {'scaler': scaler}

def load_and_preprocess(data_path, preprocess_params):
    """
    This function loads and preprocesses data from a csv file.

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
    data = pd.read_csv(data_path)
    X = data.drop(columns=['label']).values
    y = data['label'].values
    scaler = preprocess_params['scaler']
    X_scaled = scaler.transform(X)
    return X_scaled, y

def fit_model(X):
    """
    Returns the fitted model. This is a PCA model which calculates Mahalanobis distances based off the reconstruction errors to determine anomalies.

    Parameters
    ----------
    X : 2D numpy array.
        Already preprocessed data used to fit the model.

    Returns
    -------
    model : object
        Fitted model.
    """
    pca = PCA(n_components=0.99)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    recon_error = X - X_reconstructed

    mean_error = np.mean(recon_error, axis=0)
    cov_error = np.cov(recon_error, rowvar=False)

    # Diagonal loading for stability
    lambda_ = 1e-5
    cov_error += lambda_ * np.eye(cov_error.shape[0])
    inv_cov_error = np.linalg.inv(cov_error)

    # Calculate Mahalanobis distances based on reconstruction errors
    distances = np.array([np.sqrt((e - mean_error).dot(inv_cov_error).dot((e - mean_error).T)) for e in recon_error])

    # Approximate y_true for threshold tuning
    percentile = 97  # Define the approximate anomaly percentile
    approximate_y_true = (distances > np.percentile(distances, percentile)).astype(int)

    # Search for threshold that gives target accuracy
    candidate_thresholds = np.linspace(min(distances), max(distances), 1000)
    best_threshold = None
    for threshold in candidate_thresholds:
        y_pred = (distances > threshold).astype(int)
        accuracy = accuracy_score(approximate_y_true, y_pred)
        if accuracy >= 0.97:
            best_threshold = threshold
            break
    model = {
        'pca': pca,
        'mean_error': mean_error,
        'inv_cov_error': inv_cov_error,
        'threshold': best_threshold
    }

    return model

def predict(X, model):
    """
    Accepts a 2D numpy array and the anomaly detection model, that returns a 1D
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
    pca = model['pca']
    mean_error = model['mean_error']
    inv_cov_error = model['inv_cov_error']
    threshold = model['threshold']

    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    recon_error = X - X_reconstructed

    distances = np.array([np.sqrt((e - mean_error).dot(inv_cov_error).dot((e - mean_error).T)) for e in recon_error])
    y_pred = (distances > threshold).astype(int)

    return y_pred


def main():
    """
    Main function to test the anomaly detection pipeline.
    """
    train_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv'
    test_path = 'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_test.csv'

    # Fit preprocessing
    preprocess_params = fit_preprocess(train_path)

    # Load and preprocess data
    X_train, y_train = load_and_preprocess(train_path, preprocess_params)
    X_test_original, y_test_original = load_and_preprocess(test_path, preprocess_params)

    # Split 20% of training data and add it to testing data
    #X_train, X_test_extra, y_train, y_test_extra = train_test_split(X_train, y_train, test_size=0.2)
    #X_test = np.vstack((X_test_original, X_test_extra))
    #y_test = np.hstack((y_test_original, y_test_extra))

    # Train model
    model = fit_model(X_train)

    # Predict and evaluate
    y_pred_train = predict(X_train, model)
    y_pred_test = predict(X_test_original, model)

    #test_loss = calculate_binary_crossentropy_loss(X_test_original, y_test_original, model)


    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test_original, y_pred_test)

    print('Training accuracy:', train_accuracy)
    print('Test accuracy:', test_accuracy)
    #print('Test loss (Binary Cross-Entropy):', test_loss)

if __name__ == '__main__':
    main()
