import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# don't forget to adapt this to numpy (i.e. send data to numpy)
def fit_preprocess(data_path):
    """
    Loads the data and returns preprocessing parameters (scaler).

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.

    Returns
    -------
    preprocess_params : dict
        Dictionary containing the fitted MinMaxScaler.
    """
    # Load data
    data = pd.read_csv(data_path)
    # Separate features
    #X = data.drop(columns=['label', 'XMEAS(5)', 'XMEAS(12)', 'XMEAS(14)', 'XMEAS(17)', 'XMEAS(26)', 'XMEAS(37)', 'XMEAS(40)', 'XMEAS(41)', 'XMV(4)', 'XMV(7)', 'XMV(11)']).values
    X = data.drop(columns=['label', 'XMEAS(40)', 'XMEAS(41)', 'XMV(4)', 'XMV(7)', 'XMV(11)']).values 
    # Fit scaler
    scaler = MinMaxScaler()
    scaler.fit(X)

    preprocess_params = {'scaler': scaler}
    return preprocess_params


def load_and_preprocess(data_path, preprocess_params):
    """
    Loads and preprocesses the data using the provided scaler.

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.
    preprocess_params : dict
        Dictionary containing the parameters used to preprocess the data.

    Returns
    -------
    X : 2D numpy array
        Preprocessed data.
    y : 1D numpy array
        True normal/anomaly labels of the data.
    """
    # Load data
    data = pd.read_csv(data_path)
    # Separate features and labels
    #X = data.drop(columns=['label', 'XMEAS(5)', 'XMEAS(12)', 'XMEAS(14)', 'XMEAS(17)', 'XMEAS(26)', 'XMEAS(37)', 'XMEAS(40)', 'XMEAS(41)', 'XMV(4)', 'XMV(7)', 'XMV(11)']).values
    X = data.drop(columns=['label', 'XMEAS(40)', 'XMEAS(41)', 'XMV(4)', 'XMV(7)', 'XMV(11)']).values 
    y = data['label'].values

    # Preprocess features
    scaler = preprocess_params['scaler']
    X_scaled = scaler.transform(X)

    return X_scaled, y


def fit_model(X):
    """
    Trains PCA for dimensionality reduction and determines a clustering-based threshold on reconstruction error.

    Parameters
    ----------
    X : 2D numpy array
        Preprocessed training data.

    Returns
    -------
    model : object
        Dictionary containing the trained PCA model and clustering threshold.
    """
    # Apply PCA
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
    X_pca = pca.fit_transform(X)

    # Reconstruct data back to original space
    X_reconstructed = pca.inverse_transform(X_pca)

    # Calculate reconstruction errors
    reconstruction_errors = np.mean((X - X_reconstructed)**2, axis=1).reshape(-1, 1)

    # Apply KMeans clustering on reconstruction errors
    kmeans = KMeans(n_clusters=2, random_state=42).fit(reconstruction_errors)

    # Identify the cluster with the lower mean reconstruction error as normal
    cluster_centers = kmeans.cluster_centers_.flatten()
    normal_cluster = np.argmin(cluster_centers)

    # Set threshold as the maximum error in the normal cluster
    threshold = np.max(reconstruction_errors[kmeans.labels_ == normal_cluster])

    return {'pca': pca, 'threshold': threshold}


def predict(X, model):
    """
    Predicts anomalies using the trained PCA model and reconstruction error threshold.

    Parameters
    ----------
    X : 2D numpy array
        Preprocessed data.
    model : object
        Dictionary containing the trained PCA model and threshold.

    Returns
    -------
    y_pred : 1D numpy array
        Array of anomaly predictions (0 for normal, 1 for anomaly).
    """
    pca = model['pca']
    threshold = model['threshold']

    # Transform and reconstruct data
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)

    # Calculate reconstruction errors
    reconstruction_errors = np.mean((X - X_reconstructed)**2, axis=1)

    # Predict anomalies based on the threshold
    y_pred = (reconstruction_errors > threshold).astype(int)

    return y_pred

def calculate_binary_crossentropy_loss(X, y, model):
    """
    Calculates the binary cross-entropy loss for the autoencoder.

    Parameters
    ----------
    X : 2D numpy array
        Preprocessed data.
    y : 1D numpy array
        True normal/anomaly labels of the data.
    model : object
        Dictionary containing the trained autoencoder.

    Returns
    -------
    loss : float
        Binary cross-entropy loss.
    """
    # Transform and reconstruct data
    pca = model['pca']
    X_pca = pca.transform(X)
    reconstructions = pca.inverse_transform(X_pca)

    # Compute reconstruction errors as probabilities
    reconstruction_errors = np.mean((X - reconstructions)**2, axis=1)
    probabilities = 1 - (reconstruction_errors / (reconstruction_errors.max() + 1e-10))

    # Compute binary cross-entropy loss
    loss = log_loss(y, probabilities)
    return loss


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