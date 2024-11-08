def MyAlg(
        problem,
        bounds,  
        budget, 
        rep
        ): 
    
    '''
    - problem: 
        data-type: Test_function object
        This is the WO problem
    - bounds: 
        data-type: np.array([[4., 7.], [70., 100.]])
        Flowrate reactand B from 4 to 7 kg/s
        Reactor temperature from 70 to 100 degree Celsius
    - budget: 
        data-type: integer
        budget for WO problem: 20 iterations
    - rep:
        data-type: integer
        repetition to determine the starting point
    '''
    
    ######### DO NOT CHANGE ANYTHING FROM HERE ############
    x_start = problem.x0[rep].flatten() # numpy nd.array for example [ 6.9 83. ]
    obj = problem.fun_test              # objective function: obj = -(1043.38 * flow_product_R + 20.92 * flow_product_E -79.23 * inlet_flow_reactant_A - 118.34 * inlet_flow_reactant_B) 
    con1 = problem.WO_con1_test         # constraint 1: X_A <= 0.12
    con2 = problem.WO_con2_test         # constraint 2: X_G <= 0.08
    ######### TO HERE ####################################
    

    '''
    This is where your algorithm comes in.
    Use the objective functions and constraints as follows:
    x_start: this is where your algorithm starts from. Needs to be included, see below for an example
    obj_value = obj(input) with input as numpy.ndarray of shape (2,) and obj_value as float
    con1_value = con1(input) see obj
    con2_value = con2(value) see obj
    '''
    ##############################
    ###### Your Code START #######
    ##############################
    from scipy.stats import norm
    from scipy.optimize import minimize
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Gaussian Process Regressor
    class GaussianProcessRegressor:
        def __init__(self, kernel, alpha=1e-10):
            self.kernel = kernel
            self.alpha = alpha
            self.is_single_sample = True  # Track if we have only one training point

        def fit(self, X_train, y_train):
            # Ensure X_train and y_train are 2D arrays
            X_train = np.atleast_2d(X_train).reshape(-1,2)
            y_train = np.atleast_2d(y_train).reshape(-1, 1)  # Reshape to (n_samples, 1)
            self.X_train = X_train
            self.y_train = y_train
            self.K = self.kernel(X_train) + self.alpha * np.eye(len(X_train))
            # If we have a single sample, invert directly as a scalar, which occurs if variance=1 which occurs if single sample (even if more than 1 feature: (1,n))
            if self.K.shape == (1, 1):
                self.K_inv = 1.0 / self.K[0, 0]  # Inverse of a 1x1 matrix is just 1 / element
                self.alpha_ = self.K_inv * y_train  # Scalar multiplication
                self.is_single_sample = True
            else:
                # Otherwise, use the pseudo-inverse for multiple samples
                self.K_inv = np.linalg.pinv(self.K)
                self.alpha_ = self.K_inv @ y_train
                self.is_single_sample = False

        def predict(self, X_test, return_std=False):
             # Ensure X_test is a 2D array
            X_test = np.atleast_2d(X_test)
            K_trans = self.kernel(X_test)
            if self.is_single_sample:
                y_mean = K_trans * self.alpha_
            else:
                y_mean = K_trans @ self.alpha_

            if return_std:
                # Compute variance: for a single sample, kernel variance is simpler
                if self.is_single_sample:
                    K_test = self.kernel(X_test)
                    y_var = K_test - (K_trans * self.K_inv * K_trans)
                else:
                    # For multiple samples, use matrix operations
                    K_test = self.kernel(X_test, X_test)
                    y_var = K_test - K_trans @ self.K_inv @ K_trans.T
                return y_mean, np.sqrt(np.diag(y_var))
            else:
                return y_mean

    # Define the RBF kernel
    def rbf_kernel(X, length_scale=1.0, variance=1.0):
        """
        RBF Kernel for a single sample with multiple features.

        Parameters:
        - X: array-like of shape (1, n_features), single input sample
        - length_scale: float, length scale of the kernel
        - variance: float, variance of the kernel

        Returns:
        - K: array-like of shape (1, 1), the RBF kernel value
        """
        # Ensure X is a 2D array with a single row
        X = np.atleast_2d(X)
        
        if X.shape[0] == 1:
            # Only one sample, return the variance as the self-similarity
            return np.array([[variance]])
        else:
            # If there are multiple samples, calculate pairwise similarities
            sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
            return variance * np.exp(-0.5 / length_scale**2 * sqdist)

    # Define the acquisition function (Expected Improvement)
    def acquisition_function(x, gp, y_max):
        x=np.array(x).reshape(-1,2)
        mu, sigma = gp.predict(x, return_std=True)
        with np.errstate(divide='warn'):
            imp = mu - y_max
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei

    # Define the optimization function
    def optimize_acquisition(gp, y_max):
        # Initial guess for the optimizer
        x0 = np.array([np.random.uniform(low, high) for (low, high) in bounds]).reshape(-1,2)
        
        # Bounds for the optimizer
        bound = bounds
        
        # Constraints for the optimizer
        cons = [{'type': 'ineq', 'fun': con1},
                {'type': 'ineq', 'fun': con2}]
        
        # Minimize the negative acquisition function
        res = minimize(lambda x: -acquisition_function(x, gp, y_max), x0=x0.flatten(), bounds=bound, constraints=cons)
        
        return res.x.reshape(1,-1) if res.success else None

    X_sample = np.array([[x_start[0], x_start[1]]])  # Starting point
    Y_sample = np.array([[obj(x_start)]])           # Initial output sample
        
    # Create the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=rbf_kernel, alpha=1e-2)

    # Fit the Gaussian Process model to the initial samples
    gp.fit(X_sample, Y_sample)

    
    # Perform Bayesian Optimization
    n_iterations = budget
    for i in range(n_iterations):
        # Find the next point to sample
        x_next = optimize_acquisition(gp, Y_sample.max())
        
        # Sample the black-box function at the new point
        y_next = obj(x_next)
        
        # Add the new sample to the existing samples
        X_sample = np.vstack((X_sample, x_next.reshape(1, -1)))
        Y_sample = np.vstack((Y_sample, [[y_next]]))
        
        # Update the Gaussian Process model with the new samples
        gp.fit(X_sample, Y_sample)

    # Print the best found value and its location
    best_index = np.argmax(Y_sample)
    print(f"Best value found: {Y_sample[best_index][0]} at location {X_sample[best_index]}")

    ##############################
    ###### Your Code END #########
    ##############################

    ################# ADJUST THESE ######################
    team_names = ['Member1', 'Member2']
    cids = ['01234567', '01234567']
    return team_names,cids