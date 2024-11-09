def YourAlg(
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



    class Bayesian_Guassian :
        def __init__(self, kernel, kernel_pred, noise=1e-10):
            self.kernel = kernel    # kernel function enables us to measure correlation between data points
            self.kernel_predict = kernel_pred
            self.noise = noise      # noise level inputted to capture uncertainty 

        def fit(self, x, y):        # feed input data x and output observed values y here to enable GP to measure correlation/co-variance
            self.X_train = np.array(x).reshape(-1,2)
            self.y_train = np.array(y).reshape(-1,1)
            self.K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
            self.K_inv = np.linalg.inv(self.K)
    
        def predict(self, X):
            K_s = self.kernel_predict(self.X_train, np.array(X).reshape(-1,2))
            K_ss = self.kernel_predict(np.array(X), np.array(X)) + self.noise
            mu_s = K_s.T.dot(self.K_inv).dot(self.y_train)
            cov_s = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
            return mu_s, cov_s
        
    def gaussian_kernel_matrix(X, sigma=1):
        distances = np.sum((X[:, np.newaxis] - X) ** 2, axis=-1)
        kernel_matrix = np.exp(-distances / (2 * sigma ** 2))
    
        return kernel_matrix
    
    def gaussian_kernel_vector(X_sample, X, sigma=1):
        if X.shape[0] == 1 :
            X=np.tile(X, (X_sample.shape[0],1))
            distance = np.sum((X_sample-X) ** 2, axis = -1)
            kernel_matrix_predict = np.exp(-distance / (2 * sigma ** 2)).reshape(-1,1)
        else :
            distance = np.sum((X_sample-X) ** 2, axis = -1)
            kernel_matrix_predict = np.exp(-distance / (2 * sigma ** 2))
        
        return kernel_matrix_predict


    def expected_improvement(X, X_sample, Y_sample, gp, xi=0.01):
        mu, sigma = gp.predict(X)
        mu_sample = gp.predict(X_sample)[0]
        sigma = np.sqrt(np.diag(sigma))

        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    
    def propose_location(acquisition, X_sample, Y_sample, gp, bounds, n_restarts=25):
        dim = X_sample.shape[0]
        min_val = 1
        min_x = None

        def min_obj(X):
            return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gp)
        
        starting_points = np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n_restarts)])


        for x0 in starting_points:
            # Constraints for the optimizer
            cons = [{'type': 'ineq', 'fun': con1},
                    {'type': 'ineq', 'fun': con2}]
            
            res = minimize(min_obj, x0=x0.flatten(), bounds=bounds, constraints=cons, method='L-BFGS-B')
            
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x

        return min_x.reshape(-1, 1)

    # Initial samples
    X_init = np.vstack((np.array(x_start),np.array([np.random.uniform(low, high) for (low, high) in bounds]).reshape(2,)))
    Y_init = np.array([obj(X_init[0,:]),obj(X_init[1,:])])

    # Bounds for the search space
    bounds = bounds

    # Gaussian Process
    gp = Bayesian_Guassian(kernel=gaussian_kernel_matrix, kernel_pred=gaussian_kernel_vector)
    gp.fit(X_init, Y_init)

    # Bayesian Optimization loop
    n_iter = budget-1
    X_sample = X_init
    Y_sample = Y_init

    for i in range(n_iter):
        X_next = propose_location(expected_improvement, X_sample, Y_sample, gp, bounds)
        Y_next = obj(np.array(X_next))
        X_sample = np.vstack((X_sample, np.array(X_next).reshape(1,-1)))
        Y_sample = np.append(Y_sample, np.array([Y_next]).reshape(-1))
        gp.fit(X_sample, Y_sample)
        print(f"Iteration {i+1}: X_next = {X_next.flatten()}, Y_next = {Y_next}")

        print(f"Best location: {X_sample[np.argmin(Y_sample)]}, Best value: {np.min(Y_sample)}")
    ##############################
    ###### Your Code END #########
    ##############################

    ################# ADJUST THESE ######################
    team_names = ['Member1', 'Member2']
    cids = ['01234567', '01234567']
    return team_names,cids