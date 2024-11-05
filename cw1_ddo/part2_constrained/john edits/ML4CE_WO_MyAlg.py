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
        def __init__(self, kernel, noise=1e-10):
            self.kernel = kernel    # kernel function enables us to measure correlation between data points
            self.noise = noise      # noise level inputted to capture uncertainty 

        def fit(self, x, y):        # feed input data x and output observed values y here to enable GP to measure correlation/co-variance
            self.X_train = np.array(x)
            self.y_train = np.array(y)
            self.K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
            self.K_inv = np.linalg.inv(self.K)
    
        def predict(self, X):
            K_s = self.kernel(self.X_train, np.array(X))
            K_ss = self.kernel(np.array(X), np.array(X)) + self.noise * np.eye(len(X))
            mu_s = K_s.T.dot(self.K_inv).dot(self.y_train)
            cov_s = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
            return mu_s, cov_s
        
    def rbf_kernel(X1, X2, length_scale=1.0):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / length_scale * sqdist)


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
        dim = X_sample.shape[1]
        min_val = 1
        min_x = None

        def min_obj(X):
            return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gp)

        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x

        return min_x.reshape(-1, 1)

    # Initial samples
    X_init = np.array(x_start)
    Y_init = np.array(obj(X_init.reshape(2,)))

    # Bounds for the search space
    bounds = bounds

    # Gaussian Process
    gp = Bayesian_Guassian(kernel=rbf_kernel)
    gp.fit(X_init, Y_init)

    # Bayesian Optimization loop
    n_iter = budget
    X_sample = X_init
    Y_sample = Y_init

    for i in range(n_iter):
        X_next = propose_location(expected_improvement, X_sample, Y_sample, gp, bounds)
        if con1(np.array(X_next).reshape(2,)) <= 0.12 and con2(np.array(X_next.reshape(2,))) <= 0.08:  # Check the constraint
            Y_next = obj(np.array(X_next.reshape(2,)))
            X_sample = np.vstack((X_sample, np.array(X_next.reshape(2,))))
            Y_sample = np.append(Y_sample, Y_next)
            gp.fit(X_sample, Y_sample)
            print(f"Iteration {i+1}: X_next = {X_next.flatten()}, Y_next = {Y_next}")
        else:
            print(f"Iteration {i+1}: X_next = {X_next.flatten()} does not satisfy the constraint")
            xi=xi-0.001

        print(f"Best location: {X_sample[np.argmin(Y_sample)]}, Best value: {np.min(Y_sample)}")
    ##############################
    ###### Your Code END #########
    ##############################

    ################# ADJUST THESE ######################
    team_names = ['Member1', 'Member2']
    cids = ['01234567', '01234567']
    return team_names,cids