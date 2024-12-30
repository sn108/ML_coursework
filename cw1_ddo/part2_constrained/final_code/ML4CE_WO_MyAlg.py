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
    import numpy as np
    from scipy.interpolate import SmoothBivariateSpline
    from scipy.optimize import minimize


    def spline_optimiser(
        fun, x0, bounds, constraints, max_iter, tol=1e-6, verbose=False
    ):
        """
        Algorithm using iterative quadratic spline interpolation with trust-region search.

        Parameters:
            fun : callable
                The objective function to minimize.
            x0 : ndarray
                Initial guess.
            bounds : sequence of (float, float)
                Bounds for variables. 
            constraints : sequence of dict
                Constraints definitions 
            max_iter : int
                Maximum number of iterations.
            tol : float
                Tolerance for stopping criteria.
            verbose : bool
                If True, print iteration details.

        Returns:
            res : OptimizeResult
                The optimization result represented as a `scipy.optimize.OptimizeResult`.
        """
        #n = len(x0)

        # Initialize points and function values
        X_init = np.vstack((np.array(x0), np.array([np.random.uniform(low, high) for (low, high) in bounds]).reshape(1, -1)))
        Y_init = np.array([fun(X_init[0, :]), fun(X_init[1, :])])
        f_best = min(Y_init)

        for iteration in range(max_iter - 9):
            # Ensure sufficient points for spline fitting, we have to use up 9 function evaluations
            if len(Y_init) < 9:  # Minimum required for quadratic spline
                random_points = np.array([
                    [np.random.uniform(low, high) for (low, high) in bounds]
                    for _ in range(9 - len(Y_init))
                ])
                random_values = np.array([fun(p) for p in random_points])
                X_init = np.vstack((X_init, random_points))
                Y_init = np.append(Y_init, random_values)

            # Fit a SmoothBivariateSpline to the sampled points
            spline = SmoothBivariateSpline(X_init[:, 0], X_init[:, 1], Y_init,  kx=2, ky=2, s=40.0)

            # Define spline objective
            def spline_objective(x):
                return spline(x[0], x[1])[0]  # Evaluate the spline at point x

            # Optimize the spline using scipy.optimize.minimize
            res = minimize(
                spline_objective,
                X_init[-1, :],  # Start from the last valid point
                method='trust-constr',
                bounds=bounds,
                constraints=constraints
            )    # we are optimising the spline model, not the actual model, hence why iterations have a limit of 50 here

            x_new = res.x  # Get the new candidate from the result

            # Ensure bounds are respected
            if bounds is not None:
                x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

            f_new = fun(x_new)

            # Update the sample size, and refit the spline model with the optimised data point
            X_init = np.vstack((X_init, np.array(x_new).reshape(1, -1)))
            Y_init = np.append(Y_init, np.array([f_new]).reshape(-1))
            x, f_best = x_new, f_new

            # Return result
            return {
                'x': x,
                'fun': f_best,
                #'nit': iteration,
                #'success': True,
                #'message': 'Optimization terminated successfully.'
            }


    constraints = [
        {'type': 'ineq', 'fun': con1},
        {'type': 'ineq', 'fun': con2}
    ]

    # Run quad spline interpolation that uses a trust-region search
    opt = spline_optimiser(obj, x_start, bounds, constraints, budget, verbose=False)
    #print("Optimization Result:", opt)

    ##############################
    ###### Your Code END #########
    ##############################

    ################# ADJUST THESE ######################
    team_names = ['John Adeyemi', 'Bellal Ahadi', 'Stutya Nagpal', 'Shiam Srikumar']
    cids = ['02093393', 'please insert your CIDS']
    return team_names,cids
