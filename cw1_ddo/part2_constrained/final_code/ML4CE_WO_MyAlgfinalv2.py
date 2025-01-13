def YourAlgv2(
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
        fun, x0, bounds, con1, con2, budget):
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

        Returns:
            res : OptimizeResult
                The optimization result`.
        """
        
      
        # Initialize points and function values
        X_init = np.array([x0] + [[np.random.uniform(low, high) for low, high in bounds] for _ in range(3)])
        Y_init = np.array([fun(x) for x in X_init])
        con_surr1 = np.array([con1(x) for x in X_init])
        con_surr2 = np.array([con2(x) for x in X_init])

        for iteration in range(budget-4):
            # Determine spline degree based on the number of points
            n_points = len(Y_init)
            if n_points < 9:
                kx, ky = 1, 1  # Linear spline
            elif n_points < 16:
                kx, ky = 2, 2  # Quadratic spline
            else:
                kx, ky = 3, 3  # Cubic spline

            # Index of the minimum value in Y_init
            min_index = np.argmin(Y_init)

            # Corresponding x_value in X_init
            x = X_init[min_index, :]
            # Fit splines to the data
            spline = SmoothBivariateSpline(X_init[:, 0], X_init[:, 1], Y_init, kx=kx, ky=ky, s=5000.0)
            spline_con1 = SmoothBivariateSpline(X_init[:, 0], X_init[:, 1], con_surr1, kx=kx, ky=ky, s=40.0)
            spline_con2 = SmoothBivariateSpline(X_init[:, 0], X_init[:, 1], con_surr2, kx=kx, ky=ky, s=40.0)

            # Define spline objective and constraints
            def spline_objective(x):
                return spline(x[0], x[1])[0]

            def con1_surr(x):
                return spline_con1(x[0], x[1])[0]

            def con2_surr(x):
                return spline_con2(x[0], x[1])[0]

            constraints = [
                {'type': 'ineq', 'fun': con1_surr},
                {'type': 'ineq', 'fun': con2_surr}
            ]

            # Optimize the spline model
            res = minimize(
                spline_objective,
                x,  # Start from the last valid point
                method='trust-constr',
                bounds=bounds,
                constraints=constraints
            )

            x_new = res.x  # New candidate point
            f_new = fun(x_new)  # Evaluate the true function

            # Update samples and refit splines
            X_init = np.vstack((X_init, x_new))
            Y_init = np.append(Y_init, f_new)
            con_surr1 = np.append(con_surr1, con1(x_new))
            con_surr2 = np.append(con_surr2, con2(x_new))
            x, f_best = x_new, f_new
            # Break if budget is exhausted
            if iteration + 1 >= budget-3:
                break

        # Return the best result
        return {'x': x, 'fun': f_best}

    # Run the optimization
    opt = spline_optimiser(obj, x_start, bounds, con1, con2, budget)

    ##############################
    ###### Your Code END #########
    ##############################

    ################# ADJUST THESE ######################
    team_names = ['John Adeyemi', 'Bellal Ahadi', 'Stutya Nagpal', 'Shiam Srikumar']
    cids = ['02093393', 'please insert your CIDS']
    return team_names,cids