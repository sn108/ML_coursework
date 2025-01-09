from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import time as clock
import math
import random 

##############################
# Example Exploration Scheme #
##############################
####################################
# Please modify the function below #
####################################
def explorer(x_t: np.array, u_bounds: dict, timestep: int, model = None, uncertainty_threshold = 0.1) -> np.array:
    '''
    Function to collect more data to train the model.
    x_t (np.array) - Current state 
    u_bounds (dict) - Bounds on control inputs
    timestep (int) - Current timestep
    
    Output:
    u_plus - Next control input
    '''

    u_lower = u_bounds['low']
    u_upper = u_bounds['high']

    dimension = len(u_lower)
    max_samples = 1000  # Limit for Sobol-like sequence
    idx = timestep % max_samples
    
    # Generate Sobol-like sequences using Van der Corput
    def van_der_corput_sequence(n, base):
        sequence = []
        for i in range(n):
            number = 0
            denominator = 1
            while i > 0:
                denominator *= base
                number += (i % base) / denominator
                i //= base
            sequence.append(number)
        return np.array(sequence)

    sobol_like = np.zeros(dimension)

    for d in range(dimension):
        sequence = van_der_corput_sequence(max_samples, base=2 + d)
        sobol_like[d] = sequence[idx]

    u_candidate = u_lower + sobol_like * (u_upper - u_lower)

    if model is not None:
        # Predict the next state using the current model
        x_candidate = np.hstack([x_t, u_candidate]).reshape(1, -1)
        y_pred = model.predict(x_candidate)

        # Calculate uncertainty (e.g., variance of predictions)
        uncertainty = np.std(y_pred)

        # Adjust control input based on uncertainty threshold
        if uncertainty < uncertainty_threshold:
            print("Low uncertainty, exploring new region.")
            u_candidate = u_lower + np.random.uniform(0, 1, size=u_lower.shape) * (u_upper - u_lower)

    return u_candidate

    #u_plus = np.random.uniform(u_lower, u_upper, size=u_lower.shape)
    
    #return u_plus
    

########################
#Example Model Training#
########################
####################################
# Please modify the function below #
####################################
def model_trainer(data: tuple, env: callable) -> callable:
    """
    Trains a predictive model using a neural network with regularization and data augmentation.
    Parameters:
    data (tuple): A tuple containing `data_states` and `data_controls` arrays.
    env (callable): An environment object containing `o_space` and `a_space` bounds.

    Returns:
    callable: Trained predictive model.
    """
    data_states, data_controls = data

    # Select relevant state dimensions (1 and 8) and normalize features
    selected_states = data_states[:, [1, 8], :]
    o_low, o_high = env.env_params['o_space']['low'][[1, 8]], env.env_params['o_space']['high'][[1, 8]]
    a_low, a_high = env.env_params['a_space']['low'], env.env_params['a_space']['high']

    selected_states_norm = (selected_states - o_low.reshape(1, -1, 1)) / (o_high.reshape(1, -1, 1) - o_low.reshape(1, -1, 1)) * 2 - 1
    data_controls_norm = (data_controls - a_low.reshape(1, -1, 1)) / (a_high.reshape(1, -1, 1)) * 2 - 1

    # Reshape data for training
    reps, states, n_steps = selected_states_norm.shape
    _, controls, _ = data_controls_norm.shape

    X_states = selected_states_norm[:, :, :-1].reshape(-1, states)
    X_states_sq = X_states**2
    X_states_cub = X_states**3
    cross_linear = np.prod(X_states,axis=1,keepdims=True)
    cross_quad_1 = (X_states[:, 0] ** 2 * X_states[:, 1]).reshape(-1, 1)
    cross_quad_2 = (X_states[:, 1] ** 2 * X_states[:, 0]).reshape(-1, 1)
    X_states = np.concatenate((X_states,cross_linear,X_states_sq,cross_quad_1,cross_quad_2,X_states_cub,1/X_states),axis =1)
    X_controls = data_controls_norm[:, :, :-1].reshape(-1, controls)
    X = np.hstack([X_states, X_controls])  # Combine state and control inputs
    y = selected_states_norm[:, :, 1:].reshape(-1, states)

    # Data Augmentation: Add small noise to inputs
    noise_factor = 0.01
    X_augmented = X + np.random.normal(0, noise_factor, X.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Lasso(alpha=0.003, max_iter=10000)
    
    # Neural Network with Regularization
    #mlp = MLPRegressor(
        #hidden_layer_sizes=(64, 64),       # Reduced complexity for better generalization
        #activation='relu',
        #solver='adam',
        #max_iter=1000,
        #learning_rate_init=0.001,
        #early_stopping=True,
        #validation_fraction=0.1,
        #alpha=0.001,                         # L2 regularization to reduce overfitting
        #n_iter_no_change=20,
    #)

    #model = Pipeline([
    #('scaler', StandardScaler()),  # Scale data
    #('lasso', lasso),              # Feature selection
    #('mlp', mlp)                   # Neural network
    #])
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Training Completed - MSE: {mse:.4f}, R2: {r2:.3f}")

    return model

def active_data_collection(env, u_bounds, num_timesteps=500, retrain_interval=50):
    """
    Perform active data collection for model predictive control using the explorer function.
    Parameters:
    env (callable): Environment or system simulator to interact with.
    u_bounds (dict): Control input bounds.
    num_timesteps (int): Total number of timesteps for data collection.
    retrain_interval (int): Interval for retraining the model.

    Returns:
    callable: Trained model.
    """
    # Initialize an empty dataset
    X_data = []
    y_data = []

    # Initialize the model
    model = None

    # Initialize state
    x_t = env.reset()  # Assume the environment has a reset method to provide an initial state

    for timestep in range(num_timesteps):
        # Use the explorer function to collect a control input
        u_candidate = explorer(x_t, u_bounds, timestep, model=model, uncertainty_threshold=0.1)

        # Simulate the system or get the next state from the environment
        x_next, y_next = env.step(u_candidate)  # Example environment step method returning state and output

        # Append to the dataset
        X_data.append(np.hstack([x_t, u_candidate]))
        y_data.append(y_next)

        # Update the current state
        x_t = x_next

        # Periodically retrain the model with new data
        if timestep % retrain_interval == 0 and len(X_data) > 0:
            data_states = np.array(X_data)[:, :x_t.shape[0]].reshape(-1, x_t.shape[0], 1)
            data_controls = np.array(X_data)[:, x_t.shape[0]:].reshape(-1, u_candidate.shape[0], 1)
            model = model_trainer((data_states, data_controls), env, u_bounds, timestep, model)

            print(f"Retrained model at timestep {timestep}. Data size: {len(X_data)}")

    # Return the trained model
    return model

####################
#Example Controller#
####################
####################################
# Please modify the function below #
####################################
def controller(x: np.array, f: callable, sp: np.array, env: callable, u_prev: np.array) -> np.array:
    """
    Model Predictive Control algorithm to optimize control inputs.
    Parameters:
    x (np.array): Current state.
    f (callable): Trained prediction model.
    sp (np.array): Setpoint (target state).
    env (callable): Environment with bounds and parameters.
    u_prev (np.array): Previous control input.

    Returns:
    np.array: Optimal control input.
    """
    controller.team_names = ['Bellal Ahadi', 'Shiam Shrikumar']
    controller.cids = ['01234567', '01234567']

    o_space = env.env_params['o_space']
    a_space = env.env_params['a_space']

    Q_base = 10  # State cost weight
    R_base = 500  # Control cost weight


    horizon = 2
    n_controls = a_space['low'].shape[0]
    control_bounds = [(a_space['low'][i], a_space['high'][i]) for i in range(n_controls)]

    u_prev_norm = (u_prev - a_space['low']) / (a_space['high'] - a_space['low']) * 2 - 1

    evaluation_interval = 0.1  # Evaluate every 0.1 seconds
    time_elapsed = 0  # Track time for evaluation

    # State and error metrics
    error = np.linalg.norm(x - sp)
    error_rate = 0  # Initial rate of error change

    # Adjust weights dynamically based on error and rate of change
    def adjust_weights(error, error_rate):
        if error > 1.0:
            Q = Q_base * 2  # Higher weight for state error
            R = R_base * 0.5  # Lower weight for control effort
        elif error > 0.5:
            Q = Q_base  # Use base values
            R = R_base
        else:
            Q = Q_base * 0.5  # Lower weight for state error
            R = R_base * 2  # Higher weight for control effort

        # Incorporate error rate into adjustment
        if abs(error_rate) > 0.5:  # If error is changing rapidly
            Q *= 1.5  # Prioritize state error
            R *= 0.8  # Allow more control effort
        return Q, R


    def predict_next_state(current_state, control):
        current_state_norm = np.array((current_state - o_space['low'][[1, 8]]) / (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) * 2 - 1)
        current_state_norm = current_state_norm.reshape(1,-1)
        current_state_norm_sq = current_state_norm**2
        current_state_norm_cub = current_state_norm**3
        cross_linear = np.prod(current_state_norm,axis=1,keepdims=True)
        cross_quad_1 = (current_state_norm[:, 0] ** 2 * current_state_norm[:, 1]).reshape(-1, 1)
        cross_quad_2 = (current_state_norm[:, 1] ** 2 * current_state_norm[:, 0]).reshape(-1, 1)
        current_state_norm = np.concatenate((current_state_norm,cross_linear,current_state_norm_sq,cross_quad_1,cross_quad_2,current_state_norm_cub,1/current_state_norm),axis=1)
        control = control.reshape(1,-1)
        x = np.hstack([current_state_norm, control])
        prediction = f.predict(x.reshape(1, -1)).flatten()
        return (prediction + 1) / 2 * (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) + o_space['low'][[1, 8]]
    
    # Controller objective function 
    #def objective(u_sequence):
        #cost = 0
        #x_pred = x_current
        #for i in range(horizon):
            #error = x_pred - sp
            #cost += np.sum(error**2) * Q
            #if i == 0:
                #cost += np.sum((u_sequence[i*n_controls:(i+1)*n_controls]-u_prev)**2)*R
            #else:
                #cost += np.sum((u_sequence[i*n_controls:(i+1)*n_controls]-u_sequence[(i-1)*n_controls:(i)*n_controls])**2)*R
            #u_current = u_sequence[i*n_controls:(i+1)*n_controls]
            #cost += np.sum((u_current - u_prev)**2) * R
            #x_pred = predict_next_state(x_pred, u_current)
        #return cost
    
    def cost_function(u_sequence, Q, R):
        u_sequence = u_sequence.reshape(horizon, n_controls)
        cost = 0
        x_pred = x[1]  # Current state

        for i in range(horizon):

            if i == 0:
                cost += np.sum((u_sequence[i] - u_prev_norm)**2) * R
            else:
                cost += np.sum((u_sequence[i] - u_sequence[i - 1])**2) * R

            x_pred = predict_next_state(x_pred, u_sequence[i])
            error = x_pred - sp
            cost += np.sum(error**2) * Q

            residual = np.abs(x_pred - sp)
            cost += np.sum(residual) * 0.1
        return cost
    
    def grad(x, Q, R, e=1e-5):
        
        gradient = np.zeros(len(x))
        for i in range(len(x)):
            x_1 = x
            x_1[i] = x[i] + e
            f0 = cost_function(x,Q,R)
            f1 = cost_function(x_1,Q,R)
            gradient[i] = (f1-f0)/e
        return gradient
    
    #while time_elapsed < evaluation_interval:

        # Compute rate of error change
        new_error = np.linalg.norm(x - sp)
        error_rate = (new_error - error) / evaluation_interval
        error = new_error

        # Adjust Q and R dynamically
        Q, R = adjust_weights(error, error_rate)

        u_init = np.tile(u_prev_norm, horizon)
        u_bounds_norm = [(-1, 1)] * (horizon * n_controls)
        iter_tot = 100
        tol = 1e-6
        x_i = u_init
        lr = 0.01
        for i in range(iter_tot):
            f_ori = cost_function(x_i,Q,R)
            gradient = grad(x_i,Q,R)
            x_new = x_i - (lr*gradient)
            f_new = cost_function(x_new,Q,R)
            f_best = cost_function(x_i,Q,R)
            if abs(f_new-f_ori)<tol:
                break
            if f_new < f_ori:
                x_i = x_new
            else:
                lr = lr*0.5
        time_elapsed += evaluation_interval
        u_optimal = x_i[n_controls*horizon-1:]
    #return (u_optimal + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']
    

    while time_elapsed < evaluation_interval:

        # Compute rate of error change
        new_error = np.linalg.norm(x - sp)
        error_rate = (new_error - error) / evaluation_interval
        error = new_error

        # Adjust Q and R dynamically
        Q, R = adjust_weights(error, error_rate)

        u_init = np.tile(u_prev_norm, horizon)
        u_bounds_norm = [(-1, 1)] * (horizon * n_controls)

        result = minimize(cost_function, u_init, args=(Q, R), bounds=u_bounds_norm, method='SLSQP')

        if result.success:
            u_optimal = result.x[horizon*n_controls-2:]
        else:
            print("Optimization failed. Using previous control.")
            u_optimal = u_prev_norm

        # Update time elapsed
        time_elapsed += evaluation_interval
        
    return (u_optimal + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']

    #u_init = np.tile(u_prev_norm, horizon)
    #u_bounds_norm = [(-1, 1)] * (horizon * n_controls)

    #result = minimize(cost_function, u_init, bounds=u_bounds_norm, method='SLSQP')

    #if result.success:
        #u_optimal = result.x[:n_controls]
    #else:
        #print("Optimization failed. Using previous control.")
        #u_optimal = u_prev_norm

    #return (u_optimal + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']

    #return u_optimal
    # Initial control guess and bounds (working in normalised control inputs)
    #u_init = np.ones((horizon, 2)) * u_prev
    #bounds = [(-1, 1)] * (horizon * n_controls)
    
    #def Ball_sampling(udim, r_i):
        
        #u      = np.random.normal(0,1,udim)  # random sampling in a ball
        #norm   = np.sum(u**2)**(0.5)
        #r      = random.random()**(1.0/udim)
        #d_init = r*u/norm*r_i*2      # random sampling in a ball

        #return d_init

    #def local_search_quad_model(u_0, r, n_s):
        
        # extract dimension
        #u_dim = u_0.shape[0]
        # note: we change x_list dimention to (n_d,x_dim) so that it has same dimensions as X


        # evaluate first point
        #f_best, u_best = objective(u_0.flatten()), u_0
        #u_best = u_best.reshape(1,u_dim)

        # === first sampling inside the radius === #
        # - (similar to stochastic local search: with proper programming should be a function) - #
        #localu   = np.zeros((n_s,u_dim))  # points sampled
        #localval = np.zeros((n_s))        # function values sampled
        # sampling loop
        #for sample_i in range(n_s):
            #u_trial = u_best + Ball_sampling(u_dim, r) # sampling
            #localu[sample_i,:] = u_trial
            #localval[sample_i] = objective(u_trial.flatten())
        # tracking evaluations
        #return localval, localu
    
    #def quad_func(d, params,u0var):
        
        #quad_func = 0
    
        #n_p = params.shape[0]
        #parameters = params.reshape(n_p,1)
        #n_u = u0var.shape[1]
        #X = u0var + d
        #X = X.astype(np.int64)
        #XBF = X
        #XBF = np.hstack((XBF,X**2))
        #for dim_i in range(n_u-1):
            #for dim_j in range(dim_i+1,n_u):
                #XBF = np.hstack((XBF,X[:,dim_i].reshape(X.shape[0],1)*X[:,dim_j].reshape(X.shape[0],1)))
        #quad_func = XBF@parameters
        #return quad_func
    
    #def opt_quadratic_model(params,u0var,r_t):
        
        #res = minimize(quad_func, args=(params,u0var), x0=np.zeros(2*horizon), method='L-BFGS-B')
        #d_sol = res.x
   
        #return u0var + d_sol
    
    #iter_tol = 100
    #iter_ = int(iter_tol/5)
    #r_t = 1
    #u_orig = u_init.flatten()
    #u_best = u_init.reshape(n_controls*horizon,1)
    #for i in range(iter_):
        #f_val, u_val = local_search_quad_model(u_best,r_t,10)
        #model = Pipeline([('poly',PolynomialFeatures(degree=2,include_bias=False)),('linear',LinearRegression())])
        #model.fit(u_val,f_val)
        #params = model.named_steps['linear'].coef_
        #u_best = u_best.reshape(1,n_controls*horizon)
        #u_best = opt_quadratic_model(params,u_best,1)
        #u_best = u_best.flatten()
        #if objective(u_best)>objective(u_orig):
            #u_best = u_orig
            #r_t = r_t*0.5
        #else:
            #u_orig = u_best
        #u_best = u_best.reshape(n_controls*horizon,1)
        
    #u_best = u_best[:n_controls]
    #u_best = u_best.reshape(1,n_controls)
    #return (u_best + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']

    # Use scipy minimize to optimise the control cost
    #result = minimize(objective, u_init.flatten(), method='powell', bounds=bounds)

    # Return the control input in the actual bounds
    #optimal_control = result.x[:2]
    #return (optimal_control + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']
