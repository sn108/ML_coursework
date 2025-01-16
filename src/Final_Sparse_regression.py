from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, ElasticNet, Ridge
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
from scipy.stats.qmc import LatinHypercube

def explorer(x_t: np.array, u_bounds: dict, timestep: int, num_samples: int=300, model=None, uncertainty_threshold=0.1) -> np.array:
    '''
    Function to collect more data to train the model.
    Combines Van der Corput sequences with Latin Hypercube Sampling (LHS).
    '''
    u_lower = u_bounds['low']
    u_upper = u_bounds['high']

    dimension = len(u_lower)
    # print(dimension)
    max_samples = 1000  # Limit for Sobol-like sequence
    
    idx = timestep % max_samples
    intervals = np.linspace(0, 1, num_samples + 1)

    # Generate LHS samples
    lhs_sampler = LatinHypercube(d=dimension)
    lhs_samples = lhs_sampler.random(n=300)  # Single sample of LHS
    # print(lhs_samples.shape)
   
    samples_0 = np.sort(lhs_samples[:,0])
    samples_1 = np.sort(lhs_samples[:,1])
    
    # print(samples_1.shape)
    # print(samples_0.shape)

    samples_1 = np.concatenate([samples_1[2:,],samples_1[:2]])
    lhs_samples = np.vstack([samples_0,samples_1])
    
    lhs_samples = lhs_samples.T
    # print(lhs_samples.shape)
    
    # Scale the sample to the input bounds
    # for i in range(300):
    #     u_plus = u_lower +  (u_upper - u_lower)*lhs_samples[:,i]
    # u_plus = u_lower[:, np.newaxis] + (u_upper - u_lower)[:, np.newaxis] * lhs_samples
    # u_plus = u_lower[:, np.newaxis] + (u_upper[:, np.newaxis] - u_lower[:, np.newaxis]) * lhs_samples

    # for i in range(dimension):
    #     lhs_sample[i] = np.random.uniform(intervals[timestep % num_samples], intervals[(timestep % num_samples) + 1])
    # intervals[timestep % num_samples], intervals[(timestep % num_samples) + 1]
    
    # if not hasattr(explorer, "current_cycle"):
    #     explorer.current_cycle = -1

    # if timestep == 0:
    #     explorer.current_cycle += 1

    # row_index = timestep % num_samples + explorer.current_cycle * 60
    # # row_index = intervals[(timestep % num_samples)]
    # lhs_sample = lhs_samples[row_index,:]
    
    # u_plus = u_lower +  (u_upper - u_lower)*lhs_sample

    # return u_plus
    
    lhs_sample = np.zeros(dimension)
    scaled_sample = np.zeros(dimension)
    for i in range(dimension):
        # Divide the range into "nsteps" intervals (60 timesteps)
        interval_size = (u_upper[i] - u_lower[i]) / 60

        # Determine which interval to sample from for this timestep
        interval_idx = (timestep + i) % 60  # Shift index for each variable
        lower_bound = u_lower[i] + interval_idx * interval_size
        upper_bound = lower_bound + interval_size

        # Sample within the selected interval
        lhs_sample[i] = np.random.uniform(lower_bound, upper_bound)
        scaled_sample[i] =  lower_bound + (upper_bound - lower_bound) * lhs_sample[i]

    # return lhs_sample

    return scaled_sample

def model_trainer(data: tuple, env: callable) -> callable:
    """
    Trains a predictive model using a neural network with regularization and data augmentation.
    Parameters:
    data (tuple): A tuple containing `data_states` and `data_controls` arrays.
    env (callable): An environment object containing `o_space` and `a_space` bounds.

    Returns:
    callable: Trained predictive model.
    """
    # del explorer.current_state

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
    # X_states = np.concatenate((X_states,cross_linear,X_states_sq,cross_quad_1,cross_quad_2,X_states_cub,1/X_states,1/X_states_sq),axis =1)
    X_states = np.concatenate((X_states,cross_linear,X_states_sq,cross_quad_1,cross_quad_2,X_states_cub,1/X_states),axis =1)
    X_controls = data_controls_norm[:, :, :-1].reshape(-1, controls)
    X = np.hstack([X_states, X_controls])  # Combine state and control inputs
    y = selected_states_norm[:, :, 1:].reshape(-1, states)

    # Data Augmentation: Add small noise to inputs
    noise_factor = 0.01
    X_augmented = X + np.random.normal(0, noise_factor, X.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Lasso(alpha=0.003, max_iter=10000)
    
    # model = ElasticNet(alpha=0.003, l1_ratio=0.02, max_iter=10000)
    # model = Ridge(alpha=0.003,max_iter=10000)
    
    # cv_folds = 5
 
    # param_grid = {
    #     'alpha': [0.001, 0.004,0.008,0.01],
    #     'l1_ratio': [0.3,0.5, 0.7, 0.9],
    # }
    # grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='neg_mean_squared_error')
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    # model = grid_search.best_estimator_ 
    
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Training Completed - MSE: {mse:.4f}, R2: {r2:.3f}")

    return model

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

    # Q_base = 300, R_base = 1000
    # Q_base = 30  # State cost weight
    # R_base = 100  # Control cost weight [changed from R =100]
    
    Q_base = 10  # State cost weight
    R_base = 150  # Control cost weight [changed from R =100]
    
    # S = 1520  # Terminal state cost weight
    # I = 1500  # Intiial state cost weight

    # S = 350  # Terminal state cost weight
    S = 100  # Terminal state cost weight
    I = 100  # Intiial state cost weight
    

    horizon = 5 # Increase horizon for better set-point tracking
    # integral_cost = 29.85
    x_current = x[1]  # Current state

    # horizon = 2
    n_controls = a_space['low'].shape[0]
    control_bounds = [(a_space['low'][i], a_space['high'][i]) for i in range(n_controls)]

    u_prev_norm = (u_prev - a_space['low']) / (a_space['high'] - a_space['low']) * 2 - 1

    evaluation_interval = 0.1  # Evaluate every 0.1 seconds
    time_elapsed = 0  # Track time for evaluation

    # State and error metrics
    error = np.linalg.norm(x - sp)
    error_rate = 0  # Initial rate of error change

    def adjust_weights(error, error_rate):
        if error > 1.0:
            Q = Q_base * 2.0  # Higher weight for state error
            R = R_base * 0.5  # Lower weight for control effort
            # I = I * 0.9
        elif error > 0.5:
            Q = Q_base * 1.0 # Use base values
            R = R_base * 1.0
            # I = I*0.7
        else:
            Q = Q_base * 0.5  # Lower weight for state error
            R = R_base * 2  # Higher weight for control effort

        # Incorporate error rate into adjustment
        if abs(error_rate) > 0.5:  # If error is changing rapidly
            Q *= 1.5 # Prioritize state error
            R *= 1.0  # Allow more control effort
        return Q, R

    
        
    def predict_next_state(current_state, control):
        current_state_norm = np.array((current_state - o_space['low'][[1, 8]]) / (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) * 2 - 1)
        current_state_norm = current_state_norm.reshape(1,-1)
        current_state_norm_sq = current_state_norm**2
        current_state_norm_cub = current_state_norm**3
        cross_linear = np.prod(current_state_norm,axis=1,keepdims=True)
        cross_quad_1 = (current_state_norm[:, 0] ** 2 * current_state_norm[:, 1]).reshape(-1, 1)
        cross_quad_2 = (current_state_norm[:, 1] ** 2 * current_state_norm[:, 0]).reshape(-1, 1)
        # current_state_norm = np.concatenate((current_state_norm,cross_linear,current_state_norm_sq,cross_quad_1,cross_quad_2,current_state_norm_cub,1/current_state_norm,1/current_state_norm_sq),axis=1)
        current_state_norm = np.concatenate((current_state_norm,cross_linear,current_state_norm_sq,cross_quad_1,cross_quad_2,current_state_norm_cub,1/current_state_norm),axis=1)
        control = control.reshape(1,-1)
        x = np.hstack([current_state_norm, control])
        prediction = f.predict(x.reshape(1, -1)).flatten()
        return (prediction + 1) / 2 * (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) + o_space['low'][[1, 8]]
    
    # Controller objective function 
    def cost_function(u_sequence, Q, R):
        u_sequence = u_sequence.reshape(horizon, n_controls)
        cost = 0
        x_pred = x[1]  # Current state
        integral = np.zeros_like(x_current)

        # initial_error = x_pred - sp
        # cost += I * np.sum(initial_error**2)

        for i in range(horizon):

            # error = x_pred - sp
            # time_weight = (horizon - i) / horizon  # Increase weight as horizon progresses
            # cost += time_weight * np.sum(error**2) * Q
            
            # u_current = u_sequence[i * n_controls:(i + 1) * n_controls]
            # cost += np.sum((u_current - u_prev_norm)**2) * R
            
            # x_pred = predict_next_state(x_pred, u_current)


            error = x_pred - sp
            time_weight = (horizon - i) / horizon  # Increase weight as horizon progresses          
            cost += time_weight * np.sum(error**2) * Q
            
            # REMOVED ALL Integral cost
            
            # if i == 0:
            #     u_current = u_sequence[i * n_controls:(i + 1) * n_controls]
            #     cost += np.sum((u_current - u_prev_norm)**2) * R
            #     # cost += np.sum((u_sequence[i] - u_prev_norm)**2) * R
            # else:
            #     cost += np.sum((u_sequence[i] - u_sequence[i - 1])**2) * R
            
            u_current = u_sequence[i * n_controls:(i + 1) * n_controls]
            cost += np.sum((u_current - u_prev_norm)**2) * R
            x_pred = predict_next_state(x_pred, u_sequence[i])
            
        #  Terminal state cost
        final_error = x_pred - sp
        cost += S * np.sum(final_error**2)
        
        return cost
    

    while time_elapsed < evaluation_interval:

        # Compute rate of error change
        new_error = np.linalg.norm(x - sp)
        error_rate = (new_error - error) / evaluation_interval
        error = new_error

        # Adjust Q and R dynamically
        
        # Q, R = adjust_weights(error, error_rate)

        Q,R = Q_base, R_base
        u_init = np.tile(u_prev_norm, horizon)
        u_bounds_norm = [(-1, 1)] * (horizon * n_controls)

        # result = minimize(cost_function, u_init, args=(Q, R), bounds=u_bounds_norm, method='SLSQP')
        result = minimize(cost_function, u_init, args=(Q, R), bounds=u_bounds_norm, method='COBYLA')  # Debugging and limits)

        if result.success:
            u_optimal = result.x[horizon*n_controls-2:]
        else:
            print("Optimization failed. Using previous control.")
            u_optimal = u_prev_norm

        # Update time elapsed
        time_elapsed += evaluation_interval/1
        
    return (u_optimal + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']

   
