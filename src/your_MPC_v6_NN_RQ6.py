import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

def explorer(x_t: np.array, u_bounds: dict, timestep: int, model=None, uncertainty_threshold=0.1) -> np.array:
    """
    Function to collect more data to train the model using active learning.
    Parameters:
    x_t (np.array): Current state.
    u_bounds (dict): Bounds on control inputs.
    timestep (int): Current timestep.
    model (callable): Predictive model to assess uncertainty.
    uncertainty_threshold (float): Threshold for uncertainty-based exploration.

    Returns:
    np.array: Next control input.
    """
    #u_bounds = {0.1,0.3,0.5,0.7,0.9}
    u_lower = u_bounds['low']
    u_upper = u_bounds['high']

    #u_lower = 0.1
    #u_upper = 0.9

    dimension = len(u_lower)
    #print(u_lower)
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
    
    # Active learning: Evaluate uncertainty if model is provided
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

def model_trainer(data: tuple, env: callable, model=None) -> callable:
    """
    Trains a neural network model using the provided data and environment parameters, using the explorer function.
    Parameters:
    data (np.array): A tuple containing two numpy arrays: data_states and data_controls.
                    data_states is a 3D array with shape (reps, states, n_steps).
                    data_controls is a 3D array with shape (reps, controls, n_steps).
    env (callable): An environment object that contains the environment parameters.
    u_bounds (dict): Control input bounds.
    timestep (int): Current timestep for active learning.
    model (MLPRegressor): The existing neural network model (if any).
    Returns:
    model (MLPRegressor): The trained neural network model.
    """
    data_states, data_controls = data

    # Select only states with indices 1 and 8
    selected_states = data_states[:, [1, 8], :]

    # Normalize the selected states and controls
    o_low, o_high = env.env_params['o_space']['low'][[1, 8]], env.env_params['o_space']['high'][[1, 8]]
    a_low, a_high = env.env_params['a_space']['low'], env.env_params['a_space']['high']
    selected_states_norm = (selected_states - o_low.reshape(1, -1, 1)) / (o_high.reshape(1, -1, 1) - o_low.reshape(1, -1, 1)) * 2 - 1
    data_controls_norm = (data_controls - a_low.reshape(1, -1, 1)) / (a_high.reshape(1, -1, 1) - a_low.reshape(1, -1, 1)) * 2 - 1

    # Get the dimensions
    reps, states, n_steps = selected_states_norm.shape
    _, controls, _ = data_controls_norm.shape

    # Prepare the data
    X_states = selected_states_norm[:, :, :-1].reshape(-1, states)
    X_controls = data_controls_norm[:, :, :-1].reshape(-1, controls)
    X = np.hstack([X_states, X_controls])
    y = selected_states_norm[:, :, 1:].reshape(-1, states)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the neural network model
    model = MLPRegressor(
        hidden_layer_sizes=(64, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        learning_rate_init=0.01,
        alpha=0.001,  # L2 regularization
        early_stopping=True,
        validation_fraction=0.1
    ) 

    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.3f}")

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

def controller(x: np.array, f: callable, sp: np.array, env: callable, u_prev: np.array, evaluation_interval=0.5, total_duration=1.0) -> np.array:
    """
    Enhanced MPC with dynamic adjustment of Q and R and periodic evaluation.
    Parameters:
    x (np.array): Current state.
    f (callable): Trained prediction model.
    sp (np.array): Setpoint (target state).
    env (callable): Environment with bounds and parameters.
    u_prev (np.array): Previous control input.
    evaluation_interval (float): Time interval for periodic evaluation.
    total_duration (float): Total duration for the controller to evaluate.

    Returns:
    np.array: Optimal control input.
    """
    controller.team_names = ['Bellal Ahadi', 'Shiam Shrikumar']
    controller.cids = ['01234567', '01234567']

    o_space = env.env_params['o_space']
    a_space = env.env_params['a_space']

    Q_base = 50
    R_base = 500
    max_horizon = 4

    n_controls = a_space['low'].shape[0]
    u_prev_norm = (u_prev - a_space['low']) / (a_space['high'] - a_space['low']) * 2 - 1

    def predict_next_state(current_state, control):
        current_state_norm = (current_state - o_space['low'][[1, 8]]) / (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) * 2 - 1
        x = np.hstack([current_state_norm, control])
        prediction = f.predict(x.reshape(1, -1)).flatten()
        return (prediction + 1) / 2 * (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) + o_space['low'][[1, 8]]

    def adjust_weights_optimized(error, error_rate, Q_base, R_base, sp, model, x_t):
        """
        Dynamically adjust Q and R, and optimize alpha and beta for the best convergence to the setpoint.
        Parameters:
        error (float): Current error magnitude.
        error_rate (float): Rate of change of the error.
        Q_base (float): Base weight for state cost.
        R_base (float): Base weight for control cost.
        sp (np.array): Setpoint (target state).
        model (callable): Predictive model for error prediction.
        x_t (np.array): Current state.

        Returns:
        tuple: Updated Q, R, alpha, beta
        """
        def optimization_function(params):
            """
            Cost function that optimizes alpha and beta to minimize the error from the setpoint.
            """
            alpha, beta = params
            
            # Adjust Q and R based on the optimized alpha and beta
            Q = Q_base * np.exp(alpha * error)  # Exponentially increase Q with error
            R = R_base * np.exp(-beta * abs(error_rate))  # Exponentially decrease R with rapid error change

            x_input = x_t[:4]

            # Predict the next state using the model (e.g., MLPRegressor)
            #x_candidate = np.hstack([x_t, model.predict(np.hstack([x_t, sp]).reshape(1, -1))])  # Example prediction
            x_candidate = model.predict(x_input.reshape(1,-1)).flatten()
            # Calculate new error with the optimized weights
            new_error = np.linalg.norm(x_candidate - sp)
            
            # Return the cost function (the larger the error, the worse the performance)
            return new_error

        # Optimize alpha and beta
        result = minimize(optimization_function, x0=[2.0, 1.0], bounds=((0.1, 5.0), (0.1, 5.0)))  # Initial guess and bounds for alpha and beta
        
        # Retrieve the optimized alpha and beta
        alpha_opt, beta_opt = result.x

        # Update Q and R with the optimal values
        Q = Q_base * np.exp(alpha_opt * error)  # Exponentially increase Q with optimized alpha
        R = R_base * np.exp(-beta_opt * abs(error_rate))  # Exponentially decrease R with optimized beta

        # Further adjust based on thresholds for fine-tuning
        if error > 0.5:
            Q *= 2  # Additional boost for high errors
            R *= 0.5  # Lower control penalty for high errors
        elif error > 0.35:
            Q *= 1.5
            R *= 0.7


        return Q, R, alpha_opt, beta_opt

    def cost_function(u_sequence, Q, R):
        u_sequence = u_sequence.reshape(horizon, n_controls)
        cost = 0
        x_pred = x[1]
        prev_error = np.linalg.norm(x - sp)

        for t in range(horizon):
            x_pred = predict_next_state(x_pred, u_sequence[t])
            error = np.linalg.norm(x_pred - sp)

            # Compute cost terms
            cost += np.sum((error**2) * Q)

            if t == 0:
                cost += np.sum((u_sequence[t] - u_prev_norm)**2) * R
            else:
                cost += np.sum((u_sequence[t] - u_sequence[t - 1])**2) * R

            if error > prev_error:
                cost += 100 * (error - prev_error)  # Penalize wrong-direction actions

            prev_error = error

        return cost

    # Compute number of evaluation steps
    num_steps = int(total_duration / evaluation_interval)

    # Periodic evaluation loop
    for step in range(num_steps):
        error = np.linalg.norm(x - sp)
        error_rate = (error - np.linalg.norm(x - sp)) / evaluation_interval  # Simplified error rate computation

          # Dynamically adjust Q and R and optimize alpha and beta
        Q, R, alpha_opt, beta_opt = adjust_weights_optimized(error, error_rate, Q_base, R_base, sp, f, x)

        # Adjust horizon for higher setpoints
        if error > 0.5:
            horizon = max_horizon
        else:
            horizon = 2

        # Optimize using adjusted weights
        u_init = np.tile(u_prev_norm, horizon)
        u_bounds_norm = [(-1, 1)] * (horizon * n_controls)
        result = minimize(cost_function, u_init, args=(Q, R), bounds=u_bounds_norm, method='SLSQP')

        if result.success:
            u_optimal = result.x[:n_controls]
        else:
            print("Optimization failed at step:", step)
            u_optimal = u_prev_norm

    # Return the final optimized control
    return (u_optimal + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']
