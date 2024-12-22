import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Explorer function
def explorer(x_t: np.array, u_bounds: dict, timestep: int) -> np.array:
    """
    Function to collect more data to train the model.
    Parameters:
    x_t (np.array): Current state.
    u_bounds (dict): Bounds on control inputs.
    timestep (int): Current timestep.

    Returns:
    np.array: Next control input.
    """
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

    u_plus = u_lower + sobol_like * (u_upper - u_lower)
    return u_plus

# Model Trainer
def model_trainer(data: tuple, env: callable) -> callable:
    """
    Trains a predictive model using the provided data and environment parameters.
    Parameters:
    data (tuple): A tuple containing `data_states` and `data_controls` arrays.
    env (callable): An environment object containing `o_space` and `a_space` bounds.

    Returns:
    callable: Trained predictive model.
    """
    data_states, data_controls = data

    selected_states = data_states[:, [1, 8], :]

    o_low, o_high = env.env_params['o_space']['low'][[1, 8]], env.env_params['o_space']['high'][[1, 8]]
    a_low, a_high = env.env_params['a_space']['low'], env.env_params['a_space']['high']

    selected_states_norm = (selected_states - o_low.reshape(1, -1, 1)) / (o_high.reshape(1, -1, 1) - o_low.reshape(1, -1, 1)) * 2 - 1
    data_controls_norm = (data_controls - a_low.reshape(1, -1, 1)) / (a_high.reshape(1, -1, 1)) * 2 - 1

    reps, states, n_steps = selected_states_norm.shape
    _, controls, _ = data_controls_norm.shape

    X_states = selected_states_norm[:, :, :-1].reshape(-1, states)
    X_controls = data_controls_norm[:, :, :-1].reshape(-1, controls)
    X = np.hstack([X_states, X_controls])
    y = selected_states_norm[:, :, 1:].reshape(-1, states)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Training Completed - MSE: {mse:.4f}, R2: {r2:.3f}")

    return model

# MPC Controller
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

    Q = 10  # State cost weight
    R = 500  # Control cost weight

    horizon = 5
    n_controls = a_space['low'].shape[0]
    control_bounds = [(a_space['low'][i], a_space['high'][i]) for i in range(n_controls)]

    u_prev_norm = (u_prev - a_space['low']) / (a_space['high'] - a_space['low']) * 2 - 1

    def predict_next_state(current_state, control):
        current_state_norm = (current_state - o_space['low'][[1, 8]]) / (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) * 2 - 1
        x = np.hstack([current_state_norm, control])
        prediction = f.predict(x.reshape(1, -1)).flatten()
        return (prediction + 1) / 2 * (o_space['high'][[1, 8]] - o_space['low'][[1, 8]]) + o_space['low'][[1, 8]]

    def cost_function(u_sequence):
        u_sequence = u_sequence.reshape(horizon, n_controls)
        cost = 0
        x_pred = x[1]  # Current state

        for i in range(horizon):
            error = x_pred - sp
            cost += np.sum(error**2) * Q
            if i == 0:
                cost += np.sum((u_sequence[i] - u_prev_norm)**2) * R
            else:
                cost += np.sum((u_sequence[i] - u_sequence[i - 1])**2) * R
            x_pred = predict_next_state(x_pred, u_sequence[i])

        return cost

    u_init = np.tile(u_prev_norm, horizon)
    u_bounds_norm = [(-1, 1)] * (horizon * n_controls)

    result = minimize(cost_function, u_init, bounds=u_bounds_norm, method='SLSQP')

    if result.success:
        u_optimal = result.x[:n_controls]
    else:
        print("Optimization failed. Using previous control.")
        u_optimal = u_prev_norm

    return (u_optimal + 1) / 2 * (a_space['high'] - a_space['low']) + a_space['low']
