from scipy.optimize import minimize
import numpy as np
import random 
def Ball_sampling(ndim, r_i):
    '''
    This function samples randomly withing a ball of radius r_i
    '''
    u      = np.random.normal(0,1,ndim)  # random sampling in a ball
    norm   = np.sum(u**2)**(0.5)
    r      = random.random()**(1.0/ndim)
    d_init = r*u/norm*r_i*2      # random sampling in a ball

    return d_init
    
def local_search_quad_model(f, x_0, r, n_s):
    '''
    This function is an optimization routine following a local random search
    '''
    # extract dimension
    x_dim = x_0.shape[0]
    # note: we change x_list dimention to (n_d,x_dim) so that it has same dimensions as X


    # evaluate first point
    f_best, x_best = f.fun_test(x_0), x_0
    x_best = x_best.reshape(1,x_dim)

    # === first sampling inside the radius === #
    # - (similar to stochastic local search: with proper programming should be a function) - #
    localx   = np.zeros((n_s,x_dim))  # points sampled
    localval = np.zeros((n_s))        # function values sampled
    # sampling loop
    for sample_i in range(n_s):
        x_trial = x_best + Ball_sampling(x_dim, r) # sampling
        localx[sample_i,:] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # tracking evaluations
    return localval, localx

#########################
# --- Random search --- #
#########################

def Random_search(f, n_p, bounds_rs, iter_rs):

    # arrays to store sampled points
    localx   = np.zeros((n_p,iter_rs))  # points sampled
    localval = np.zeros((iter_rs))        # function values sampled
    # bounds
    bounds_range = bounds_rs[:,1] - bounds_rs[:,0]
    bounds_bias  = bounds_rs[:,0]

    for sample_i in range(iter_rs):
        x_trial = np.random.uniform(0, 1, n_p)*bounds_range + bounds_bias # sampling
        localx[:,sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial)
    # choosing the best
    minindex = np.argmin(localval)
    f_b      = localval[minindex]
    x_b      = localx[:,minindex]

    #return f_b, x_b
    return localval, localx

def LS(params, X, Y):
    '''
    X: matrix: [x^(1),...,x^(n_d)]
    Y: vecor:  [f(x^(1)),...,f(x^(n_d))]
    '''
    # renaming parameters for clarity (this can be avoided)
    #a = params[0]; b = params[1]; c = params[2]
    # number of datapoints
    n_d = Y.shape[0]
    n_p = params.shape[0]
    WLS = 0
    n_x = X.shape[1]
    XBF = X**2
    # weighted least squares
    for dim_i in range(n_x-1):
        for dim_j in range(dim_i+1,n_x):
            XBF = np.hstack((XBF,X[:,dim_i].reshape(X.shape[0],1)*X[:,dim_j].reshape(X.shape[0],1)))
    parameters = params.reshape(n_p,1)
    model = XBF@parameters
    #model = np.zeros(X.shape[0])
    #model = model.reshape(X.shape[0],1)
    #for i in range(n_d):
        #for d in range(params.shape[0]):
            #model[i][0] = model[i][0] + params[d]*X[i][d]
    LS = np.sum([(model[j]-Y[j])**2 for j in range(n_d)]) 
    return LS

def quadratic_model_estimation(X, Y, p0):
    '''
    X: matrix: [x^(1),...,x^(n_d)]
    Y: vecor:  [f(x^(1)),...,f(x^(n_d))]
    p0: initial guess, preferably from last iteration
    '''
    # minimizing weighted least squares function with scipy
    res = minimize(LS, args=(X,Y), x0=p0, method='L-BFGS-B' )
    # obtaining solution
    params = res.x
    #f_val  = res.fun

    return params
    
def quad_func(d, params,x0var):
    '''
    d: vector:  [d_1,d_2]: distance from centre in x,y coordinates
    '''
    quad_func = 0
    #for i in range(32):
        #quad_func = quad_func + params[i]*(x0var[i]+d[i])
    n_p = params.shape[0]
    parameters = params.reshape(n_p,1)
    n_x = x0var.shape[1]
    X = x0var + d
    XBF = X**2
    for dim_i in range(n_x-1):
        for dim_j in range(dim_i+1,n_x):
            XBF = np.hstack((XBF,X[:,dim_i].reshape(X.shape[0],1)*X[:,dim_j].reshape(X.shape[0],1)))
    quad_func = XBF@parameters
    return quad_func

##########################################
# --- optimising the quadratic model --- #
##########################################
from scipy.optimize import NonlinearConstraint
def opt_quadratic_model(params,x0var,r_t):
    '''
    a,b,c: parameters estimated for the quadratic model
    x0var: initial point: last
    '''
    # trust region constraint
    cons = ({'type': 'ineq', 'fun': lambda d:  r_t**2 - (np.sum(d**2)) })
    #con = lambda x:  (x[0]**2 + x[1]**2)**2
    #nlc = NonlinearConstraint(con, -np.inf, r_t**2)

    # minimising quadratic model
    res = minimize(quad_func, args=(params,x0var), x0=np.zeros(32),
                   method='trust-constr', constraints=cons, bounds=((-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t),(-r_t,r_t)))
    # Note: bounds are added: nonlinear trust region is handled poorly by SLSQP
    # retrieving solution
    d_sol = res.x
    #print('res = ',res) print status
    # returning solution
    return x0var + d_sol
####################
# Powell algorithm #
####################

def your_alg(f, x_dim, bounds, iter_tot):
    '''
    params: parameters that define the rbf model
    X:      matrix of previous datapoints
    '''

    #n_rs = int(min(100,max(iter_tot*.05,100)))       # iterations to find good starting point

    # evaluate first point
    #f_val, x_val = Random_search(f, x_dim, bounds, n_rs)
    #iter_          = iter_tot - n_rs
    x_best = (bounds[:,1]-bounds[:,0])/2
    params_dim = int(0.5*x_dim*(1+x_dim))
    params = np.ones(params_dim)
    iter_ = int(iter_tot/20)
    for i in range(iter_):
        x_best = x_best.reshape(x_dim,1)
        f_val, x_val = local_search_quad_model(f,x_best,0.5,19)
        params = quadratic_model_estimation(x_val,f_val,params)
        x_best = x_best.reshape(1,x_dim)
        #xbf = x_best**2
        #for dim_i in range(x_dim):
            #for dim_j in range(dim_i+1,x_dim):
                #xbf = np.hstack((xbf,x_best[:,dim_i].reshape(x_best.shape[0],1)*x_best[:,dim_j].reshape(x_best.shape[0],1)))
        x_best = opt_quadratic_model(params,x_best,5)
    f_best = f.fun_test(x_best)
    #opt = minimize(f.fun_test, x_best, bounds=bounds, method='Powell', 
                    #options={'maxfev': iter_}) 

    team_names = ['7','8']
    cids = ['01234567']
    return x_best, f_best, team_names, cids
    #return opt.x, opt.fun, team_names, cids
