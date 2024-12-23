from scipy.optimize import minimize
import sklearn
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline 
import numpy as np
import random 

class quad_sur:
    
    def __init__(self,func,x_dim,bound,iter_tot):
        self.fun = func
        self.xdim = x_dim
        self.bounds = bound
        self.iter_tol = iter_tot
        self.init = (bound[:,1]-bound[:,0])/3
        
    def Ball_sampling(self, ndim, r_i):
        '''
        This function samples randomly withing a ball of radius r_i
        '''
        u      = np.random.normal(0,1,ndim)  # random sampling in a ball
        norm   = np.sum(u**2)**(0.5)
        r      = random.random()**(1.0/ndim)
        d_init = r*u/norm*r_i*2      # random sampling in a ball

        return d_init
    
    def local_search_quad_model(self, x_0, r, n_s):
        '''
        This function is an optimization routine following a local random search
        '''
        # extract dimension
        x_dim = x_0.shape[0]
        # note: we change x_list dimention to (n_d,x_dim) so that it has same dimensions as X


        # evaluate first point
        f_best, x_best = self.fun.fun_test(x_0), x_0
        x_best = x_best.reshape(1,x_dim)

        # === first sampling inside the radius === #
        # - (similar to stochastic local search: with proper programming should be a function) - #
        localx   = np.zeros((n_s,x_dim))  # points sampled
        localval = np.zeros((n_s))        # function values sampled
        # sampling loop
        for sample_i in range(n_s):
            x_trial = x_best + self.Ball_sampling(x_dim, r) # sampling
            localx[sample_i,:] = x_trial
            localval[sample_i] = self.fun.fun_test(x_trial)
        # tracking evaluations
        return localval, localx
        
    def quad_func(self, d, params,x0var):
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
        X = X.astype(np.int64)
        XBF = X
        XBF = np.hstack((XBF,X**2))
        for dim_i in range(n_x-1):
            for dim_j in range(dim_i+1,n_x):
                XBF = np.hstack((XBF,X[:,dim_i].reshape(X.shape[0],1)*X[:,dim_j].reshape(X.shape[0],1)))
        quad_func = XBF@parameters
        return quad_func

    def opt_quadratic_model(self, params,x0var,r_t):
        '''
        a,b,c: parameters estimated for the quadratic model
        x0var: initial point: last
        '''
        res = minimize(self.quad_func, args=(params,x0var), x0=np.zeros(32), method='L-BFGS-B' )
        d_sol = res.x
    
        return x0var + d_sol

    def main(self):
        '''
        params: parameters that define the rbf model
        X:      matrix of previous datapoints
        '''
        x_best = self.init
        iter_ = int(self.iter_tol/5)
        r_t = 1
        for i in range(iter_):
            x_orig = x_best
            x_best = x_best.reshape(self.xdim,1)
            f_val, x_val = self.local_search_quad_model(x_best,r_t,4)
            model = Pipeline([('poly',PolynomialFeatures(degree=2,include_bias=False)),('linear',LinearRegression())])
            model.fit(x_val,f_val)
            params = model.named_steps['linear'].coef_
            x_best = x_best.reshape(1,self.xdim)
            x_best = self.opt_quadratic_model(params,x_best,0.5)
            if self.fun.fun_test(x_best)>self.fun.fun_test(x_orig):
                x_best = x_orig
                r_t = r_t*0.5
        f_best = self.fun.fun_test(x_best)
        return x_best, f_best

def our_alg(function,x_dim,bounds,iter_total):
    quad = quad_sur(function,x_dim,bounds,iter_total)
    x_best,f_best = quad.main()
    team_names = ['Shiam Srikumar', 'Bellal Ahidi', 'Stutya Nagpal', 'John Adeyemi']
    cids = ['0202119']
    return x_best, f_best, team_names, cids
    
