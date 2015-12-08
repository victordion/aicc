import numpy as np
from scipy import stats
import statsmodels.api as sm

""" model: a linear model
Y: [N 1] sized data
X: [N M] sized data
Np: number of parameters of the model
N: number of data points
M: 
"""

def aci(model, Y, X, Np, N):
    pass

def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results


# params: beta0, beta[last], beta[last-1], ...
def genData(sigma, params, out_dim, N):
    
    x = np.ones((N, out_dim))
    y = np.zeros((N, 1))


    for i in range(N):
        for j in range(out_dim):
            if j == 0:
                x[i, j] = 1
            else:
                x[i, j] = np.random.normal(0, 1)
                
    for i in range(N):
       for j in range(len(params)):
           y[i] += x[i][j] * params[j]
       y[i] += np.random.normal(0, sigma)    

    return (x, y)

def compute_residual(params, y, x):
    params = params[1::-1]
    result = 0
    n = len(y)
    m = len(x[0])
    for i in range(n):
        temp = 0
        for j in range(m):
            if j < len(params):
                temp += x[i][j] * params[j]



x, y = genData(1, [2, 4, 3], 8, 20)


print y.T
print x.T
result = reg_m(y.T[0], x.T[1:4, :])
print result.summary()
print result.params



