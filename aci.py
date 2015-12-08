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


def genData(sigma, params, out_dim, N):
    
    x = np.zeros((N, out_dim))
    y = np.zeros((N, 1))

    coefs = np.zeros((1, out_dim))

    for i in range(out_dim):
        if i < len(params):
            coefs[0, i] = params[i]

    for i in range(N):
        x[i, :] = np.random.rand(1, out_dim)
        y[i] = sum(np.multiply(x[i, :], coefs[0, :])) + np.random.normal(0, sigma)

     
    return (x, y)


x, y = genData(1.0, [1, 2, 3], 7, 10)


print x.T
print y.T

result = reg_m(y.T, x.T)
    



