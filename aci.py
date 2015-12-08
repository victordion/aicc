"""
Date: December 7 2015
Author: Huanzhu Xu and Jianwei Cui
"""
import math
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

def computeAIC(residual_error, dim, N):
    return N * (math.log(residual_error) + 1) + 2 * (dim + 1)

def computeAICC(residual_error, dim, N):
    return computeAIC(residual_error, dim, N) + 2 * (dim + 1) * (dim + 2) / (N - dim - 2)


def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

"""
Input:
    params: beta0, beta1, beta2, ...
    N: number of data points
    out_dim: number of generated input data dimension, including the constant dimension
    sigma: std. dev. of the noise
Output:
    y: a row vector of size N
    x: data point matrix, with out_dim rows and N columns

"""
def genData(sigma, params, out_dim, N):
    x = np.ones((out_dim, N))
    y = np.zeros(N)

    for i in range(out_dim):
        for j in range(N):
            """ first element is always 1 """
            if i == 0:
                x[i, j] = 1
            else:
                x[i, j] = np.random.normal(0, 1)
                
    for j in range(N):
        for i in range(len(params)):
           y[j] += x[i][j] * params[i]
        y[j] += np.random.normal(0, sigma)    

    return (x, y)

def computeResidualError(params, y, x):
    """
    Need to reverse so params follow the pattern: beta0, beta1, beta2, beta3
    """
    params = params[::-1]

    residual_error = 0.0
    """ Number of data points """
    N = len(y)
    """ Number of effective dimension """
    dim = len(params)

    for j in range(N):
        predict = 0
        for i in range(dim):
            predict += x[i][j] * params[i]
        #print "Predict: %f, Real: %f" % (predict, y[j]) 
        residual_error += (y[j] - predict)**2

    return residual_error / N

x, y = genData(1, [.5, 1, 2, 3], 8, 20)

def getModelCriterion(params, y, x, judge_func):
    sigma_2 = computeResidualError(params, y, x)
    score = judge_func(sigma_2, len(params), len(y))
    print "Model score with order: %d is %f" % (len(params), score)
    return score

print y
print x

#result = reg_m(y, x[1:4, :])
#print result.summary()
#print result.params

for order in range(1, 7):
    params = reg_m(y, x[1: order + 1, :]).params
    getModelCriterion(params, y, x, computeAIC)




