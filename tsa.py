import matplotlib.pyplot as plt
import math
import numpy as np
import statsmodels.tsa.api as tsa
import sys


def genData(sigma, params, N):
    data = np.zeros(N)
    for i in range(len(params)):
        data[i] = np.random.normal(0, sigma)
        #data[i] = 5
	
    for i in range(len(params), N):
	for j in range(len(params)):
            data[i] += params[j] * data[i - j - 1]
        data[i] += np.random.normal(0, sigma)

    return data


def getCriterion(order, data, judge_func):
    results = tsa.AR(data).fit(order)
    N = len(data)
    dim = order
    #residual_error = getResidualError(data, results.params)

    if judge_func == "aic":
        return results.aic
        #return N * (math.log(residual_error) + 1) + 2 * (dim + 1) 
    elif judge_func == "bic":
        return results.bic
        #return N * (math.log(residual_error) + 1) + math.log(N) * (dim + 1)
    else:
        return results.aic + 2 * (dim + 1) * (dim + 2) / (N - dim - 2)
        #return N * (math.log(residual_error) + 1) + 2 * (dim + 1) + 2 * (dim + 1) * (dim + 2) / (N - dim - 2)


def getResidualError(data, fit_params):
    err = 0.0
    predicted = np.copy(data)
    fit_params = fit_params[1:]

    for i in range(len(fit_params)):
        predicted[i] = np.random.normal(0, 1)
        err += (data[i] - predicted[i])**2

    for i in range(len(fit_params), len(data)):
        predicted[i] = 0.0
        for j in range(0, len(fit_params)):
            predicted[i] += fit_params[j] * predicted[i - j - 1]
         
        err += (data[i] - predicted[i])**2
    
    #print fit_params
    #print data
    #print predicted
    #print "Order Number %d; Error: %5.10f" % (len(fit_params), err/(len(data)))
    #response = raw_input()
    return err/len(data)

if __name__ == "__main__":
    N = 23
    noise_sigma = 1
    params = [0.99, -0.8, 0.7]
	
    aic_by_order = np.zeros(N)
    aic_win_times = np.zeros(N)
    
    aicc_by_order = np.zeros(N)
    aicc_win_times = np.zeros(N)
    
    bic_by_order = np.zeros(N)
    bic_win_times = np.zeros(N)

    for i in range(N):
        aic_by_order[i] = sys.float_info.max    
        aicc_by_order[i] = sys.float_info.max    
        bic_by_order[i] = sys.float_info.max    

    for experiemnt in range(100):
        data = genData(noise_sigma, params, N)
        print "Generate data is: "
        print data
        for order in range(1, N - 4):
            aic_by_order[order] = getCriterion(order, data, "aic")
            aicc_by_order[order] = getCriterion(order, data, "aicc")
            bic_by_order[order] = getCriterion(order, data, "bic")
        print bic_by_order        

        aic_winner_order = np.argmin(aic_by_order)
        aicc_winner_order = np.argmin(aicc_by_order)
        bic_winner_order = np.argmin(bic_by_order)
        
        aic_win_times[aic_winner_order] += 1
        aicc_win_times[aicc_winner_order] += 1
        bic_win_times[bic_winner_order] += 1

    print 'aic_win'
    print aic_win_times
    print
    print 'aicc_win'
    print aicc_win_times
    print
    print 'bic_win'
    print bic_win_times
    """
    data = genData(1, [0.99, -0.8], 20)
    print data

    results = tsa.AR(data).fit(2)
    print results.params
    print results.bic
    print getCriterion(2, data, "bic")

    print "--"
    results = tsa.AR(data).fit(5)
    print results.params
    print results.bic
    print getCriterion(5, data, "bic")
    """
