# cost function for the SIR model for python 2.7
# Marisa Eisenberg (marisae@umich.edu)
# Yu-Han Kao (kaoyh@umich.edu) -7-9-17

import numpy as np
import sir_ode
from scipy.stats import poisson
from scipy.stats import norm

from scipy.integrate import odeint as ode


def NLL(params, data, times):  # negative log likelihood
    params = np.abs(params)
    data = np.array(data)
    res = ode(sir_ode.model, sir_ode.x0fcn(params, data), times, args=(params,))
    y = sir_ode.yfcn(res, params)
    nll = sum(y) - sum(data * np.log(y))
    # note this is a slightly shortened version--there's an additive constant term missing but it
    # makes calculation faster and won't alter the threshold. Alternatively, can do:
    # nll = -sum(np.log(poisson.pmf(np.round(data),np.round(y)))) # the round is b/c Poisson is for (integer) count data
    # this can also barf if data and y are too far apart because the dpois will be ~0, which makes the log angry

    # ML using normally distributed measurement error (least squares)
    # nll = -sum(np.log(norm.pdf(data,y,0.1*np.mean(data)))) # example WLS assuming sigma = 0.1*mean(data)
    # nll = sum((y - data)**2)  # alternatively can do OLS but note this will mess with the thresholds
    #                             for the profile! This version of OLS is off by a scaling factor from
    #                             actual LL units.
    return nll

def NLL_new(params, data, times):  # negative log likelihood
    params_new = np.abs(params)
    data = np.array(data)
    res_new = ode(sir_ode.model_new, sir_ode.x0fcn_new(params_new, data), times, args=(params,))
    y_new = sir_ode.yfcn_new(res_new, params_new)
    nll_new = sum(y_new) - sum(data * np.log(y_new))
    # note this is a slightly shortened version--there's an additive constant term missing but it
    # makes calculation faster and won't alter the threshold. Alternatively, can do:
    # nll = -sum(np.log(poisson.pmf(np.round(data),np.round(y)))) # the round is b/c Poisson is for (integer) count data
    # this can also barf if data and y are too far apart because the dpois will be ~0, which makes the log angry

    # ML using normally distributed measurement error (least squares)
    # nll = -sum(np.log(norm.pdf(data,y,0.1*np.mean(data)))) # example WLS assuming sigma = 0.1*mean(data)
    # nll = sum((y - data)**2)  # alternatively can do OLS but note this will mess with the thresholds
    #                             for the profile! This version of OLS is off by a scaling factor from
    #                             actual LL units.
    return nll_new