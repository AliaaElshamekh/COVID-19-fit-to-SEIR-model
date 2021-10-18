# Profile Likelihood Generator
# Marisa Eisenberg (marisae@umich.edu)
# Yu-Han Kao (kaoyh@umich.edu) -7-9-17

# Input definitions
# params = starting parameters (all, including the one to be profiled)
# profparam = index within params for the parameter to be profiled
#   ---reminder to make this allow you to pass the name instead later on
# costfun = cost function for the model - should include params, times, and data as arguments.
#   Note costfun doesn't need to be specially set up for fixing the profiled parameter,
#   it's just the regular function you would use to estimate all the parameters
#   (it will get reworked to fix one of them inside ProfLike)
# times, data = data set (times & values, or whatever makes sense)
#   ---possibly change this so it's included in costfun and not a separate set of inputs? Hmm.
# perrange = the percent/fraction range to profile the parameter over (default is 0.5)
# numpoints = number of points to profile at in each direction (default is 10)

# Output
# A list with:
#   - profparvals: the values of the profiled parameter that were used
#   - fnvals: the cost function value at each profiled parameter value
#   - convergence: the convergence value at each profiled parameter value
#   - paramestvals: the estimates of the other parameters at each profiled parameter value

import numpy as np
import scipy.optimize as optimize
import copy

import sir_ode
import sir_cost

def proflike (params, profindex, cost_func, times, data, perrange = 0.5, numpoints = 10):
	profrangedown = np.linspace(params[profindex], params[profindex] * (1 - perrange), numpoints).tolist()
	profrangeup = np.linspace(params[profindex], params[profindex] * (1 + perrange), numpoints).tolist()[1:] #skip the duplicated values
	profrange = [profrangedown, profrangeup]
	currvals = []
	currparams = []
	currflags = []

	def profcost (fit_params, profparam, profindex, data, times, cost_func):
		paramstest = fit_params.tolist()
		paramstest.insert(profindex, profparam)
		return cost_func (paramstest, data, times)

	fit_params = params.tolist() #make a copy of params so we won't change the origianl list
	fit_params.pop(profindex)
	print ('Starting profile...')
	for i in range(len(profrange)):
		for j in profrange[i]:
			print (i, j)
			optimizer = optimize.minimize(profcost, fit_params, args=(j, profindex, data, times, cost_func), method='Nelder-Mead')
			fit_params = np.abs(optimizer.x).tolist() #save current fitted params as starting values for next round
			#print optimizer.fun
			currvals.append(optimizer.fun)
			currflags.append(optimizer.success)
			currparams.append(np.abs(optimizer.x).tolist())

	#structure the return output
	profrangedown.reverse()
	out_profparam = profrangedown+profrangeup
	temp_ind = range(len(profrangedown))
	reversed(temp_ind)#.reverse()
	out_params = [currparams[i] for i in temp_ind]+currparams[len(profrangedown):]
	out_fvals = [currvals[i] for i in temp_ind]+currvals[len(profrangedown):]
	out_flags = [currflags[i] for i in temp_ind]+currflags[len(profrangedown):]
	output = {'profparam': out_profparam, 'fitparam': np.array(out_params), 'fcnvals': out_fvals, 'convergence': out_flags}
	return output