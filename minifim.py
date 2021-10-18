# Simplified FIM (Fisher information matirx) function for the SIR model
# Marisa Eisenberg (marisae@umich.edu)
# Yu-Han Kao (kaoyh@umich.edu) -7-9-17

import numpy as np
import sir_ode

from scipy.integrate import odeint as ode

def minifisher (times, params, data, delta = 0.001):
	#params = np.array(params)
	listX = []
	params_1 = np.array (params)
	params_2 = np.array (params)
	for i in range(len(params)):
		params_1[i] = params[i] * (1+delta)
		params_2[i]= params[i] * (1-delta)

		res_1 = ode(sir_ode.model, sir_ode.x0fcn(params_1,data), times, args=(params_1,))
		res_2 = ode(sir_ode.model, sir_ode.x0fcn(params_2,data), times, args=(params_2,))
		subX = (sir_ode.yfcn(res_1, params_1) - sir_ode.yfcn(res_2, params_2)) / (2 * delta * params[i])
		listX.append(subX.tolist())
	X = np.matrix(listX)
	FIM = np.dot(X, X.transpose())
	return FIM

def minifisher_new (times, params, data, delta = 0.001):
	#params = np.array(params)
	listX_new = []
	params_1 = np.array (params)
	params_2 = np.array (params)
	params_3 = np.array(params)
	for i in range(len(params)):
		params_1[i] = params[i] * (1+delta)
		params_2[i]= params[i] * (1-delta)
		params_3[i] = params[i] * (1 - delta)
		res_1_new = ode(sir_ode.model_new, sir_ode.x0fcn_new(params_1,data), times, args=(params_1,))
		res_2_new = ode(sir_ode.model_new, sir_ode.x0fcn_new(params_2,data), times, args=(params_2,))
		res_3_new = ode(sir_ode.model_new, sir_ode.x0fcn_new(params_3, data), times, args=(params_3,))
		subX_new = (sir_ode.yfcn_new(res_1_new, params_1) - sir_ode.yfcn_new(res_2_new, params_2)- sir_ode.yfcn_new(res_3_new, params_3))  / (2 * delta * params[i])
		listX_new.append(subX_new.tolist())
	X_new = np.matrix(listX_new)
	FIM_new = np.dot(X_new, X_new.transpose())
	return FIM_new
