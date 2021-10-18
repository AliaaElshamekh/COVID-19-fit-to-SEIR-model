# model equations for the scaled SIR model for python 2.7
# Marisa Eisenberg (marisae@umich.edu)
# Yu-Han Kao (kaoyh@umich.edu) -7-9-17

import numpy as np

def model(ini, time_step, params):
	Y = np.zeros(3) #column vector for the state variables
	X = ini
	mu = 0
	beta = params[0]
	gamma = params[1]
	Y[0] = mu - beta*X[0]*X[1] - mu*X[0] #S
	Y[1] = beta*X[0]*X[1] - gamma*X[1] - mu*X[1] #I
	Y[2] = gamma*X[1] - mu*X[2] #R
	return Y


def model_new(ini, time_step, params):
	Y = np.zeros(4) #column vector for the state variables
	X = ini
	mu = 0
	beta = params[0]
	gamma = params[1]
	sigma = params[2]
	Y[0] = mu - beta*X[0]*X[2] - mu*X[0] #S
	Y[1] = beta*X[0]*X[2] - gamma*X[1] - mu*X[1] #E
	Y[2] = gamma*X[1] - sigma*X[2] - mu*X[2] #I
	Y[3]= sigma*X[2]
	return Y

def x0fcn(params, data):
	S0 = 1.0 - (data[0]/params[2])
	I0 = data[0]/params[2]
	R0 = 0.0
	X0 = [S0, I0, R0]

	return X0

def x0fcn_new(params, data):
	S0 = 1.0 - (data[0]/params[3])
	E0 = 0
	I0 = data[0]/params[3]
	R0 = 0.0
	X0 = [S0,E0, I0, R0]

	return X0

def yfcn(res, params):
	return res[:,1]*params[2]

def yfcn_new(res, params):
	return res[:,2]*params[3]

def Rfcn_new(res, params):
	return res[:,3]*params[3]

def Sfcn_new(res, params):
	return res[:,0]*params[3]

def Efcn_new(res, params):
	return res[:,1]*params[3]